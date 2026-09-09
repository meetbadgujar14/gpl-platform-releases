"""
compiler/branch_b.py
======================
Branch B — compiler for Wave A-E goals.

All goals reaching Branch B go directly to the LLM wizard (Phase 5).
No deterministic attempt is made — the wizard always runs.

Why: Wave A-E goals are complex natural language business questions.
A deterministic path can produce mathematically valid but semantically
wrong formulas (e.g. SUM instead of MAX for "highest X" goals) that
pass all mathematical verification checks silently. The wizard, which
reasons about the goal text at each constrained step, produces more
semantically correct formulas.

The wizard (5-8 constrained LLM calls):
  5a table → 5b aggregation → 5c column → 5d filter field →
  5e filter value → 5f date column → 5g time scope → 5h support

After wizard: zero-LLM assembler builds formula from wizard_state.

Inline verification:
  - Strict type range check (percent [0,110] catches oracle=200%)
  - Complement identity for filtered COUNT goals → ai_locked=True if proven
  - exec_formula re-execution for wizard aterms → VERIFY_wizard_execution
"""

import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from anthropic import Anthropic
from core.config import settings
from compiler._shared import (
    pk_col, time_col, measure_col, build_filters,
    state_resolvable, range_ok, complement_verify, NOISE_STATES,
)
from compiler.operators.gpl_operators import (
    MEASURE, MEASURE_SNAPSHOT_DEDUPED,
    MEASURE_MONTH_FIXED, MEASURE_QUARTER_FIXED,
    MEASURE_YTD, MEASURE_GROUPED,
    build_namespace,
)
from compiler.slot_constants import infer_aggregation
from compiler.wizard_steps import run_wizard
from compiler.wizard_assembler import assemble as wizard_assemble

log = logging.getLogger(__name__)

def _get_client():
    """Build a fresh client per call so a runtime API-key update takes effect immediately."""
    return Anthropic(api_key=settings.ANTHROPIC_API_KEY)


def _get_model():
    return settings.ANTHROPIC_MODEL

_TIME_OPERATOR = {
    "this_month":   ("MEASURE_MONTH_FIXED",   "this_month"),
    "last_month":   ("MEASURE_MONTH_FIXED",   "last_month"),
    "this_quarter": ("MEASURE_QUARTER_FIXED", "this_quarter"),
    "last_quarter": ("MEASURE_QUARTER_FIXED", "last_quarter"),
    "this_year":    ("MEASURE_QUARTER_FIXED", "this_year"),
    "last_year":    ("MEASURE_QUARTER_FIXED", "last_year"),
    "ytd":          ("MEASURE_YTD",           None),
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Ambiguity detection ────────────────────────────────────────────────────────

# Linguistic signals — only applied when state=all (noise).
# When state resolves to a known enum value, the computation is fully
# determined by the slots regardless of what words appear in the goal text.
# "orders at risk" with state=On Hold is just COUNT WHERE status='On Hold'.

_THRESHOLD_SIGNALS = frozenset({
    "threshold", "exceed", "exceeding", "concentration",
    "disproportionate", "high-value", "high value", "reorder",
    "outlier", "anomaly", "spike", "surge",
})

_CONCEPT_SIGNALS = frozenset({
    "active state", "terminal state", "terminal orders",
    "open state", "closed state",
    "good standing", "bad standing", "positive state", "negative state",
})


def _is_ambiguous(
    goal:          Dict,
    atom:          Dict,
    field_values:  Dict,
    atoms_all:     Dict[str, Dict],
    relationships: List[Dict],
) -> bool:
    """
    Determine if a goal requires LLM reasoning to produce its formula.

    DETERMINISTIC when slots + atom definition give everything needed.
    AMBIGUOUS (needs LLM) in three structural cases:

    1. CROSS-ENTITY WITHOUT JOIN — goal text references another entity and
       no FK joins them in atom_relationships.json. Always checked.

    2. THRESHOLD / CONCEPT — only checked when state=all (noise).
       When state resolves to a known enum value, the slots fully define
       the computation — contextual language in the goal text is irrelevant.

    3. UNRESOLVABLE STATE — state is non-noise but not in field_values.
    """
    slots     = goal.get("slots", {})
    state     = slots.get("state", "all")
    domain    = slots.get("domain", "").lower()
    goal_text = goal.get("goal", "").lower()

    # ── Rule 1: Cross-entity without a known FK join ─────────────────────────
    primary_cid = atom.get("canonical_id", "")
    known_joins = {
        (r["from_atom"], r["to_atom"])
        for r in relationships
    } | {
        (r["to_atom"], r["from_atom"])
        for r in relationships
    }

    for other_cid, other_atom in atoms_all.items():
        if other_cid == primary_cid:
            continue
        other_entity = other_atom.get("entity_name", "")
        if not other_entity:
            parts = other_cid.replace(domain + "_", "", 1).split("_")
            other_entity = parts[0] if parts else ""

        if other_entity and other_entity.lower() in goal_text:
            if (primary_cid, other_cid) not in known_joins:
                log.debug(f"[branch_b] AMBIGUOUS (cross-entity no FK): {primary_cid} ↔ {other_cid}")
                return True

    # ── State resolves → always deterministic beyond this point ──────────────
    if state not in NOISE_STATES and state_resolvable(state, atom, field_values):
        return False

    # ── Rule 2: state=all + threshold or concept signals ─────────────────────
    if any(signal in goal_text for signal in _THRESHOLD_SIGNALS):
        log.debug(f"[branch_b] AMBIGUOUS (threshold, state=all): {goal_text[:60]}")
        return True

    if any(signal in goal_text for signal in _CONCEPT_SIGNALS):
        log.debug(f"[branch_b] AMBIGUOUS (concept, state=all): {goal_text[:60]}")
        return True

    # ── Rule 3: Unresolvable non-noise state ─────────────────────────────────
    if state not in NOISE_STATES:
        log.debug(f"[branch_b] AMBIGUOUS (unresolvable state={state!r})")
        return True

    # state=all, no signals → simple unfiltered MEASURE or MEASURE_GROUPED
    return False


# ── Deterministic formula builder ─────────────────────────────────────────────

def _build_deterministic(
    goal: Dict, atom: Dict, field_values: Dict, csv_root: str
) -> Dict:
    """
    Build and execute a formula from slots alone — no LLM.

    Returns {oracle_value, formula_line, filters} or raises ValueError.
    """
    slots    = goal["slots"]
    unit     = slots.get("unit", "count")
    time     = slots.get("time", "all_time")
    scope    = slots.get("scope", "total")
    measure  = slots.get("measure", "count")
    series   = slots.get("series", "scalar")

    atom_cid   = atom.get("canonical_id", "")
    filters    = build_filters(slots, atom, field_values)
    col        = measure_col(atom, measure)
    pk         = pk_col(atom)
    date_col   = time_col(atom)
    agg        = infer_aggregation(measure)
    time_op    = _TIME_OPERATOR.get(time)

    # ── RATE / PERCENT goals ───────────────────────────────────────────────────
    if unit == "percent":
        # count_X / count_all * 100
        if time_op:
            op_name, which = time_op
            if not date_col:
                raise ValueError(f"No time column for {atom_cid}")
            if op_name == "MEASURE_MONTH_FIXED":
                num_r = MEASURE_MONTH_FIXED(atom_cid, date_col, "COUNT_DISTINCT", pk, which, filters)
                den_r = MEASURE_MONTH_FIXED(atom_cid, date_col, "COUNT_DISTINCT", pk, which, None)
            elif op_name == "MEASURE_YTD":
                num_r = MEASURE_YTD(atom_cid, date_col, "COUNT_DISTINCT", pk, filters)
                den_r = MEASURE_YTD(atom_cid, date_col, "COUNT_DISTINCT", pk, None)
            else:
                num_r = MEASURE_QUARTER_FIXED(atom_cid, date_col, "COUNT_DISTINCT", pk, which, filters)
                den_r = MEASURE_QUARTER_FIXED(atom_cid, date_col, "COUNT_DISTINCT", pk, which, None)
        else:
            num_r = MEASURE(atom_cid, "COUNT_DISTINCT", pk, filters)
            den_r = MEASURE(atom_cid, "COUNT_DISTINCT", pk, None)

        num_v = float(num_r["oracle_value"])
        den_v = float(den_r["oracle_value"])
        oracle = round((num_v / den_v * 100) if den_v else 0.0, 6)
        formula_line = (
            f"RATIO: {num_r['formula_line']} / {den_r['formula_line']} * 100"
        )
        return {"oracle_value": oracle, "formula_line": formula_line, "filters": filters}

    # ── AVERAGE / scope=average ────────────────────────────────────────────────
    if scope == "average" or measure.startswith("average_") or measure.startswith("avg_"):
        if time_op:
            op_name, which = time_op
            if not date_col:
                raise ValueError(f"No time column for {atom_cid}")
            if op_name == "MEASURE_MONTH_FIXED":
                r = MEASURE_MONTH_FIXED(atom_cid, date_col, "AVG", col, which, filters)
            elif op_name == "MEASURE_YTD":
                r = MEASURE_YTD(atom_cid, date_col, "AVG", col, filters)
            else:
                r = MEASURE_QUARTER_FIXED(atom_cid, date_col, "AVG", col, which, filters)
        else:
            r = MEASURE(atom_cid, "AVG", col, filters)
        return {"oracle_value": r["oracle_value"], "formula_line": r["formula_line"], "filters": filters}

    # ── GROUPED / scope=by_X or series=by_X ───────────────────────────────────
    axis = ""
    if scope.startswith("by_"):
        axis = scope[3:]
    elif series not in ("scalar", ""):
        axis = series[3:] if series.startswith("by_") else series

    if axis:
        time_axes = {"month", "quarter", "week", "year", "day"}
        if axis in time_axes:
            if not date_col:
                raise ValueError(f"No time column for {atom_cid}")
            r = MEASURE_GROUPED(atom_cid, agg, col, filters=filters,
                                time_bucket={"date_col": date_col, "granularity": axis})
        else:
            r = MEASURE_GROUPED(atom_cid, agg, col, group_by_col=axis, filters=filters)
        return {"oracle_value": r["oracle_value"], "formula_line": r["formula_line"], "filters": filters}

    # ── TIME-SCOPED direct ────────────────────────────────────────────────────
    if time_op:
        op_name, which = time_op
        if not date_col:
            raise ValueError(f"No time column for {atom_cid}")
        if op_name == "MEASURE_MONTH_FIXED":
            r = MEASURE_MONTH_FIXED(atom_cid, date_col, agg, col, which, filters)
        elif op_name == "MEASURE_YTD":
            r = MEASURE_YTD(atom_cid, date_col, agg, col, filters)
        else:
            r = MEASURE_QUARTER_FIXED(atom_cid, date_col, agg, col, which, filters)
        return {"oracle_value": r["oracle_value"], "formula_line": r["formula_line"], "filters": filters}

    # ── DIRECT ────────────────────────────────────────────────────────────────
    r = MEASURE(atom_cid, agg, col, filters)
    return {"oracle_value": r["oracle_value"], "formula_line": r["formula_line"], "filters": filters}


# ── LLM Wizard ────────────────────────────────────────────────────────────────

def _build_wizard(
    goal:            Dict,
    atoms_all:       Dict[str, Dict],
    field_values:    Dict,
    canonical_index: Dict,
    csv_root:        str,
    hints:           List[Dict] = None,
) -> Dict:
    """
    8-step constrained LLM wizard.
    Replaces the old single-call _build_llm().

    Each step asks the LLM one question with a closed enum.
    The LLM cannot hallucinate — every option is derived from real schema data.

    Returns {oracle_value, formula_line, filters, wizard_state} or raises ValueError.
    """
    build_namespace(csv_root)

    # Phase 5: run all wizard steps → wizard_state
    wizard_state = run_wizard(
        goal=goal,
        atoms_all=atoms_all,
        field_values=field_values,
        canonical_index=canonical_index,
        hints=hints or [],
    )

    # Phase 6: assemble formula from wizard_state → oracle_value + formula_line
    result = wizard_assemble(
        wizard_state=wizard_state,
        goal=goal,
        csv_root=csv_root,
        canonical_index=canonical_index,
    )

    return {
        "oracle_value":  result["oracle_value"],
        "formula_line":  result["formula_line"],
        "exec_formula":  result.get("exec_formula", ""),
        "filters":       wizard_state["filter"],
        "wizard_state":  wizard_state,
        "reasoning":     f"wizard/{result['operator_used']} steps={wizard_state['steps_taken']}",
    }


# ── Inline verification ────────────────────────────────────────────────────────

def _verify(
    oracle_value: Any, formula_line: str, unit: str,
    filters: List[Dict], atom: Dict, via_llm: bool,
    exec_formula: str = "",
    csv_root: str = "",
) -> Tuple[bool, str, bool]:
    """
    Run inline verification. Returns (verified, verify_reason, ai_locked).

    Checks:
      1. Strict type range (percent [0,110], not [-100000,100000])
      2. Complement identity for filtered COUNT goals
      3. exec_formula re-execution for wizard aterms — if the executable
         Python call re-produces the oracle within 2%, set ai_locked=True
    """
    if not range_ok(oracle_value, unit):
        return False, "RANGE_FAIL", False

    # Complement identity for filtered COUNT goals
    is_count    = unit == "count" or (
        isinstance(oracle_value, float) and oracle_value == int(oracle_value)
        and unit not in ("currency", "percent", "days")
    )
    is_filtered = bool(filters)

    if is_filtered and is_count and not isinstance(oracle_value, dict):
        atom_cid = atom.get("canonical_id", "")
        grain    = atom.get("grain_keys", [pk_col(atom)])
        sort_c   = atom.get("dedup_sort_col")

        def run(f):
            if sort_c:
                return MEASURE_SNAPSHOT_DEDUPED(
                    atom_cid, sort_c, grain, "COUNT_DISTINCT", pk_col(atom), f
                )["oracle_value"]
            return MEASURE(atom_cid, "COUNT_DISTINCT", pk_col(atom), f)["oracle_value"]

        try:
            proven = complement_verify(float(oracle_value), run, filters)
            if proven:
                method = "PROVEN_complement_llm" if via_llm else "PROVEN_complement"
                return True, method, True
        except Exception:
            pass

    # ── Wizard re-execution: use exec_formula to verify and lock ──────────────
    # If we have an executable formula string (from wizard_assembler), re-run
    # it and compare to stored oracle. If it matches within 2%, ai_locked=True.
    if via_llm and exec_formula:
        try:
            ns = build_namespace(csv_root)
            exec(f"__r__ = {exec_formula}", ns)
            re_result = ns.get("__r__", {})
            re_val = re_result.get("oracle_value") if isinstance(re_result, dict) else re_result
            if re_val is not None:
                stored = float(oracle_value or 0)
                re_f   = float(re_val)
                drift  = abs((re_f - stored) / stored) if stored != 0 else (0 if re_f == 0 else 1)
                if drift <= 0.02:
                    return True, "VERIFY_wizard_execution", True
        except Exception as e:
            log.debug(f"[_verify] exec_formula re-execution failed: {e}")

    reason = "LLM_WIZARD_execution" if via_llm else "DETERMINISTIC_execution"
    return True, reason, not via_llm


# ── Entry point ────────────────────────────────────────────────────────────────

def compile_goal(
    goal:          Dict,
    atom:          Dict,
    field_values:  Dict,
    csv_root:      str,
    canonical_index: Dict,
    atoms_all:     Optional[Dict[str, Dict]] = None,
    relationships: Optional[List[Dict]] = None,
    hints:         Optional[List[Dict]] = None,
) -> Dict:
    """
    Compile one Wave A-E goal via the LLM wizard.

    All Branch B goals go directly to the wizard — no deterministic
    attempt first. This avoids silent wrong-formula compilation where
    a deterministic path produces a mathematically valid but semantically
    incorrect formula (e.g. SUM instead of MAX for a "highest X" goal).

    Args:
        goal:            goal object from pending_branch_b
        atom:            primary atom for this goal's entity
        field_values:    full field_values.json content
        csv_root:        path to data/mock_data
        canonical_index: already-compiled CIDs
        atoms_all:       all atoms in the vertical
        relationships:   atom_relationships.json content
    """
    cid  = goal["canonical_id"]
    slots= goal.get("slots", {})
    unit = slots.get("unit", "count")
    wave = goal.get("wave")

    build_namespace(csv_root)
    via_llm = True

    try:
        log.info(f"[branch_b] WIZARD: {cid}")
        result = _build_wizard(
            goal=goal,
            atoms_all=atoms_all or {},
            field_values=field_values,
            canonical_index=canonical_index,
            csv_root=csv_root,
            hints=hints or [],
        )
    except Exception as e:
        return {"status": "failed", "reason": str(e)}

    oracle_value = result["oracle_value"]
    formula_line = result["formula_line"]
    exec_formula = result.get("exec_formula", "")
    filters      = result.get("filters", [])

    # ── Override unit for composite goals ─────────────────────────────────────
    # The assembler derives the accurate unit from actual column types.
    # Write it back into slots so the aterm reflects the true unit.
    if result.get("derived_unit"):
        unit = result["derived_unit"]
        slots = dict(slots)  # don't mutate the original goal slots
        slots["unit"] = unit
        log.info(f"[branch_b] Composite unit override → {unit}")

    # ── Inline verification ───────────────────────────────────────────────────
    verified, verify_reason, ai_locked = _verify(
        oracle_value, formula_line, unit, filters, atom, via_llm,
        exec_formula=exec_formula,
        csv_root=csv_root,
    )
    if not verified:
        return {
            "status": "failed",
            "reason": f"Verification failed ({verify_reason}): oracle={oracle_value} unit={unit}",
        }

    aterm = {
        "id":            f"aterm:{cid}",
        "type":          "Aterm",
        "canonical_id":  cid,
        "kind":          "operator",
        "version":       1,
        "slots":         slots,
        "oracle_value":  oracle_value,
        "formula_line":  formula_line,
        "exec_formula":  exec_formula,
        "verified":      verified,
        "ai_locked":     ai_locked,
        "lock_method":   "llm_wizard" if via_llm else "deterministic_b",
        "verify_reason": verify_reason,
        "created_at":    _now(),
        "usage_count":   0,
        "source_goal":   goal.get("goal", cid),
        "wave":          wave,
        "via_llm":       via_llm,
        "llm_reasoning": result.get("reasoning", ""),
        "wizard_steps":  result.get("wizard_state", {}).get("steps_taken", 0) if via_llm else 0,
    }

    return {"status": "compiled", "aterm": aterm}
