"""
compiler/branch_a.py
======================
Branch A — Algebraic Compiler ($0.00, ~50ms per goal).

Handles goals whose formula can be derived purely from the canonical_id +
atom definition. No LLM involved anywhere in this branch.

Entry point: compile_goal(goal, atom, field_values, csv_root)

Returns one of:
  {"status": "compiled",  "aterm": {...}}   — success, aterm ready to persist
  {"status": "routed",    "route": "ROUTE_COMPOSITION" | "ROUTE_LLM_WIZARD"}
                                            — not for Branch A, caller handles
  {"status": "failed",    "reason": str}    — Branch A attempted but failed

WHAT BRANCH A HANDLES (from decision_tree.py):
  MEASURE                   — primitive totals, counts, max/min on any table
  MEASURE_SNAPSHOT_DEDUPED  — filtered/unfiltered counts on state tables
  MEASURE_MONTH_FIXED       — this_month / last_month
  MEASURE_QUARTER_FIXED     — this_quarter / last_quarter / this_year / last_year
  MEASURE_YTD               — year to date
  MEASURE_GROUPED           — cross-table (Wave 7) and series (Wave 8)

VERIFICATION:
  For FILTERED goals (state != all), Branch A proves correctness via
  complement identity:
    count_X + count_not_X == count_all
  If this holds → ai_locked: true, verify_reason: PROVEN_complement
  If it doesn't → ai_locked: false, verify_reason: VERIFY_execution_only

  For all other goals → VERIFY_execution (oracle exists, range checked)
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from compiler.decision_tree import (
    select_operator,
    ROUTE_COMPOSITION, ROUTE_LLM_WIZARD,
    OPERATOR_MEASURE, OPERATOR_MEASURE_SNAPSHOT_DEDUPED,
    OPERATOR_MEASURE_MONTH_FIXED, OPERATOR_MEASURE_QUARTER_FIXED,
    OPERATOR_MEASURE_YTD, OPERATOR_MEASURE_GROUPED,
)
from compiler.operators.gpl_operators import (
    build_namespace,
    MEASURE, MEASURE_SNAPSHOT_DEDUPED,
    MEASURE_MONTH_FIXED, MEASURE_QUARTER_FIXED,
    MEASURE_YTD, MEASURE_GROUPED,
)
from compiler.slot_constants import infer_aggregation
from compiler._shared import (
    pk_col as _pk_column,
    time_col as _time_col,
    measure_col as _measure_column,
    build_filters as _build_filters,
    range_ok as _range_ok_shared,
    complement_verify,
    NOISE_STATES as _NOISE_STATES,
)

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _range_ok(oracle_value: Any, unit: str, measure: str = "", scope: str = "") -> bool:
    # infer_aggregation now returns AVG for all percent/rate/margin measures,
    # so a SUM of a percent column should never reach here with unit=percent.
    # Remove the old bypass that silently accepted nonsense values like 2554%.
    return _range_ok_shared(oracle_value, unit)


def _join_info(goal: Dict) -> Optional[Dict]:
    j = goal.get("join")
    if not j:
        return None
    if j.get("join_atom_canonical_id") and j.get("join_from_field") and j.get("join_to_field"):
        return j
    return None


def _group_by_info(goal: Dict, slots: Dict, atom: Dict) -> Dict:
    scope  = slots.get("scope", "total")
    series = slots.get("series", "scalar")

    axis = ""
    if scope.startswith("by_"):
        axis = scope[3:]
    elif series not in ("scalar", ""):
        axis = series[3:] if series.startswith("by_") else series

    if not axis:
        return {}

    TIME_GRANULARITIES = {"month", "quarter", "week", "year", "day"}
    if axis in TIME_GRANULARITIES:
        date_col = _time_col(atom)
        if date_col:
            return {"time_bucket": {"date_col": date_col, "granularity": axis}}
        return {}

    return {"group_by_col": axis}


def compile_goal(
    goal: Dict,
    atom: Dict,
    field_values: Dict,
    csv_root: str,
) -> Dict:
    """
    Attempt to compile one goal via Branch A (algebraic).

    Args:
        goal:         goal object from wave_*.json
        atom:         atom definition dict for this goal's entity
        field_values: full field_values.json content
        csv_root:     path to data/mock_data

    Returns:
        {
          "status":  "compiled" | "routed" | "failed",
          "aterm":   {...}    (only when status="compiled"),
          "route":   str      (only when status="routed"),
          "reason":  str      (only when status="failed"),
        }
    """
    cid        = goal["canonical_id"]
    slots      = goal.get("slots", {})
    complexity = goal.get("complexity", "PRIMITIVE")
    wave       = goal.get("wave")

    record_type = atom.get("record_type", "record")

    # ── Step 1: route decision ────────────────────────────────────────────────
    operator = select_operator(slots, record_type, complexity)

    if operator in (ROUTE_COMPOSITION, ROUTE_LLM_WIZARD):
        return {"status": "routed", "route": operator}

    # ── Step 2: assemble operator arguments ───────────────────────────────────
    atom_cid   = atom.get("canonical_id", "")
    measure    = slots.get("measure", "count")
    time_scope = slots.get("time", "all_time")
    agg        = infer_aggregation(measure)
    col        = _measure_column(atom, measure)
    filters    = _build_filters(slots, atom, field_values)

    try:
        # Operator-specific argument assembly + execution
        if operator == OPERATOR_MEASURE:
            result = MEASURE(atom_cid, agg, col, filters)

        elif operator == OPERATOR_MEASURE_SNAPSHOT_DEDUPED:
            grain_keys    = atom.get("grain_keys") or [_pk_column(atom)]
            dedup_sort_col = atom.get("dedup_sort_col") or _time_col(atom) or ""
            result = MEASURE_SNAPSHOT_DEDUPED(
                atom_cid, dedup_sort_col, grain_keys, agg, col, filters
            )

        elif operator == OPERATOR_MEASURE_MONTH_FIXED:
            date_col = _time_col(atom)
            if not date_col:
                return {"status": "failed", "reason": f"No time column for {atom_cid}"}
            which = "last_month" if "last" in time_scope else "this_month"
            result = MEASURE_MONTH_FIXED(atom_cid, date_col, agg, col, which, filters)

        elif operator == OPERATOR_MEASURE_QUARTER_FIXED:
            date_col = _time_col(atom)
            if not date_col:
                return {"status": "failed", "reason": f"No time column for {atom_cid}"}
            result = MEASURE_QUARTER_FIXED(atom_cid, date_col, agg, col, time_scope, filters)

        elif operator == OPERATOR_MEASURE_YTD:
            date_col = _time_col(atom)
            if not date_col:
                return {"status": "failed", "reason": f"No time column for {atom_cid}"}
            result = MEASURE_YTD(atom_cid, date_col, agg, col, filters)

        elif operator == OPERATOR_MEASURE_GROUPED:
            join  = _join_info(goal)
            group = _group_by_info(goal, slots, atom)

            if not group:
                return {
                    "status": "failed",
                    "reason": f"Could not determine grouping axis for {cid}"
                }

            kwargs: Dict[str, Any] = {"filters": filters}
            kwargs.update(group)

            if join:
                kwargs["join_atom_canonical_id"] = join["join_atom_canonical_id"]
                kwargs["join_from_field"]         = join["join_from_field"]
                kwargs["join_to_field"]           = join["join_to_field"]

            result = MEASURE_GROUPED(atom_cid, agg, col, **kwargs)

        else:
            return {"status": "routed", "route": ROUTE_LLM_WIZARD}

    except Exception as e:
        log.error(f"[branch_a] Operator execution failed for {cid}: {e}")
        return {"status": "failed", "reason": str(e)}

    oracle_value = result.get("oracle_value")
    formula_line = result.get("formula_line", "")

    # ── Step 3: range validation ──────────────────────────────────────────────
    unit    = slots.get("unit", "count")
    measure = slots.get("measure", "")
    scope   = slots.get("scope", "")
    if not _range_ok(oracle_value, unit, measure, scope):
        return {
            "status": "failed",
            "reason": f"Oracle value {oracle_value} failed range check for unit={unit}"
        }

    # ── Step 4: complement identity verification (filtered COUNT goals only) ──
    verify_reason  = "VERIFY_execution"
    ai_locked      = False
    state          = slots.get("state", "all")
    is_filtered    = state not in _NOISE_STATES and state
    is_count       = measure == "count"
    complement_ok  = None

    time_scope     = slots.get("time", "all_time")
    is_time_scoped = time_scope not in ("all_time", "alltime", "", None)

    if is_filtered and is_count and isinstance(oracle_value, float) and oracle_value > 0 and not is_time_scoped:
        try:
            complement_ok = _verify_complement(
                operator, atom_cid, atom, agg, col, filters, oracle_value, field_values
            )
            if complement_ok:
                verify_reason = "PROVEN_complement"
                ai_locked     = True
            else:
                verify_reason = "COMPLEMENT_FAILED"
                log.warning(f"[branch_a] Complement identity failed for {cid}")
        except Exception as e:
            log.warning(f"[branch_a] Complement check error for {cid}: {e}")
            verify_reason = "VERIFY_execution"
    elif not is_filtered:
        # Unfiltered goals can't be complement-proved but are structurally trusted
        verify_reason = "VERIFY_execution"
        ai_locked     = True

    # Grouped goals return a dict — structurally verified but not complement-proved
    if isinstance(oracle_value, dict):
        verify_reason = "VERIFY_grouped_execution"
        ai_locked     = True

    # ── Step 5: assemble aterm ────────────────────────────────────────────────
    aterm = {
        "id":             f"aterm:{cid}",
        "type":           "Aterm",
        "canonical_id":   cid,
        "kind":           "operator",
        "version":        1,
        "slots":          slots,
        "oracle_value":   oracle_value,
        "formula_line":   formula_line,
        "verified":       complement_ok is not False,
        "ai_locked":      ai_locked,
        "lock_method":    "combinatoric_algebraic",
        "verify_reason":  verify_reason,
        "created_at":     _now(),
        "usage_count":    0,
        "source_goal":    goal.get("goal", cid),
        "wave":           wave,
        "operator":       operator,
    }

    return {"status": "compiled", "aterm": aterm}


def _verify_complement(
    operator: str,
    atom_cid: str,
    atom: Dict,
    agg: str,
    col: Optional[str],
    filters: List[Dict],
    oracle_x: float,
    field_values: Dict,
) -> bool:
    """
    Complement identity: count_X + count_not_X == count_all
    Runs three executions and checks the identity.
    Only valid for filtered COUNT goals with oracle_x > 0.
    """
    # Get not-X filter
    not_filters = [dict(f, op="!=") for f in filters]

    try:
        if operator == OPERATOR_MEASURE:
            r_not = MEASURE(atom_cid, agg, col, not_filters)
            r_all = MEASURE(atom_cid, agg, col, None)
        elif operator == OPERATOR_MEASURE_SNAPSHOT_DEDUPED:
            grain = atom.get("grain_keys") or []
            sort  = atom.get("dedup_sort_col") or ""
            r_not = MEASURE_SNAPSHOT_DEDUPED(atom_cid, sort, grain, agg, col, not_filters)
            r_all = MEASURE_SNAPSHOT_DEDUPED(atom_cid, sort, grain, agg, col, None)
        else:
            return False

        count_not_x = float(r_not["oracle_value"])
        count_all   = float(r_all["oracle_value"])

        # Allow 1% tolerance for floating point / dedup edge cases
        total    = oracle_x + count_not_x
        expected = count_all
        if expected == 0:
            return False
        pct_diff = abs(total - expected) / expected
        return pct_diff < 0.01

    except Exception as e:
        log.warning(f"[branch_a] Complement execution error: {e}")
        return False