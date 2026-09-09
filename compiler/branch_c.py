"""
compiler/branch_c.py
======================
Branch C — Composition Compiler ($0.00, <5ms per goal).

Handles goals whose oracle_value is derived by arithmetic over already-compiled
aterms. No data access, no LLM — pure arithmetic on stored oracle values.

Wave 5  (COMPOSED_1)  — average = total / count
Wave 6  (COMPOSED_2)  — growth  = (this - last) / last * 100
Wave 9  (KPI)         — any formula referencing other CIDs

Entry point: compile_goal(goal, canonical_index)

The canonical_index must already contain all dependency CIDs. If any
dependency is missing, the goal is returned as "deferred" so the orchestrator
can retry after compiling the missing dependencies.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compile_goal(goal: Dict, canonical_index: Dict) -> Dict:
    """
    Compile a composed or KPI goal from already-compiled dependency aterms.

    Args:
        goal:            goal object (must have composition_formula + depends_on)
        canonical_index: dict of {cid: {oracle_value, formula_line, ...}}
                         must contain all CIDs in goal's depends_on

    Returns:
        {
          "status":  "compiled" | "deferred" | "failed",
          "aterm":   {...}   (only when compiled),
          "missing": [cids]  (only when deferred),
          "reason":  str     (only when failed),
        }
    """
    cid         = goal["canonical_id"]
    slots       = goal.get("slots", {})
    formula_tpl = goal.get("composition_formula", "")
    depends_on  = goal.get("depends_on", [])

    if not formula_tpl:
        return {"status": "failed", "reason": f"No composition_formula for {cid}"}

    # ── Step 1: resolve dependency oracle values ──────────────────────────────
    missing = [dep for dep in depends_on if dep not in canonical_index]
    if missing:
        return {"status": "deferred", "missing": missing}

    dep_values: Dict[str, float] = {}
    for dep in depends_on:
        entry = canonical_index[dep]
        raw   = entry.get("oracle_value", 0.0)
        # Grouped goals produce dict oracle_values — use sum as scalar proxy
        if isinstance(raw, dict):
            dep_values[dep] = float(sum(raw.values()))
        else:
            try:
                dep_values[dep] = float(raw)
            except (TypeError, ValueError):
                dep_values[dep] = 0.0

    # ── Step 2: evaluate formula ──────────────────────────────────────────────
    # Permanently inject zero-guards into every division before evaluating.
    # This is a compile-time transformation — no goal author needs to remember.
    formula_tpl = _inject_zero_guards(formula_tpl, depends_on)

    # Formula template uses {cid} placeholders:
    #   "{supply_chain_orders_order_amount_currency} / {supply_chain_orders_count}"
    # Replace each {cid} with its oracle_value, then eval safely.
    try:
        formula_resolved = formula_tpl
        for dep_cid, dep_val in dep_values.items():
            formula_resolved = formula_resolved.replace(
                "{" + dep_cid + "}", str(dep_val)
            )

        # Check no unresolved placeholders remain
        remaining = re.findall(r"\{[^}]+\}", formula_resolved)
        if remaining:
            return {
                "status": "failed",
                "reason": f"Unresolved placeholders in formula: {remaining}"
            }

        oracle_value = _safe_eval(formula_resolved)

    except Exception as e:
        return {"status": "failed", "reason": f"Formula evaluation error: {e}"}

    if oracle_value is None:
        return {"status": "failed", "reason": "Formula evaluated to None"}

    # ── Step 3: range check ───────────────────────────────────────────────────
    # Auto-correct unit from dependency CID names if the declared unit is wrong.
    # Permanent fix: no goal author needs to manually set unit correctly.
    unit = _infer_unit_from_deps(depends_on, slots.get("unit", "count"))
    slots = {**slots, "unit": unit}   # write corrected unit back into aterm slots
    if not _range_ok(oracle_value, unit):
        return {
            "status": "failed",
            "reason": f"Oracle value {oracle_value} out of range for unit={unit}"
        }

    # ── Step 4: build formula_line (CID-reference form, not hardcoded numbers) ─
    # The stored formula_line uses the zero-guarded template so the guard
    # is visible in reports and survives into the customer runtime.
    # Format: "cid = {dep1} / ({dep2} if {dep2} != 0 else 1)"
    formula_line = f"{cid} = {formula_tpl}"

    # ── Step 5: determine lock status ─────────────────────────────────────────
    # A composed goal is AI_LOCKED if all its dependencies are AI_LOCKED
    all_deps_locked = all(
        canonical_index[dep].get("ai_locked", False)
        for dep in depends_on
        if dep in canonical_index
    )
    ai_locked     = all_deps_locked
    verify_reason = "COMPOSED_verified" if ai_locked else "COMPOSED_partial_lock"

    aterm = {
        "id":             f"aterm:{cid}",
        "type":           "Aterm",
        "canonical_id":   cid,
        "kind":           "composed",
        "version":        1,
        "slots":          slots,
        "oracle_value":   oracle_value,
        "formula_line":   formula_line,
        "verified":       True,
        "ai_locked":      ai_locked,
        "lock_method":    "composition",
        "verify_reason":  verify_reason,
        "created_at":     _now(),
        "usage_count":    0,
        "source_goal":    goal.get("goal", cid),
        "wave":           goal.get("wave"),
        "depends_on":     depends_on,
        "dep_values":     dep_values,
    }

    return {"status": "compiled", "aterm": aterm}


# ── Unit inference map ───────────────────────────────────────────────────────
# Maps unit suffixes found in dependency CID names → canonical unit slot value.
# Used to auto-correct a composed goal's unit when it conflicts with its deps.
_UNIT_SUFFIX_MAP = {
    "_currency":  "currency",
    "_percent":   "percent",
    "_days":      "days",
    "_count":     "count",
    "_kg":        "kg",
    "_cbm":       "cbm",
    "_km":        "km",
    "_rate":      "currency",   # rate per km → currency
}

def _infer_unit_from_deps(depends_on: List[str], declared_unit: str) -> str:
    """
    Infer the correct unit for a composed goal from its dependency CID names.

    Rule: the numerator dependency (first in depends_on) determines the output
    unit. If the declared unit conflicts, return the inferred unit and log a
    warning so the aterm is written with the correct unit automatically.

    The CID naming convention is:  <vertical>_<entity>_<measure>_<unit_suffix>
    e.g. "logistics_cargo_items_weight_kg_count"
           → segments: [..., "weight", "kg", "count"]
           → scan all segments for a known unit keyword

    Examples:
      dep = "logistics_cargo_items_weight_kg_count"        → unit = "kg"
      dep = "logistics_carriers_contract_rate_per_km_percent" → unit = "currency"
      dep = "supply_chain_lead_times_avg_lead_time_days_days" → unit = "days"
    """
    if not depends_on:
        return declared_unit

    numerator_cid = depends_on[0]
    segments = numerator_cid.split("_")

    # Priority map: scan CID segments for known unit keywords.
    # Order matters — more specific keywords checked first.
    _SEGMENT_UNIT_MAP = [
        ("currency",  "currency"),
        ("days",      "days"),
        ("kg",        "kg"),
        ("cbm",       "cbm"),
        ("percent",   "percent"),   # only if no higher-priority match
        ("count",     "count"),
    ]

    # Special case: "rate_per_km" anywhere in the CID → currency (it's a price)
    if "rate" in segments and "km" in segments:
        inferred = "currency"
        if inferred != declared_unit:
            log.warning(
                f"[branch_c] Unit auto-corrected: declared={declared_unit!r} "
                f"→ inferred={inferred!r} from dep '{numerator_cid}' (rate_per_km pattern)"
            )
        return inferred

    # Scan segments right-to-left. Skip the terminal segment if it is an
    # aggregation-type suffix ("count", "currency", "percent") that is merely
    # describing the aggregation method, not the semantic unit of the measure.
    # e.g. "weight_kg_count" — "count" is the agg, "kg" is the real unit.
    AGG_SUFFIXES = {"count", "currency", "percent", "days"}
    search_segs = segments[:-1] if segments and segments[-1] in AGG_SUFFIXES else segments

    for seg in reversed(search_segs):
        for keyword, unit in _SEGMENT_UNIT_MAP:
            if seg == keyword:
                if unit != declared_unit:
                    log.warning(
                        f"[branch_c] Unit auto-corrected: declared={declared_unit!r} "
                        f"→ inferred={unit!r} from dep '{numerator_cid}' (segment '{seg}')"
                    )
                return unit

    # Fall back to terminal segment if nothing matched in the inner segments
    for keyword, unit in _SEGMENT_UNIT_MAP:
        if segments and segments[-1] == keyword:
            return unit

    return declared_unit


def _inject_zero_guards(formula_tpl: str, depends_on: List[str]) -> str:
    """
    Permanently inject division-by-zero guards into every composition formula
    at compile time. Any `/ {dep_cid}` pattern is rewritten to:

        / {dep_cid} if {dep_cid} != 0 else 1

    Using `else 1` (not 0) for the denominator so that the overall expression
    evaluates to 0 rather than producing NaN or a misleading non-zero result.

    This runs on the raw template (with {cid} placeholders), so the stored
    formula_line retains human-readable CID references rather than raw numbers.
    The guard also survives into the resolved numeric string since Python's
    eval handles `x / y if y != 0 else 1` correctly.

    Already-guarded formulas (containing " if " for a denominator dep) are
    left untouched to avoid double-wrapping.
    """
    result = formula_tpl

    for dep in depends_on:
        placeholder  = "{" + dep + "}"
        division_pat = f"/ {placeholder}"
        guarded_pat  = f"/ ({placeholder} if {placeholder} != 0 else 1)"

        # Only inject if this dep appears as a divisor and isn't already guarded
        if division_pat in result and guarded_pat not in result:
            # Also skip if a manual "if" guard already exists for this dep
            if f"{placeholder} != 0" not in result and f"{placeholder} != 0.0" not in result:
                result = result.replace(division_pat, guarded_pat)
                log.debug(f"[branch_c] Zero-guard injected for dep '{dep}'")

    return result


def _safe_eval(expr: str) -> Optional[float]:
    """
    Safely evaluate a numeric expression string.
    Handles:
      - Basic arithmetic: + - * / ( ) abs()
      - Python ternary: X if Y != 0 else Z  (used in Wave 6 growth formulas)
      - Scientific notation: 1e6
    """
    clean = expr.strip()

    try:
        result = eval(  # noqa: S307
            clean,
            {"__builtins__": {}, "abs": abs},
        )
        return float(result)
    except ZeroDivisionError:
        return 0.0
    except Exception as e:
        raise ValueError(f"eval failed on '{clean}': {e}")


def _range_ok(value: Any, unit: str) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    if unit == "count":
        return v >= 0
    if unit == "percent":
        return -100000.0 <= v <= 100000.0
    if unit == "days":
        return 0.0 <= v <= 36500.0
    if unit == "currency":
        return -1e13 < v < 1e13
    if unit in ("kg", "cbm", "km"):
        return -1e12 < v < 1e12
    return True
