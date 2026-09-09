"""
compiler/_shared.py
====================
Shared helpers used by Branch A, Branch B, and the Verifier.
Single source of truth for atom field extraction, filter building,
range checks, and complement identity verification.
"""

import logging
import re
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

NOISE_STATES = {"all", "base", "total", ""}


def pk_col(atom: Dict) -> Optional[str]:
    for f in atom.get("fields", []):
        if f.get("role") == "primary_key":
            return f["name"]
    keys = atom.get("grain_keys", [])
    return keys[0] if keys else None


def time_col(atom: Dict) -> Optional[str]:
    for f in atom.get("fields", []):
        if f.get("role") == "time":
            return f["name"]
    return None


def measure_col(atom: Dict, measure: str) -> Optional[str]:
    if measure == "count":
        return pk_col(atom)
    m = measure.lower()
    for prefix in ("max_", "min_", "average_", "avg_", "growth_"):
        if m.startswith(prefix):
            m = m[len(prefix):]
            break
    for f in atom.get("fields", []):
        fname = f.get("name", "").lower()
        if fname == m or fname == measure.lower():
            return f["name"]
        if m in fname and f.get("role") == "measure":
            return f["name"]
    return pk_col(atom)


def build_filters(slots: Dict, atom: Dict, field_values: Dict) -> List[Dict]:
    """
    Build [{field, op, value}] from the state slot.
    field_values keys: "{atom_cid}.{column}" → [enum values]
    Returns [] for noise states.
    """
    state = slots.get("state", "all")
    if not state or state in NOISE_STATES:
        return []

    atom_cid    = atom.get("canonical_id", "")
    state_lower = state.lower()

    for fv_key, enum_values in field_values.items():
        if not fv_key.startswith(atom_cid + "."):
            continue
        col = fv_key.split(".", 1)[1]
        for v in enum_values:
            if str(v).lower() == state_lower:
                return [{"field": col, "op": "=", "value": str(v)}]

    # Heuristic fallback
    log.warning(f"[shared] State '{state}' not in field_values for {atom_cid}. Heuristic fallback.")
    for f in atom.get("fields", []):
        if any(kw in f.get("name", "").lower() for kw in ("status", "state", "stage", "phase")):
            return [{"field": f["name"], "op": "=", "value": state}]

    return []


def state_resolvable(state: str, atom: Dict, field_values: Dict) -> bool:
    """Return True if state maps to a known single enum value."""
    if not state or state in NOISE_STATES:
        return False
    return bool(build_filters({"state": state}, atom, field_values))


def range_ok(value: Any, unit: str) -> bool:
    """Range check by unit type. Percent uses strict [0, 110] — catches oracle=200%."""
    try:
        v = float(value) if not isinstance(value, dict) else 0.0
    except (TypeError, ValueError):
        return False
    if unit == "count":    return True          # MIN count can be negative (e.g. stock deficit)
    if unit == "percent":  return 0.0 <= v <= 110.0   # strict: rates can't exceed 100%
    if unit == "days":     return -36500.0 <= v <= 36500.0  # negative = days early/ahead
    if unit == "currency": return -1e13 < v < 1e13
    return True


def complement_verify(
    oracle_x: float,
    run_fn,   # callable(filters) → float
    filters:  List[Dict],
) -> bool:
    """
    Complement identity: count_X + count_not_X == count_all.
    run_fn(filters) must execute the same operator with different filter sets.
    Only meaningful for filtered COUNT goals with oracle_x > 0.
    Allows 1% tolerance for floating-point/dedup edge cases.
    """
    if oracle_x <= 0:
        return False
    try:
        not_filters = [dict(f, op="!=") for f in filters]
        count_not   = run_fn(not_filters)
        count_all   = run_fn([])
        if count_all == 0:
            return False
        pct_diff = abs((oracle_x + count_not) - count_all) / count_all
        return pct_diff < 0.01
    except Exception as e:
        log.warning(f"[shared] Complement verify error: {e}")
        return False
