"""
compiler/operators/_csv_loader.py
===================================
Data access layer for all GPL operators.

Our architecture stores mock data as individual CSV files:
  data/mock_data/{vertical}/{canonical_id}.csv
Each operator calls load_csv() with the canonical_id of the atom it needs.

Three concerns live here and nowhere else:
  1. Reading and caching CSV files (one read per canonical_id per process)
  2. Applying filter predicates [{field, op, value}]
  3. Applying aggregations (COUNT_DISTINCT / COUNT / SUM / AVG / MAX / MIN)

Operators do NOT import pandas — we use plain Python for portability and
to keep the execution namespace dependency-light. Pandas is available but
not required for these simple operations.
"""

import csv
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# Module-level CSV path root — set once by build_namespace() before any
# operator is called. Operators access data through load_csv(), not directly.
_CSV_ROOT: Optional[Path] = None

# Optional secondary root — customer mock_data/ directory.
# Set by set_mock_fallback_root(). When load_csv() finds nothing under
# _CSV_ROOT (sot_csv/) it automatically tries here before giving up.
# This means "Execute All Metrics" works out of the box even when the
# customer hasn't uploaded real SOT files yet — it uses mock data.
_MOCK_FALLBACK_ROOT: Optional[Path] = None


def set_csv_root(path: Path) -> None:
    global _CSV_ROOT
    _CSV_ROOT = Path(path)
    # Clear the cache when the root changes (e.g. between test runs)
    _load_csv_cached.cache_clear()


def set_mock_fallback_root(path: Path) -> None:
    """Set the mock_data fallback root (called once per execute() invocation)."""
    global _MOCK_FALLBACK_ROOT
    _MOCK_FALLBACK_ROOT = Path(path) if path else None


@lru_cache(maxsize=64)
def _load_csv_cached(csv_path: str) -> tuple:
    """
    Read a CSV file and return rows as a tuple of dicts (immutable for caching).
    Cached per path — each CSV is read at most once per process.
    """
    p = Path(csv_path)
    if not p.exists():
        log.warning(f"[csv_loader] CSV not found: {csv_path}")
        return ()
    with open(p, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return tuple(dict(row) for row in reader)


def load_csv(canonical_id: str) -> List[Dict]:
    """
    Load all rows for a given atom canonical_id.
    Looks for:  {_CSV_ROOT}/{vertical_prefix}/{canonical_id}.csv
    where vertical_prefix is the domain part of the canonical_id
    (e.g. "supply_chain" from "supply_chain_orders_ERP_record").

    Falls back to a flat scan of _CSV_ROOT if the vertical subdirectory
    layout doesn't match — makes the operator robust to path variations.
    """
    if _CSV_ROOT is None:
        raise RuntimeError(
            "CSV root not set. Call build_namespace(csv_root=...) before executing operators."
        )

    # Primary path: _CSV_ROOT/{vertical}/{canonical_id}.csv
    # vertical = everything before the second underscore-separated segment
    # that looks like a system name. Simplest heuristic: try subdirs first.
    direct = _CSV_ROOT / f"{canonical_id}.csv"
    if direct.exists():
        return list(_load_csv_cached(str(direct)))

    # Try one level of subdirectory (vertical name)
    for subdir in _CSV_ROOT.iterdir():
        if subdir.is_dir():
            candidate = subdir / f"{canonical_id}.csv"
            if candidate.exists():
                return list(_load_csv_cached(str(candidate)))

    # Fuzzy fallback: find the CSV whose stem best matches the canonical_id by
    # substring containment (handles cases where the file was saved without the
    # full atom suffix, e.g. "fleet_operations.csv" for
    # "logistics_fleet_operations_ERP_snapshot").
    all_csvs = list(_CSV_ROOT.glob("*.csv")) + [
        f for sub in _CSV_ROOT.iterdir() if sub.is_dir() for f in sub.glob("*.csv")
    ]
    # 1. Stem-contains match: file stem is contained in canonical_id
    for candidate in all_csvs:
        if candidate.stem and candidate.stem in canonical_id:
            log.warning(
                f"[csv_loader] Fuzzy match: '{canonical_id}' → '{candidate.name}' "
                f"(stem contained in canonical_id). Rename this file to fix."
            )
            return list(_load_csv_cached(str(candidate)))

    # 2. Canonical-contains match: canonical_id starts with stem
    for candidate in all_csvs:
        if candidate.stem and canonical_id.startswith(candidate.stem):
            log.warning(
                f"[csv_loader] Fuzzy match: '{canonical_id}' → '{candidate.name}' "
                f"(canonical_id starts with stem). Rename this file to fix."
            )
            return list(_load_csv_cached(str(candidate)))

    log.warning(f"[csv_loader] No CSV found for canonical_id='{canonical_id}' under {_CSV_ROOT}")

    # ── Mock data fallback ────────────────────────────────────────────────────
    # If sot_csv/ has nothing, try data/mock_data/{vertical}/ automatically.
    # This means Execute All Metrics works without requiring real SOT uploads.
    if _MOCK_FALLBACK_ROOT and _MOCK_FALLBACK_ROOT.exists():
        mock_direct = _MOCK_FALLBACK_ROOT / f"{canonical_id}.csv"
        if mock_direct.exists():
            log.info(f"[csv_loader] Mock fallback: '{canonical_id}' → '{mock_direct.name}'")
            return list(_load_csv_cached(str(mock_direct)))
        # Try subdirs of mock root
        for subdir in _MOCK_FALLBACK_ROOT.iterdir():
            if subdir.is_dir():
                candidate = subdir / f"{canonical_id}.csv"
                if candidate.exists():
                    log.info(f"[csv_loader] Mock fallback: '{canonical_id}' → '{candidate.name}'")
                    return list(_load_csv_cached(str(candidate)))
        # Fuzzy match within mock root
        mock_csvs = list(_MOCK_FALLBACK_ROOT.glob("*.csv")) + [
            f for sub in _MOCK_FALLBACK_ROOT.iterdir() if sub.is_dir() for f in sub.glob("*.csv")
        ]
        for candidate in mock_csvs:
            if candidate.stem and candidate.stem in canonical_id:
                log.info(f"[csv_loader] Mock fuzzy fallback: '{canonical_id}' → '{candidate.name}'")
                return list(_load_csv_cached(str(candidate)))
        for candidate in mock_csvs:
            if candidate.stem and canonical_id.startswith(candidate.stem):
                log.info(f"[csv_loader] Mock fuzzy fallback: '{canonical_id}' → '{candidate.name}'")
                return list(_load_csv_cached(str(candidate)))
        log.warning(f"[csv_loader] No CSV found in mock fallback either: {_MOCK_FALLBACK_ROOT}")

    return []


def apply_filters(rows: List[Dict], filters: Optional[List[Dict]]) -> List[Dict]:
    """
    Apply [{field, op, value}] predicates to a row list.
    Supported ops: = | != | > | < | >= | <=
    String comparison is case-insensitive and strip-whitespace-normalised.
    """
    if not filters:
        return rows

    out = rows
    for f in filters:
        field = f.get("field", "")
        op    = f.get("op", "=")
        value = str(f.get("value", "")).strip().lower()

        if op == "=":
            out = [r for r in out if _str(r.get(field)) == value]
        elif op == "!=":
            out = [r for r in out if _str(r.get(field)) != value]
        elif op == ">":
            out = [r for r in out if _num(r.get(field)) > _num(value)]
        elif op == "<":
            out = [r for r in out if _num(r.get(field)) < _num(value)]
        elif op == ">=":
            out = [r for r in out if _num(r.get(field)) >= _num(value)]
        elif op == "<=":
            out = [r for r in out if _num(r.get(field)) <= _num(value)]
        else:
            log.warning(f"[csv_loader] Unknown filter op '{op}' — skipping")

    return out


def aggregate(rows: List[Dict], agg: str, column: Optional[str]) -> float:
    """
    Aggregate a list of rows.

    agg options:
      COUNT_DISTINCT  → distinct non-null values of column (or row count if column is None)
      COUNT           → total row count
      SUM             → sum of numeric column values
      AVG             → mean of numeric column values
      MAX             → max of numeric column values
      MIN             → min of numeric column values
    """
    if agg == "COUNT":
        return float(len(rows))

    if agg == "COUNT_DISTINCT":
        if not column:
            return float(len(rows))
        vals = set(r.get(column) for r in rows if r.get(column) not in (None, ""))
        return float(len(vals))

    if not column:
        return 0.0

    values = [_num(r.get(column)) for r in rows if r.get(column) not in (None, "")]
    if not values:
        return 0.0

    if agg == "SUM":
        return float(sum(values))
    if agg == "AVG":
        return float(sum(values) / len(values))
    if agg == "MAX":
        return float(max(values))
    if agg == "MIN":
        return float(min(values))

    log.warning(f"[csv_loader] Unknown aggregation '{agg}' — returning 0.0")
    return 0.0


def dedup_latest_per_key(
    rows: List[Dict],
    sort_col: str,
    grain_keys: List[str],
) -> List[Dict]:
    """
    For state-type atoms: keep only the LATEST row per unique grain_key
    combination, determined by the highest sort_col value.

    CRITICAL: this must run BEFORE any filtering — see MEASURE_SNAPSHOT_DEDUPED
    for the full correctness explanation.

    If sort_col or grain_keys are empty/None (e.g. dimension or record tables
    that don't need dedup), returns rows unchanged.
    """
    if not sort_col or not grain_keys:
        return rows

    latest: Dict[tuple, Dict] = {}
    for r in rows:
        key      = tuple(str(r.get(k, "")) for k in grain_keys)
        sort_val = str(r.get(sort_col, ""))
        if key not in latest or sort_val > str(latest[key].get(sort_col, "")):
            latest[key] = r

    return list(latest.values())


# ── Internal helpers ──────────────────────────────────────────────────────────

def _str(v: Any) -> str:
    return str(v).strip().lower() if v is not None else ""


def _num(v: Any) -> float:
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return 0.0
