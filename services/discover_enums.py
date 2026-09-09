"""
services/discover_enums.py
===========================
Scans mock_data CSV files using DuckDB and discovers all low-cardinality
columns (≤ ENUM_THRESHOLD unique values) per atom.

Writes results to data/field_values.json:
{
  "supply_chain_orders_ERP_record.status":        ["cancelled","delivered","pending","processing","shipped"],
  "supply_chain_customers_CRM_dimension.region":  ["APAC","EMEA","AMER","EU"],
}

WHY DUCKDB:
  - Streams CSV files from disk — never loads entire file into memory
  - SQL GROUP BY queries find unique values efficiently
  - Scales to millions of rows with zero code changes
  - Each atom is its own CSV file — clean separation
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import duckdb

import core.paths as _paths

log = logging.getLogger(__name__)

ENUM_THRESHOLD = 20

_SKIP_NAME_PARTS = {
    "id", "identifier", "key", "code", "num", "number", "ref",
    "email", "mail", "phone", "url", "link",
    "date", "time", "timestamp", "at", "on",
    "created", "updated", "modified", "day", "month", "year", "week",
}

_DATE_PAT = re.compile(r"^\d{4}-\d{2}-\d{2}")
_ID_PAT   = re.compile(r"^[A-Z]{2,6}-\d+$")
_CODE_PAT = re.compile(r"^[A-Z0-9]{6,}$")


def _load_atoms() -> Dict[str, Dict]:
    if not _paths.ATOMS_PATH.exists():
        return {}
    try:
        raw     = json.loads(_paths.ATOMS_PATH.read_text(encoding="utf-8"))
        default = raw.get("_default", {})
        return {v["canonical_id"]: v for v in default.values()
                if isinstance(v, dict) and "canonical_id" in v}
    except Exception as e:
        log.error(f"discover_enums: failed to load atoms: {e}")
        return {}


def _load_existing() -> Dict[str, List]:
    if not _paths.FIELD_VALUES_PATH.exists():
        return {}
    try:
        text = _paths.FIELD_VALUES_PATH.read_text(encoding="utf-8").strip()
        return json.loads(text) if text else {}
    except Exception:
        return {}


def _col_is_categorical(col_name: str) -> bool:
    parts = set(re.split(r"[_\s]", col_name.lower()))
    return not bool(parts & _SKIP_NAME_PARTS)


def _values_are_categorical(values: List[str]) -> bool:
    if not values:
        return False
    for v in values[:10]:
        if "@" in v:                          return False
        if _DATE_PAT.match(v):                return False
        if _ID_PAT.match(v):                  return False
        if _CODE_PAT.match(v) and len(v) >= 6: return False
        if len(v) > 60:                        return False
    return True


def _skip_roles_for_atom(atom: Dict) -> set:
    return {f["name"] for f in atom.get("fields", [])
            if f.get("role") in ("measure", "primary_key", "time")}


def discover_enums(
    vertical:  Optional[str]       = None,
    atom_ids:  Optional[List[str]] = None,
) -> Dict[str, List]:
    """
    Scan CSV mock data files with DuckDB and discover enum values.

    Args:
        vertical:  If given, only scan CSVs for this vertical.
        atom_ids:  If given, only scan these specific atoms.

    Returns:
        Complete updated field_values dict (also written to field_values.json).
    """
    atoms    = _load_atoms()
    updated  = dict(_load_existing())

    # Find CSV directories to scan
    if vertical:
        search_dirs = [_paths.MOCK_DATA_DIR / vertical]
    else:
        search_dirs = (
            [d for d in _paths.MOCK_DATA_DIR.iterdir() if d.is_dir()]
            if _paths.MOCK_DATA_DIR.exists() else []
        )

    if not search_dirs:
        log.warning("discover_enums: no mock_data directories found")
        _paths.FIELD_VALUES_PATH.write_text(json.dumps(updated, indent=2), encoding="utf-8")
        return updated

    conn = duckdb.connect(database=":memory:")

    for vdir in search_dirs:
        if not vdir.exists():
            continue

        for csv_path in sorted(vdir.glob("*.csv")):
            canonical_id = csv_path.stem
            if atom_ids and canonical_id not in atom_ids:
                continue

            atom      = atoms.get(canonical_id, {})
            skip_cols = _skip_roles_for_atom(atom)

            # Get column names
            try:
                desc     = conn.execute(
                    f"SELECT * FROM read_csv_auto('{csv_path}') LIMIT 0"
                ).description or []
                all_cols = [c[0] for c in desc]
            except Exception as e:
                log.warning(f"discover_enums: cannot read {csv_path.name}: {e}")
                continue

            candidates = [c for c in all_cols
                          if c not in skip_cols and _col_is_categorical(c)]

            log.info(f"discover_enums: {canonical_id} — "
                     f"{len(all_cols)} cols, {len(candidates)} candidates")

            for col in candidates:
                try:
                    # Count distinct values first (cheap)
                    n_distinct = conn.execute(
                        f"SELECT COUNT(DISTINCT \"{col}\") "
                        f"FROM read_csv_auto('{csv_path}')"
                    ).fetchone()[0]

                    if n_distinct > ENUM_THRESHOLD:
                        log.info(f"  skip: {canonical_id}.{col} "
                                 f"({n_distinct} unique > {ENUM_THRESHOLD})")
                        continue

                    # Fetch the actual values
                    rows   = conn.execute(
                        f"SELECT DISTINCT CAST(\"{col}\" AS VARCHAR) "
                        f"FROM read_csv_auto('{csv_path}') "
                        f"WHERE \"{col}\" IS NOT NULL "
                        f"  AND CAST(\"{col}\" AS VARCHAR) != '' "
                        f"ORDER BY 1"
                    ).fetchall()
                    values = [r[0].strip() for r in rows if r[0]]

                    if not _values_are_categorical(values):
                        log.info(f"  skip: {canonical_id}.{col} "
                                 f"— values look like IDs/dates")
                        continue

                    key = f"{canonical_id}.{col}"
                    updated[key] = values
                    log.info(f"  enum: {key} = {values}")

                except Exception as e:
                    log.warning(f"  error {canonical_id}.{col}: {e}")

    conn.close()

    _paths.FIELD_VALUES_PATH.write_text(
        json.dumps(updated, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info(f"discover_enums: wrote {len(updated)} enums → {_paths.FIELD_VALUES_PATH}")
    return updated


def get_field_values() -> Dict[str, List]:
    return _load_existing()


def get_enums_for_atom(canonical_id: str) -> Dict[str, List]:
    prefix = f"{canonical_id}."
    return {k.replace(prefix, ""): v
            for k, v in _load_existing().items()
            if k.startswith(prefix)}
