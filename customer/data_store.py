"""
customer/data_store.py
=======================
Per-customer row store — tracks every data row for every atom.

One JSON file per customer:
  customer_runtime/customers/{customer_id}/data/data_store.json

Structure:
  {
    "supply_chain_shipments_wms_record": {
      "atom_id":      "supply_chain_shipments_wms_record",
      "primary_key":  "shipment_id",
      "total_rows":   40,
      "last_updated": "2026-08-14T10:00:00Z",
      "rows": [
        {
          "shipment_id": "SHP001",
          "carrier":     "DHL",
          ...
          "_meta": {
            "added_by_file":    "upload_001",
            "added_at":         "2026-08-14T10:00:00Z",
            "last_updated_at":  "2026-08-14T10:00:00Z",
            "updated_by_file":  null
          }
        }
      ]
    }
  }

On re-upload (same atom, new file):
  - New primary key   → row added
  - Existing key, data changed → row updated
  - Existing key, data same   → row unchanged
  - Existing key missing from new file → row deleted

History is written to data_history.py on every change.
"""

import json
import hashlib
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# Per-file write lock — prevents race if two sheets finish at the same time
_write_locks: Dict[str, threading.Lock] = {}
_lock_mutex  = threading.Lock()


def _get_lock(store_path: Path) -> threading.Lock:
    key = str(store_path.resolve())
    with _lock_mutex:
        if key not in _write_locks:
            _write_locks[key] = threading.Lock()
        return _write_locks[key]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _store_path(customer_data_dir: Path) -> Path:
    """Returns path to this customer's data_store.json."""
    return customer_data_dir / "data_store.json"


def _load(store_path: Path) -> Dict[str, Any]:
    if store_path.exists():
        try:
            return json.loads(store_path.read_text(encoding="utf-8"))
        except Exception:
            log.warning(f"[data_store] Could not parse {store_path} — treating as empty")
    return {}


def _save(store_path: Path, data: Dict[str, Any]) -> None:
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def compute_column_hash(columns: List[str]) -> str:
    """
    Stable 8-char hash from a sorted list of column names.
    Same columns in any order → same hash.
    """
    fingerprint = "|".join(sorted(c.strip().lower() for c in columns if c))
    return hashlib.md5(fingerprint.encode()).hexdigest()[:8]


# ── Public API ─────────────────────────────────────────────────────────────────

def _normalise_value(v: str) -> str:
    """
    Normalise a string value for comparison purposes.
    Converts common date formats to YYYY-MM-DD so that
    '01-02-2024' and '2024-02-01' are not treated as changes.
    """
    if not isinstance(v, str):
        return v
    s = v.strip()
    # DD-MM-YYYY or DD/MM/YYYY
    import re
    m = re.fullmatch(r"(\d{2})[-/](\d{2})[-/](\d{4})", s)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    # MM-DD-YYYY or MM/DD/YYYY (ambiguous — only convert if day > 12)
    m = re.fullmatch(r"(\d{2})[-/](\d{2})[-/](\d{4})", s)
    if m and int(m.group(2)) > 12:
        return f"{m.group(3)}-{m.group(1)}-{m.group(2)}"
    return s


def _normalise_row(row: dict) -> dict:
    """Return a copy of row with all string values normalised for comparison."""
    return {k: _normalise_value(v) for k, v in row.items()}


def upsert_rows(
    customer_data_dir: Path,
    atom_id:           str,
    primary_key_field: str,
    new_rows:          List[Dict[str, Any]],
    upload_id:         str,
    column_hash:       Optional[str] = None,
) -> Dict[str, Any]:
    """
    Insert or update rows for an atom by primary key.

    Args:
        customer_data_dir:  get_customer_paths(cid)["DATA_DIR"]
        atom_id:            canonical atom id (e.g. "supply_chain_shipments_wms_record")
        primary_key_field:  column used as the unique row identifier
        new_rows:           list of row dicts (already normalised by ingest_csv)
        upload_id:          filename or unique upload label for _meta tracking
        column_hash:        optional 8-char hash of column names

    Returns:
        {
            rows_added, rows_updated, rows_unchanged, rows_deleted,
            added_keys, updated_keys, deleted_keys, total_rows
        }
    """
    store_path = _store_path(customer_data_dir)

    with _get_lock(store_path):
        store       = _load(store_path)
        existing    = store.get(atom_id)
        timestamp   = _now()

        existing_rows: List[Dict] = existing["rows"] if existing else []

        # Build lookup: pk_value → (index, row)
        existing_map: Dict[Any, tuple] = {
            row.get(primary_key_field): (idx, row)
            for idx, row in enumerate(existing_rows)
            if row.get(primary_key_field) is not None
        }

        # ── Counters & logs ────────────────────────────────────────────────────
        rows_added     = 0
        rows_updated   = 0
        rows_unchanged = 0
        added_keys:   List[str] = []
        updated_keys: List[str] = []
        additions_log: List[Dict] = []
        changes_log:   List[Dict] = []

        # Work on a mutable copy of existing rows
        merged_rows = list(existing_rows)

        for new_row in new_rows:
            pk_val = new_row.get(primary_key_field)
            if pk_val is None:
                log.warning(f"[data_store] Row missing PK '{primary_key_field}' — skipped")
                continue

            # Strip _meta if caller accidentally passed it
            row_data = {k: v for k, v in new_row.items() if k != "_meta"}

            if pk_val in existing_map:
                idx, old_row = existing_map[pk_val]
                old_data = {k: v for k, v in old_row.items() if k != "_meta"}

                if _normalise_row(old_data) != _normalise_row(row_data):
                    # Changed — update in place
                    merged_rows[idx] = {
                        **row_data,
                        "_meta": {
                            "added_by_file":   old_row.get("_meta", {}).get("added_by_file", upload_id),
                            "added_at":        old_row.get("_meta", {}).get("added_at", timestamp),
                            "last_updated_at": timestamp,
                            "updated_by_file": upload_id,
                        },
                    }
                    changes_log.append({
                        "primary_key": str(pk_val),
                        "old":         old_data,
                        "new":         row_data,
                    })
                    rows_updated += 1
                    updated_keys.append(str(pk_val))
                else:
                    rows_unchanged += 1
            else:
                # New row
                merged_rows.append({
                    **row_data,
                    "_meta": {
                        "added_by_file":   upload_id,
                        "added_at":        timestamp,
                        "last_updated_at": timestamp,
                        "updated_by_file": None,
                    },
                })
                additions_log.append({
                    "primary_key": str(pk_val),
                    "new_row":     row_data,
                })
                rows_added += 1
                added_keys.append(str(pk_val))

        # ── Delete rows missing from new file (only on re-upload) ──────────────
        rows_deleted  = 0
        deleted_keys: List[str] = []
        deletions_log: List[Dict] = []

        if existing and new_rows:
            new_pk_set = {
                row.get(primary_key_field)
                for row in new_rows
                if row.get(primary_key_field) is not None
            }
            kept = []
            for row in merged_rows:
                pk = row.get(primary_key_field)
                if pk in new_pk_set or pk is None:
                    kept.append(row)
                else:
                    deletions_log.append({
                        "primary_key":  str(pk),
                        "deleted_row":  {k: v for k, v in row.items() if k != "_meta"},
                    })
                    deleted_keys.append(str(pk))
                    rows_deleted += 1
            merged_rows = kept

        # ── Persist ────────────────────────────────────────────────────────────
        store[atom_id] = {
            "atom_id":      atom_id,
            "primary_key":  primary_key_field,
            "column_hash":  column_hash or (existing.get("column_hash") if existing else None),
            "total_rows":   len(merged_rows),
            "last_updated": timestamp,
            "rows":         merged_rows,
        }
        _save(store_path, store)

    # ── Record history (outside lock — append-only file) ──────────────────────
    if rows_added or rows_updated or rows_deleted:
        from customer.data_history import record_upload
        record_upload(
            customer_data_dir = customer_data_dir,
            upload_id         = upload_id,
            atom_id           = atom_id,
            rows_added        = rows_added,
            rows_updated      = rows_updated,
            rows_unchanged    = rows_unchanged,
            rows_deleted      = rows_deleted,
            additions         = additions_log,
            changes           = changes_log,
            deletions         = deletions_log,
        )

    result = {
        "rows_added":     rows_added,
        "rows_updated":   rows_updated,
        "rows_unchanged": rows_unchanged,
        "rows_deleted":   rows_deleted,
        "added_keys":     added_keys,
        "updated_keys":   updated_keys,
        "deleted_keys":   deleted_keys,
        "total_rows":     len(merged_rows),
        "is_first_upload": existing is None,
    }

    log.info(
        f"[data_store] '{atom_id}': "
        f"+{rows_added} added, ~{rows_updated} updated, "
        f"={rows_unchanged} unchanged, -{rows_deleted} deleted "
        f"| total={len(merged_rows)}"
    )
    return result


def get_atom_info(customer_data_dir: Path, atom_id: str) -> Optional[Dict[str, Any]]:
    """Return metadata (total_rows, last_updated, primary_key) for one atom."""
    store = _load(_store_path(customer_data_dir))
    entry = store.get(atom_id)
    if not entry:
        return None
    return {
        "atom_id":      entry["atom_id"],
        "primary_key":  entry.get("primary_key"),
        "column_hash":  entry.get("column_hash"),
        "total_rows":   entry.get("total_rows", 0),
        "last_updated": entry.get("last_updated"),
    }


def list_atoms(customer_data_dir: Path) -> List[str]:
    """List all atom_ids that have data for this customer."""
    return list(_load(_store_path(customer_data_dir)).keys())


def find_atom_by_column_hash(customer_data_dir: Path, column_hash: str) -> Optional[str]:
    """
    If the same column structure was uploaded before, return its atom_id.
    Used to skip re-detection when the schema hasn't changed.
    """
    store = _load(_store_path(customer_data_dir))
    for atom_id, entry in store.items():
        if entry.get("column_hash") == column_hash:
            return atom_id
    return None
