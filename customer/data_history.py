"""
customer/data_history.py
=========================
Append-only upload history — one entry per upload per atom.

One JSON file per customer:
  customer_runtime/customers/{customer_id}/data/data_history.json

Structure:
  [
    {
      "upload_id":      "supply_chain_shipments.csv",
      "atom_id":        "supply_chain_shipments_wms_record",
      "timestamp":      "2026-08-14T10:30:00Z",
      "rows_added":     2,
      "rows_updated":   1,
      "rows_unchanged": 38,
      "rows_deleted":   0,
      "additions": [
        {"primary_key": "SHP0041", "new_row": {...}}
      ],
      "changes": [
        {"primary_key": "SHP0001", "old": {...}, "new": {...}}
      ],
      "deletions": []
    },
    ...
  ]

Only written when at least one row changed (adds, updates, or deletes).
First uploads are not recorded — history only tracks changes on re-uploads.
"""

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger(__name__)

_write_locks: Dict[str, threading.Lock] = {}
_lock_mutex  = threading.Lock()


def _get_lock(history_path: Path) -> threading.Lock:
    key = str(history_path.resolve())
    with _lock_mutex:
        if key not in _write_locks:
            _write_locks[key] = threading.Lock()
        return _write_locks[key]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _history_path(customer_data_dir: Path) -> Path:
    return customer_data_dir / "data_history.json"


def _load(history_path: Path) -> List[Dict]:
    if history_path.exists():
        try:
            return json.loads(history_path.read_text(encoding="utf-8"))
        except Exception:
            log.warning(f"[data_history] Could not parse {history_path} — starting fresh")
    return []


def _save(history_path: Path, data: List[Dict]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Public API ─────────────────────────────────────────────────────────────────

def record_upload(
    customer_data_dir: Path,
    upload_id:         str,
    atom_id:           str,
    rows_added:        int,
    rows_updated:      int,
    rows_unchanged:    int,
    rows_deleted:      int      = 0,
    additions:         List[Dict[str, Any]] = None,
    changes:           List[Dict[str, Any]] = None,
    deletions:         List[Dict[str, Any]] = None,
) -> None:
    """
    Append one history entry. Only called when something actually changed
    (rows_added > 0 or rows_updated > 0 or rows_deleted > 0).
    """
    if not (rows_added or rows_updated or rows_deleted):
        return  # nothing changed — no entry needed

    history_path = _history_path(customer_data_dir)
    entry = {
        "upload_id":      upload_id,
        "atom_id":        atom_id,
        "timestamp":      _now(),
        "rows_added":     rows_added,
        "rows_updated":   rows_updated,
        "rows_unchanged": rows_unchanged,
        "rows_deleted":   rows_deleted,
        "additions":      additions  or [],
        "updates":        changes    or [],
        "deletions":      deletions  or [],
    }

    with _get_lock(history_path):
        history = _load(history_path)
        history.append(entry)
        _save(history_path, history)

    log.info(
        f"[data_history] '{atom_id}' ← '{upload_id}': "
        f"+{rows_added} added, ~{rows_updated} updated, -{rows_deleted} deleted"
    )


def get_history_for_atom(
    customer_data_dir: Path,
    atom_id:           str,
) -> List[Dict[str, Any]]:
    """All history entries for one atom, newest first."""
    history = _load(_history_path(customer_data_dir))
    entries = [e for e in history if e.get("atom_id") == atom_id]
    return sorted(entries, key=lambda x: x.get("timestamp", ""), reverse=True)


def get_all_history(customer_data_dir: Path) -> List[Dict[str, Any]]:
    """All history entries across all atoms, newest first."""
    history = _load(_history_path(customer_data_dir))
    return sorted(history, key=lambda x: x.get("timestamp", ""), reverse=True)
