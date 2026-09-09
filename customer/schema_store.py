"""
customer/schema_store.py
==========================
Persisted storage for customer table schemas, split into two stores:

  pending_schema.json   — tables uploaded but not yet sent through the
                           onboarding pipeline. Cleared after a successful
                           Generate.
  committed_schema.json — every table that has ever been through Generate.
                           Grows over time, never cleared (except on
                           /api/customer/reset).

Both files are keyed by vertical, then by table_name:
  { "<vertical>": { "<table_name>": {schema...} } }

All public functions now accept explicit `pending_path` and `committed_path`
arguments so that each customer gets their own isolated files under
  customer_runtime/customers/{customer_id}/data/
rather than sharing a single global flat file.

The router resolves these paths via get_customer_paths(customer_id) and
passes them in — no global state, no cross-customer leakage.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# Per-file threading locks — prevents race conditions when two vertical
# pipelines finish simultaneously and both try to commit to the same
# pending_schema.json / committed_schema.json at the same time.
_file_locks: Dict[str, threading.Lock] = {}
_file_locks_mutex = threading.Lock()

def _get_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _file_locks_mutex:
        if key not in _file_locks:
            _file_locks[key] = threading.Lock()
        return _file_locks[key]


def _load(path: Path) -> Dict[str, Dict[str, Dict]]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            log.warning(f"[schema_store] Could not parse {path}, treating as empty")
    return {}


def _save(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Pending store ─────────────────────────────────────────────────────────────

def get_pending(vertical: str, pending_path: Path) -> Dict[str, Dict]:
    """Return {table_name: schema} of tables uploaded but not yet generated."""
    return _load(pending_path).get(vertical, {})


def upsert_pending(vertical: str, schema: Dict, pending_path: Path) -> None:
    """Add or overwrite one table's schema in the pending store."""
    with _get_lock(pending_path):
        store = _load(pending_path)
        store.setdefault(vertical, {})[schema["table_name"]] = schema
        _save(pending_path, store)


def remove_pending(vertical: str, table_name: str, pending_path: Path) -> bool:
    """Remove one table from the pending store. Returns True if it existed."""
    store = _load(pending_path)
    vtables = store.get(vertical, {})
    existed = table_name in vtables
    if existed:
        del vtables[table_name]
        _save(pending_path, store)
    return existed


def clear_pending(vertical: str, pending_path: Path) -> None:
    store = _load(pending_path)
    if vertical in store:
        store[vertical] = {}
        _save(pending_path, store)


# ── Committed store ───────────────────────────────────────────────────────────

def get_committed(vertical: str, committed_path: Path) -> Dict[str, Dict]:
    """Return {table_name: schema} of every table ever sent through Generate."""
    return _load(committed_path).get(vertical, {})


def commit_pending(vertical: str, pending_path: Path, committed_path: Path) -> Dict[str, Dict]:
    """
    Merge everything currently pending into the committed store (pending
    tables overwrite same-named committed ones), clear pending, and return
    the full merged {table_name: schema} for the vertical.

    Both files are locked together before any read so concurrent calls
    (e.g. logistics + supply_chain pipelines finishing at the same time)
    cannot interleave their read-modify-write cycles and lose each other's
    changes.
    """
    # Acquire both file locks in a deterministic order (alphabetical path)
    # to prevent deadlock if two threads ever lock them in opposite order.
    lock_a, lock_b = sorted(
        [_get_lock(pending_path), _get_lock(committed_path)],
        key=id,
    )
    with lock_a, lock_b:
        pending_store   = _load(pending_path)
        committed_store = _load(committed_path)

        vpending   = pending_store.get(vertical, {})
        vcommitted = committed_store.setdefault(vertical, {})
        vcommitted.update(vpending)

        _save(committed_path, committed_store)

        pending_store[vertical] = {}
        _save(pending_path, pending_store)

        log.info(
            f"[schema_store] commit_pending: vertical={vertical!r} "
            f"committed {len(vpending)} table(s) → {committed_path.name}"
        )
        return dict(vcommitted)


# ── Merged view ───────────────────────────────────────────────────────────────

def merged_view(vertical: str, pending_path: Path, committed_path: Path) -> Dict[str, Dict]:
    """
    committed ∪ pending for a vertical, with pending taking priority.
    Used to preview relationships before committing.
    """
    merged = dict(get_committed(vertical, committed_path))
    merged.update(get_pending(vertical, pending_path))
    return merged


def as_list(schema_dict: Dict[str, Dict]) -> List[Dict]:
    """Convert a {table_name: schema} dict into the List[Dict] shape that
    infer_relationships()/build_customer_schema() expect."""
    return list(schema_dict.values())
