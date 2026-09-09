"""
customer/atom_queue.py
======================
Persistent queue for unmatched-file atom creation requests.

When a customer uploads a file that has no matching atom in the factory,
the runtime enqueues the file's schema here instead of silently dropping it.
A background thread watches the queue and fires the full factory pipeline
for each entry, retrying with exponential backoff until the factory responds.

Queue file location (per customer):
    customer_runtime/customers/{id}/data/atom_creation_queue.json

Queue entry schema:
    {
        "entry_id":    str,          # uuid
        "customer_id": str,
        "table_name":  str,
        "vertical":    str,          # best-guess or "unknown"
        "columns":     [...],        # [{name, type}]
        "enums":       {...},        # {col: [values]}
        "row_count":   int,
        "queued_at":   str,          # ISO timestamp
        "attempts":    int,          # number of send attempts so far
        "last_attempt": str | None,  # ISO timestamp or None
        "status":      str,          # "pending" | "sent" | "failed"
        "error":       str | None,
    }

Background thread behaviour:
    - Wakes every POLL_INTERVAL seconds
    - Picks all "pending" entries (and "failed" ones ready for retry)
    - Posts each to POST /api/factory/customer/new_atom on the factory
    - Marks entry "sent" on success, increments attempts + backoff on failure
    - Gives up after MAX_ATTEMPTS (marks as "failed" permanently)
"""

import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ── Retry config ───────────────────────────────────────────────────────────────
POLL_INTERVAL  = 15          # seconds between queue sweeps
MAX_ATTEMPTS   = 20          # give up after this many send attempts
BASE_BACKOFF   = 30          # seconds — doubles each attempt, capped at MAX_BACKOFF
MAX_BACKOFF    = 600         # 10 minutes max between retries

QUEUE_FILENAME = "atom_creation_queue.json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _backoff_seconds(attempts: int) -> float:
    """Exponential backoff, capped at MAX_BACKOFF."""
    return min(BASE_BACKOFF * (2 ** max(attempts - 1, 0)), MAX_BACKOFF)


def _queue_path(customer_data_dir: Path) -> Path:
    return customer_data_dir / QUEUE_FILENAME


# ── Public API ─────────────────────────────────────────────────────────────────

def enqueue(
    customer_data_dir: Path,
    customer_id:       str,
    table_name:        str,
    vertical:          str,
    columns:           List[Dict[str, str]],
    enums:             Dict[str, List[str]],
    row_count:         int,
) -> str:
    """
    Add one unmatched table to the queue.
    Returns the new entry_id.
    Thread-safe via file locking pattern (read-modify-write under a lock).
    """
    entry_id = str(uuid.uuid4())
    entry: Dict[str, Any] = {
        "entry_id":    entry_id,
        "customer_id": customer_id,
        "table_name":  table_name,
        "vertical":    vertical,
        "columns":     columns,
        "enums":       enums,
        "row_count":   row_count,
        "queued_at":   _now(),
        "attempts":    0,
        "last_attempt": None,
        "status":      "pending",
        "error":       None,
    }
    _write_entry(customer_data_dir, entry)
    log.info(
        f"[atom_queue] Enqueued unmatched table '{table_name}' "
        f"for customer '{customer_id}' (entry_id={entry_id})"
    )
    return entry_id


def list_entries(customer_data_dir: Path) -> List[Dict]:
    """Return all queue entries for this customer."""
    return _load_queue(customer_data_dir)


def pending_count(customer_data_dir: Path) -> int:
    """How many entries are still pending or retrying."""
    return sum(
        1 for e in _load_queue(customer_data_dir)
        if e["status"] in ("pending", "failed") and e["attempts"] < MAX_ATTEMPTS
    )


# ── Internal queue I/O ─────────────────────────────────────────────────────────

_file_locks: Dict[str, threading.Lock] = {}
_file_locks_mutex = threading.Lock()


def _get_file_lock(path: Path) -> threading.Lock:
    key = str(path)
    with _file_locks_mutex:
        if key not in _file_locks:
            _file_locks[key] = threading.Lock()
        return _file_locks[key]


def _load_queue(customer_data_dir: Path) -> List[Dict]:
    p = _queue_path(customer_data_dir)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning(f"[atom_queue] Could not read queue file {p}: {e}")
        return []


def _save_queue(customer_data_dir: Path, entries: List[Dict]) -> None:
    p = _queue_path(customer_data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_entry(customer_data_dir: Path, entry: Dict) -> None:
    """Append or update one entry in the queue (keyed by entry_id)."""
    lock = _get_file_lock(_queue_path(customer_data_dir))
    with lock:
        entries = _load_queue(customer_data_dir)
        for i, e in enumerate(entries):
            if e["entry_id"] == entry["entry_id"]:
                entries[i] = entry
                break
        else:
            entries.append(entry)
        _save_queue(customer_data_dir, entries)


def _load_pending(customer_data_dir: Path) -> List[Dict]:
    """
    Return entries that are ready to be sent right now.
    An entry is ready if:
      - status == "pending"  (never tried), OR
      - status == "failed" AND attempts < MAX_ATTEMPTS AND backoff elapsed
    """
    now = time.time()
    ready = []
    for e in _load_queue(customer_data_dir):
        if e["status"] == "sent":
            continue
        if e["attempts"] >= MAX_ATTEMPTS:
            continue
        if e["status"] == "pending":
            ready.append(e)
        elif e["status"] == "failed":
            # Check backoff window
            last = e.get("last_attempt")
            if last:
                try:
                    last_ts = datetime.strptime(last, "%Y-%m-%dT%H:%M:%SZ").replace(
                        tzinfo=timezone.utc
                    ).timestamp()
                    wait = _backoff_seconds(e["attempts"])
                    if now - last_ts >= wait:
                        ready.append(e)
                except Exception:
                    ready.append(e)  # parse error — just retry
            else:
                ready.append(e)
    return ready


# ── Background worker ──────────────────────────────────────────────────────────

class AtomQueueWorker:
    """
    Background thread that drains the atom creation queue for ONE customer.
    One worker per customer, started by the runtime on startup and on first upload.
    """

    def __init__(self, customer_id: str, customer_data_dir: Path):
        self.customer_id       = customer_id
        self.customer_data_dir = customer_data_dir
        self._stop_event       = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._loop,
            name=f"atom-queue-{self.customer_id}",
            daemon=True,
        )
        self._thread.start()
        log.info(f"[atom_queue] Worker started for customer '{self.customer_id}'")

    def stop(self) -> None:
        self._stop_event.set()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._sweep()
            except Exception as e:
                log.error(f"[atom_queue] Sweep error for customer '{self.customer_id}': {e}")
            self._stop_event.wait(POLL_INTERVAL)

    def _sweep(self) -> None:
        entries = _load_pending(self.customer_data_dir)
        if not entries:
            return
        log.info(
            f"[atom_queue] Sweeping {len(entries)} pending entries "
            f"for customer '{self.customer_id}'"
        )
        for entry in entries:
            self._send(entry)

    def _send(self, entry: Dict) -> None:
        from core.config import settings

        factory_base = getattr(settings, "FACTORY_URL", "http://localhost:8080").rstrip("/")
        factory_url  = factory_base + "/api/factory/customer/new_atom"
        callback_url = (
            getattr(settings, "RUNTIME_CALLBACK_URL", "http://localhost:8081").rstrip("/")
            + "/api/runtime/receive_package"
        )

        entry["attempts"]    += 1
        entry["last_attempt"] = _now()

        payload = {
            "customer_id":  self.customer_id,
            "table_name":   entry["table_name"],
            "vertical":     entry["vertical"],
            "columns":      entry["columns"],
            "enums":        entry["enums"],
            "row_count":    entry["row_count"],
            "callback_url": callback_url,
            "entry_id":     entry["entry_id"],
        }

        try:
            import httpx
            resp = httpx.post(factory_url, json=payload, timeout=15)
            if resp.status_code in (200, 201):
                entry["status"] = "sent"
                entry["error"]  = None
                log.info(
                    f"[atom_queue] Sent '{entry['table_name']}' to factory "
                    f"(entry_id={entry['entry_id']}, attempt={entry['attempts']})"
                )
            else:
                entry["status"] = "failed"
                entry["error"]  = f"HTTP {resp.status_code}: {resp.text[:200]}"
                log.warning(
                    f"[atom_queue] Factory rejected '{entry['table_name']}': "
                    f"{entry['error']} (attempt={entry['attempts']})"
                )
        except Exception as e:
            entry["status"] = "failed"
            entry["error"]  = str(e)
            log.warning(
                f"[atom_queue] Could not reach factory for '{entry['table_name']}': "
                f"{e} (attempt={entry['attempts']})"
            )

        if entry["attempts"] >= MAX_ATTEMPTS and entry["status"] != "sent":
            log.error(
                f"[atom_queue] Giving up on '{entry['table_name']}' after "
                f"{MAX_ATTEMPTS} attempts (entry_id={entry['entry_id']})"
            )

        _write_entry(self.customer_data_dir, entry)


# ── Worker registry ────────────────────────────────────────────────────────────
# One worker per customer_id, created lazily on first enqueue or startup.

_workers: Dict[str, AtomQueueWorker] = {}
_workers_lock = threading.Lock()


def get_or_create_worker(customer_id: str, customer_data_dir: Path) -> AtomQueueWorker:
    """Return the running worker for this customer, creating and starting one if needed."""
    with _workers_lock:
        if customer_id not in _workers:
            worker = AtomQueueWorker(customer_id, customer_data_dir)
            worker.start()
            _workers[customer_id] = worker
        return _workers[customer_id]


def start_all_workers_from_disk(customers_root: Path) -> None:
    """
    Called at runtime startup. Scans customer_runtime/customers/ for any
    customers that have a non-empty atom queue and starts a worker for each.
    This ensures queued items from a previous session are retried on restart.
    """
    if not customers_root.exists():
        return
    for customer_dir in customers_root.iterdir():
        if not customer_dir.is_dir():
            continue
        customer_id = customer_dir.name
        data_dir    = customer_dir / "data"
        queue_file  = data_dir / QUEUE_FILENAME
        if not queue_file.exists():
            continue
        try:
            entries = json.loads(queue_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        has_pending = any(
            e.get("status") in ("pending", "failed") and e.get("attempts", 0) < MAX_ATTEMPTS
            for e in entries
        )
        if has_pending:
            log.info(
                f"[atom_queue] Startup: found pending entries for customer "
                f"'{customer_id}', starting worker"
            )
            get_or_create_worker(customer_id, data_dir)
