"""
routers/customer_onboard_router.py
====================================
Factory-side endpoint for customer onboarding.

Changes from v17:
  - customer_id is now part of every session (passed in payload from runtime)
  - Per-customer locks replace the single global _PATH_SWAP_LOCK
    → two different customers can run pipelines simultaneously
    → same customer cannot run two pipelines at once
  - GET /api/factory/customer/onboard/sessions returns grouped-by-customer view
  - GET /api/factory/customers lists all customers from customers.json

Flow:
  POST /api/factory/customer/onboard
    ← runtime sends: customer_id + vertical + callback_url + tables + enums
    → starts full 10-stage pipeline in a background thread
    → returns { onboard_id } immediately

  GET /api/factory/customer/onboard/{onboard_id}/status
    → returns stage/status — polled by runtime every 3s

  GET /api/factory/customer/onboard/sessions
    → returns all sessions grouped by customer_id — used by Runtime Logs tab

  GET /api/factory/customers
    → lists all customers from customers.json — used to pre-render cards
"""

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

log = logging.getLogger(__name__)

router = APIRouter()

# ── In-memory session store ───────────────────────────────────────────────────
# { onboard_id: { customer_id, stage, stages, overall_pct, status, error, ... } }
_sessions: Dict[str, Dict] = {}
_sessions_lock = threading.Lock()   # protects _sessions dict

# ── Per-customer pipeline locks ───────────────────────────────────────────────
# One pipeline at a time PER customer. Two different customers can run in parallel.
_customer_locks: Dict[str, threading.Lock] = {}
_customer_locks_mutex = threading.Lock()


def _get_customer_lock(customer_id: str) -> threading.Lock:
    with _customer_locks_mutex:
        if customer_id not in _customer_locks:
            _customer_locks[customer_id] = threading.Lock()
        return _customer_locks[customer_id]


# ── Customers list ────────────────────────────────────────────────────────────
_CUSTOMERS_FILE  = Path(__file__).parent.parent / "customers.json"
_SESSIONS_FILE   = Path(__file__).parent.parent / "data" / "sessions.json"
_MAX_RUNS_KEPT   = 3   # keep the last N completed runs per customer


def _load_session_history() -> Dict[str, list]:
    """Load persisted session history from disk. Returns {customer_id: [sessions]}."""
    try:
        if _SESSIONS_FILE.exists():
            return json.loads(_SESSIONS_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning(f"[sessions] Could not load session history: {e}")
    return {}


def _save_session_history(customer_id: str, session: Dict) -> None:
    """
    Append a completed/failed session to the per-customer history on disk.
    Keeps only the last _MAX_RUNS_KEPT entries per customer.
    Uses a file-level lock so concurrent pipeline completions don't race.
    """
    lock = _get_customer_lock(f"__sessions_file__")
    with lock:
        try:
            history = _load_session_history()
            runs    = history.get(customer_id, [])

            # Build the compact record to persist (drop large stage data arrays)
            record = {
                "onboard_id":  session.get("onboard_id"),
                "vertical":    session.get("vertical"),
                "status":      session.get("status"),
                "error":       session.get("error"),
                "created_at":  session.get("created_at"),
                "finished_at": time.time(),
                "overall_pct": session.get("overall_pct", 0),
                "stages": [
                    {"id": s["id"], "label": s["label"], "status": s["status"],
                     "error": s.get("error")}
                    for s in session.get("stages", [])
                ],
            }

            # Prepend newest first, keep last _MAX_RUNS_KEPT
            runs = [record] + [r for r in runs if r["onboard_id"] != record["onboard_id"]]
            runs = runs[:_MAX_RUNS_KEPT]

            history[customer_id] = runs
            _SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
            _SESSIONS_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")
            log.info(f"[sessions] Persisted run for customer={customer_id} vertical={session.get('vertical')} status={session.get('status')}")
        except Exception as e:
            log.warning(f"[sessions] Could not save session history: {e}")


# Load history into memory on startup
_session_history: Dict[str, list] = _load_session_history()
log.info(f"[sessions] Loaded history for {len(_session_history)} customer(s) from disk")


def _load_customers() -> list:
    try:
        with open(_CUSTOMERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


# ── Stage definitions ─────────────────────────────────────────────────────────
STAGE_NAMES = [
    "reconcile",
    "relationships",
    "mockdata",
    "enforce_enums",
    "seed",
    "vocabulary",
    "goals",
    "compile",
    "verify",
    "package_and_ship",
]

STAGE_LABELS = {
    "reconcile":        "Atom Reconciliation",
    "relationships":    "Relationship Establishment",
    "mockdata":         "Mock Data Generation",
    "enforce_enums":    "Enum Enforcement",
    "seed":             "Seed Generation",
    "vocabulary":       "Vocabulary Generation",
    "goals":            "Goal Generation",
    "compile":          "Compilation",
    "verify":           "Verification",
    "package_and_ship": "Package & Ship",
}


def _new_session(customer_id: str, vertical: str, callback_url: str) -> str:
    onboard_id = str(uuid.uuid4())
    stages = [
        {"id": s, "label": STAGE_LABELS[s], "status": "idle"}
        for s in STAGE_NAMES
    ]
    with _sessions_lock:
        _sessions[onboard_id] = {
            "onboard_id":   onboard_id,
            "customer_id":  customer_id,
            "vertical":     vertical,
            "callback_url": callback_url,
            "stage":        None,
            "stages":       stages,
            "overall_pct":  0,
            "status":       "pending",
            "error":        None,
            "created_at":   time.time(),
        }
    return onboard_id


def _set_stage(onboard_id: str, stage_id: str):
    with _sessions_lock:
        s = _sessions.get(onboard_id)
        if not s:
            return
        s["stage"]  = stage_id
        s["status"] = "running"
        idx = STAGE_NAMES.index(stage_id)
        s["overall_pct"] = round(idx / len(STAGE_NAMES) * 100)
        for st in s["stages"]:
            if st["id"] == stage_id:
                st["status"] = "running"
                break
    log.info(f"[onboard:{onboard_id}] Stage → {stage_id}")


def _complete_stage(onboard_id: str, stage_id: str, data: Optional[Dict] = None):
    with _sessions_lock:
        s = _sessions.get(onboard_id)
        if not s:
            return
        idx = STAGE_NAMES.index(stage_id)
        s["overall_pct"] = round((idx + 1) / len(STAGE_NAMES) * 100)
        for st in s["stages"]:
            if st["id"] == stage_id:
                st["status"] = "done"
                if data:
                    st["data"] = data
                break


def _fail_stage(onboard_id: str, stage_id: str, error: str):
    with _sessions_lock:
        s = _sessions.get(onboard_id)
        if not s:
            return
        s["status"] = "failed"
        s["error"]  = error
        for st in s["stages"]:
            if st["id"] == stage_id:
                st["status"] = "failed"
                st["error"]  = error
                break
    log.error(f"[onboard:{onboard_id}] Stage {stage_id} failed: {error}")
    # Persist failure to disk so history survives factory restarts
    with _sessions_lock:
        s = _sessions.get(onboard_id)
    if s:
        _save_session_history(s["customer_id"], s)
        _session_history.setdefault(s["customer_id"], []).insert(0, s)


def _finish(onboard_id: str):
    with _sessions_lock:
        s = _sessions.get(onboard_id)
        if s:
            s["status"]      = "done"
            s["overall_pct"] = 100
            s["stage"]       = "done"
            # Persist to disk so history survives factory restarts
            _save_session_history(s["customer_id"], s)
            _session_history.setdefault(s["customer_id"], []).insert(0, s)


# ── Request model ─────────────────────────────────────────────────────────────

class OnboardRequest(BaseModel):
    customer_id:    str                   # NEW — identifies which customer
    vertical:       str
    callback_url:   str
    tables:         List[Dict[str, Any]]
    enums:          Dict[str, Any] = {}
    mock_row_count: int            = 50   # rows per record/event/snapshot table


# ── Pipeline runner ───────────────────────────────────────────────────────────

def _run_pipeline(onboard_id: str, payload: OnboardRequest):
    """
    Full 10-stage pipeline — runs in a background thread.
    Acquires the per-customer lock so the same customer can't run two
    pipelines simultaneously. Different customers run in parallel.
    """
    customer_id  = payload.customer_id
    vertical     = payload.vertical
    callback_url = payload.callback_url

    # Per-customer lock — blocks if this customer is already running
    customer_lock = _get_customer_lock(customer_id)

    with customer_lock:
        log.info(f"[onboard:{onboard_id}] Lock acquired for customer={customer_id}")
        _run_pipeline_stages(onboard_id, payload)


def _run_pipeline_stages(onboard_id: str, payload: OnboardRequest):
    """Runs all stages once the lock is held."""
    customer_id  = payload.customer_id
    vertical     = payload.vertical
    callback_url = payload.callback_url

    customer_schema = {
        "tables":        payload.tables,
        "relationships": [],
    }

    # Write enums to customer-namespaced path
    import core.paths as p
    import json as _json

    # Use per-customer paths
    from core.paths import get_customer_paths
    cpaths = get_customer_paths(customer_id)

    enums_path       = cpaths["ENUMS_PATH"]
    field_vals_path  = cpaths["FIELD_VALUES_PATH"]

    enums_path.parent.mkdir(parents=True, exist_ok=True)
    field_vals_path.parent.mkdir(parents=True, exist_ok=True)

    enums_path.write_text(
        _json.dumps(payload.enums, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    field_vals_path.write_text(
        _json.dumps(payload.enums, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    from customer.onboarding import _customer_path_context_for_customer

    def run_stage(stage_id, fn):
        _set_stage(onboard_id, stage_id)
        try:
            with _customer_path_context_for_customer(customer_id, vertical):
                result = fn()
            _complete_stage(onboard_id, stage_id, result)
            return result
        except Exception as e:
            _fail_stage(onboard_id, stage_id, str(e))
            raise

    try:
        # 1. Atom Reconciliation
        def _reconcile():
            import json as _json
            import logging as _rlog

            # ── Bootstrap: copy factory atoms for this vertical into customer atoms.json ──
            # New customers have an empty atoms.json. VSA CUSTOMER mode calls
            # get_existing_atoms() which reads from the (path-swapped) customer
            # atoms.json — finds nothing — returns "no_changes" — seed then fails
            # with "No atoms found for vertical". Fix: pre-seed customer atoms.json
            # with factory template atoms for this vertical before VSA runs.
            _factory_atoms_path = cpaths["ATOMS_PATH"].parent.parent.parent.parent.parent / "data" / "atoms.json"
            _customer_atoms_path = cpaths["ATOMS_PATH"]
            _customer_atoms_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                _factory_data = _json.loads(_factory_atoms_path.read_text(encoding="utf-8"))
            except Exception:
                _factory_data = {"_default": {}}

            _factory_default = _factory_data.get("_default", {})
            _vertical_atoms  = {
                k: v for k, v in _factory_default.items()
                if isinstance(v, dict) and v.get("domain") == vertical
            }

            if _vertical_atoms:
                try:
                    _existing = _json.loads(_customer_atoms_path.read_text(encoding="utf-8"))                         if _customer_atoms_path.exists() else {"_default": {}}
                except Exception:
                    _existing = {"_default": {}}
                _existing_default = _existing.get("_default", {})
                _existing_cids = {
                    v.get("canonical_id") for v in _existing_default.values()
                    if isinstance(v, dict)
                }
                _next_key = max(
                    (int(k) for k in _existing_default if str(k).isdigit()), default=0
                ) + 1
                _added = 0
                for _atom in _vertical_atoms.values():
                    if _atom.get("canonical_id") not in _existing_cids:
                        _existing_default[str(_next_key)] = _atom
                        _next_key += 1
                        _added += 1
                _existing["_default"] = _existing_default
                _customer_atoms_path.write_text(
                    _json.dumps(_existing, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                _rlog.getLogger(__name__).info(
                    f"[reconcile] Pre-seeded {_added} factory atoms for '{vertical}' "
                    f"into customer '{customer_id}' atoms.json"
                )

            # ── Now run VSA CUSTOMER mode to reconcile new tables ──────────────────
            from agents.vertical_schema_agent import run as vsa_run
            result = vsa_run(vertical=vertical, mode="CUSTOMER", customer_schema=customer_schema)
            return {
                "atoms_created": len(result.get("atoms_created", [])),
                "atoms_updated": len(result.get("atoms_updated", [])),
            }
        run_stage("reconcile", _reconcile)

        # 2. Relationship Establishment
        def _relationships():
            from customer.sot_ingestion import infer_relationships
            table_schemas = [
                {
                    "table_name": t["name"],
                    "columns":    [{"name": c["name"], "type": c.get("type", "string")} for c in t.get("columns", [])],
                    "row_count":  t.get("row_count", 0),
                }
                for t in payload.tables
            ]
            relationships = infer_relationships(table_schemas)
            return {"relationships_found": len(relationships)}
        run_stage("relationships", _relationships)

        # 3. Mock Data Generation
        def _mockdata():
            from agents.mock_data_agent import run as mock_run
            result = mock_run(
                vertical=vertical,
                mode="CUSTOMER",
                mock_row_count=payload.mock_row_count,
            )
            return {
                "atoms_processed": result.get("atoms_processed", 0),
                "total_rows":      result.get("total_rows", 0),
            }
        run_stage("mockdata", _mockdata)

        # 4. Enum Enforcement
        def _enforce_enums():
            import csv as _csv
            import random as _random
            tables_fixed  = 0
            columns_fixed = 0
            mock_dir = cpaths["MOCK_DATA_DIR"] / vertical
            if not mock_dir.exists():
                return {"tables_fixed": 0, "columns_fixed": 0, "status": "skipped"}
            for csv_file in sorted(mock_dir.glob("*.csv")):
                table_name  = csv_file.stem
                table_enums = payload.enums.get(table_name, {})
                if not table_enums:
                    continue
                rows = []
                with open(csv_file, newline="", encoding="utf-8") as f:
                    reader     = _csv.DictReader(f)
                    fieldnames = list(reader.fieldnames or [])
                    rows       = list(reader)
                if not rows:
                    continue
                did_fix = False
                for col, allowed in table_enums.items():
                    if col not in fieldnames or not allowed:
                        continue
                    for row in rows:
                        row[col] = _random.choice(allowed)
                    columns_fixed += 1
                    did_fix = True
                if did_fix:
                    with open(csv_file, "w", newline="", encoding="utf-8") as f:
                        writer = _csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(rows)
                    tables_fixed += 1
            return {"tables_fixed": tables_fixed, "columns_fixed": columns_fixed}
        run_stage("enforce_enums", _enforce_enums)

        # 5. Seed Generation
        def _seed():
            from agents.seed_agent import run as seed_run
            result = seed_run(vertical=vertical)
            if result.get("status") == "error":
                raise RuntimeError(result.get("error", "SeedAgent failed"))
            seed     = result.get("seed", {})
            entities = seed.get("entities", {})
            return {
                "entities":       len(entities),
                "total_measures": sum(len(e.get("measures", [])) for e in entities.values()),
            }
        run_stage("seed", _seed)

        # 6. Vocabulary Generation
        def _vocab():
            from agents.vocabulary_agent import run as vocab_run
            result = vocab_run(vertical=vertical)
            return {"total_entries": result.get("total_entries", 0)}
        run_stage("vocabulary", _vocab)

        # 7. Goal Generation
        def _goals():
            from services.goal_generator import generate_goals
            from agents.domain_goals_agent import generate_domain_goals
            result_19 = generate_goals(vertical=vertical)
            all_cids  = list(result_19.get("all_cids", []))
            result_ae = generate_domain_goals(vertical=vertical, wave_19_cids=all_cids)
            return {
                "total_goals": result_19.get("total_goals", 0) + result_ae.get("total_ae_goals", 0),
                "waves_19":    result_19.get("total_goals", 0),
                "waves_ae":    result_ae.get("total_ae_goals", 0),
            }
        run_stage("goals", _goals)

        # 8. Compilation
        def _compile():
            from compiler.compiler_orchestrator import compile_vertical
            result = compile_vertical(vertical=vertical)
            return {
                "total_compiled": result.get("total_compiled", 0),
                "branch_a":       result.get("branch_a", 0),
                "branch_b":       result.get("branch_b", 0),
                "branch_c":       result.get("branch_c", 0),
            }
        run_stage("compile", _compile)

        # 9. Verification
        def _verify():
            from compiler.verifier import verify_vertical
            result = verify_vertical(vertical=vertical)
            locked = result.get("ai_locked_count", 0)
            total  = result.get("total", 0)
            return {
                "total":     total,
                "ai_locked": locked,
                "lock_rate": round(locked / max(total, 1) * 100, 1),
            }
        run_stage("verify", _verify)

        # 10. Package & Ship
        _set_stage(onboard_id, "package_and_ship")
        try:
            from compiler.deployment_engine import package_vertical
            with _customer_path_context_for_customer(customer_id, vertical):
                pkg_result = package_vertical(vertical=vertical)

            pkg_path = pkg_result.get("package_path", "")
            if not pkg_path or not Path(pkg_path).exists():
                raise FileNotFoundError(f"Package not found: {pkg_path}")

            import urllib.request as _urlreq
            with open(pkg_path, "rb") as f:
                _pkg_bytes = f.read()
            _req = _urlreq.Request(
                callback_url,
                data=_pkg_bytes,
                headers={
                    "Content-Type":   "application/zip",
                    "X-Vertical":     vertical,
                    "X-Onboard-Id":   onboard_id,
                    "X-Customer-Id":  customer_id,
                    "X-Metric-Count": str(pkg_result.get("metric_count", 0)),
                    "X-Locked-Count": str(pkg_result.get("locked_count", 0)),
                    "X-Package-Name": Path(pkg_path).name,
                },
                method="POST",
            )
            with _urlreq.urlopen(_req, timeout=60) as _resp:
                _status = _resp.status
                _body   = _resp.read().decode("utf-8", errors="replace")[:200]
            if _status not in (200, 201):
                raise RuntimeError(f"Runtime rejected package: HTTP {_status} — {_body}")

            _complete_stage(onboard_id, "package_and_ship", {
                "package_name": Path(pkg_path).name,
                "metric_count": pkg_result.get("metric_count", 0),
                "locked_count": pkg_result.get("locked_count", 0),
            })
        except Exception as e:
            _fail_stage(onboard_id, "package_and_ship", str(e))
            raise

        _finish(onboard_id)
        log.info(f"[onboard:{onboard_id}] Pipeline complete — customer={customer_id} vertical={vertical}")

    except Exception as e:
        log.error(f"[onboard:{onboard_id}] Pipeline aborted: {e}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/api/factory/customer/onboard")
async def start_onboard(req: OnboardRequest):
    onboard_id = _new_session(req.customer_id, req.vertical, req.callback_url)
    t = threading.Thread(target=_run_pipeline, args=(onboard_id, req), daemon=True)
    t.start()
    log.info(f"[onboard:{onboard_id}] Started — customer={req.customer_id} vertical={req.vertical}")
    return JSONResponse({"onboard_id": onboard_id, "vertical": req.vertical, "customer_id": req.customer_id})


# IMPORTANT: /sessions MUST be registered before /{onboard_id}/status
@router.get("/api/factory/customer/onboard/sessions")
async def list_sessions():
    """
    Returns all sessions grouped by customer_id.
    Used by agents.html Runtime Logs tab to render per-customer cards.
    """
    customers   = _load_customers()
    customer_ids = [c["customer_id"] for c in customers]

    # Build skeleton for ALL registered customers (even those with no runs yet)
    grouped: Dict[str, dict] = {}
    for c in customers:
        cid = c["customer_id"]
        grouped[cid] = {
            "customer_id":  cid,
            "display_name": c.get("display_name", cid),
            "sessions":     [],
        }

    # 1. Start with live in-memory sessions (running or just-started pipelines)
    seen_onboard_ids = set()
    with _sessions_lock:
        for s in _sessions.values():
            cid = s.get("customer_id", "unknown")
            if cid not in grouped:
                grouped[cid] = {"customer_id": cid, "display_name": cid, "sessions": []}
            grouped[cid]["sessions"].append({
                "onboard_id":  s["onboard_id"],
                "vertical":    s["vertical"],
                "status":      s["status"],
                "overall_pct": s["overall_pct"],
                "stage":       s["stage"],
                "stages":      s["stages"],
                "created_at":  s["created_at"],
                "error":       s["error"],
            })
            seen_onboard_ids.add(s["onboard_id"])

    # 2. Merge persisted history (completed/failed runs from previous sessions)
    #    Skip any onboard_id already in memory (avoid duplicates)
    for cid, runs in _session_history.items():
        if cid not in grouped:
            grouped[cid] = {"customer_id": cid, "display_name": cid, "sessions": []}
        for r in runs:
            if r.get("onboard_id") not in seen_onboard_ids:
                grouped[cid]["sessions"].append(r)
                seen_onboard_ids.add(r.get("onboard_id"))

    # 3. Sort each customer's sessions newest-first, keep last _MAX_RUNS_KEPT
    for cid in grouped:
        grouped[cid]["sessions"].sort(key=lambda x: -(x.get("created_at") or 0))
        grouped[cid]["sessions"] = grouped[cid]["sessions"][:_MAX_RUNS_KEPT]

    return JSONResponse({"customers": list(grouped.values())})


@router.get("/api/factory/customer/onboard/{onboard_id}/status")
async def onboard_status(onboard_id: str):
    with _sessions_lock:
        s = _sessions.get(onboard_id)
    if not s:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    return JSONResponse({
        "onboard_id":  s["onboard_id"],
        "customer_id": s["customer_id"],
        "vertical":    s["vertical"],
        "stage":       s["stage"],
        "stages":      s["stages"],
        "overall_pct": s["overall_pct"],
        "status":      s["status"],
        "error":       s["error"],
    })


@router.get("/api/factory/customers")
async def list_customers():
    """All registered customers — used by agents.html to pre-render Runtime Logs cards."""
    customers = _load_customers()
    return JSONResponse({
        "customers": [
            {
                "customer_id":  c["customer_id"],
                "display_name": c.get("display_name", c["customer_id"]),
            }
            for c in customers
        ]
    })


# ── POST /api/factory/customer/new_atom ───────────────────────────────────────
# Called by the runtime atom_queue worker when a customer uploads a file that
# has no matching atom.  Runs the full 10-stage pipeline for a single new table,
# then ships the compiled aterms back to the runtime via callback_url.

class NewAtomRequest(BaseModel):
    customer_id:  str
    table_name:   str
    vertical:     str                    # best-guess from detect_vertical(), may be "unknown"
    columns:      List[Dict[str, Any]]   # [{name, type}]
    enums:        Dict[str, Any] = {}
    row_count:    int            = 0
    callback_url: str            = ""
    entry_id:     str            = ""    # atom_queue entry_id — echoed back in response


@router.post("/api/factory/customer/new_atom")
async def create_new_atom(req: NewAtomRequest):
    """
    Triggered by the runtime when an uploaded file has no matching atom.

    Builds a single-table OnboardRequest and runs the full 10-stage pipeline
    (reconcile → relationships → mockdata → enforce_enums → seed → vocab →
    goals → compile → verify → package_and_ship) in a background thread.

    Returns { onboard_id, entry_id } immediately; progress can be polled
    via GET /api/factory/customer/onboard/{onboard_id}/status.

    If vertical is "unknown", the reconcile stage (VerticalSchemaAgent CUSTOMER
    mode) will infer the correct vertical from the table schema.
    """
    # Resolve vertical: if unknown, let the agent decide; fall back to a
    # sanitised version of the table_name prefix as a best-guess domain.
    vertical = req.vertical
    if not vertical or vertical == "unknown":
        # Derive a candidate vertical from the table_name prefix
        # e.g. "logistics_orders_erp" → "logistics"
        parts = req.table_name.replace("-", "_").split("_")
        vertical = parts[0] if parts else "general"

    # Build an OnboardRequest with a single table
    onboard_payload = OnboardRequest(
        customer_id  = req.customer_id,
        vertical     = vertical,
        callback_url = req.callback_url,
        tables       = [
            {
                "name":      req.table_name,
                "columns":   req.columns,
                "row_count": req.row_count,
            }
        ],
        enums        = {req.table_name: req.enums} if req.enums else {},
    )

    onboard_id = _new_session(req.customer_id, vertical, req.callback_url)
    t = threading.Thread(
        target=_run_pipeline,
        args=(onboard_id, onboard_payload),
        daemon=True,
    )
    t.start()

    log.info(
        f"[new_atom:{onboard_id}] Started — customer={req.customer_id} "
        f"table={req.table_name!r} vertical={vertical!r} entry_id={req.entry_id!r}"
    )

    return JSONResponse({
        "onboard_id":  onboard_id,
        "entry_id":    req.entry_id,
        "customer_id": req.customer_id,
        "table_name":  req.table_name,
        "vertical":    vertical,
    })
