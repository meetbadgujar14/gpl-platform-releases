"""
routers/customer_router.py
============================
Customer Runtime API.

POST /api/customer/upload/add
    Receives CSV files + vertical. Ingests them and stages them as
    "pending" (persisted to disk, so this can be called repeatedly across
    separate sessions before generating). Does NOT run the factory
    pipeline. Returns the full committed+pending table list and a full
    relationship re-scan for preview.

GET /api/customer/upload/pending
    Lists tables currently staged (committed ∪ pending) + relationships,
    without triggering generation.

DELETE /api/customer/upload/pending/{table_name}
    Removes one table from the pending set before it's generated.

POST /api/customer/generate
    Merges everything pending into the permanent committed history, runs a
    full relationship re-scan across ALL committed tables (old + new
    together), then runs: atom reconciliation → full factory pipeline
    (VerticalSchemaAgent CUSTOMER mode → MockDataAgent → seed → vocab →
    goals → compile → verify → deploy → unpack to knowledge_store).
    Clears the pending set on success.

POST /api/customer/query
    Takes a natural language question + vertical.
    Runs: IntentResolver → HIT: execute formula against sot_csv/
                         → MISS: trigger factory compile → receive aterm
                                 → store in knowledge_store → execute
    Returns: oracle_value, canonical_id, method, formula.

GET /api/customer/status
    Returns knowledge store stats — metrics available, ai_locked count, SOT tables.

DELETE /api/customer/reset
    Wipes customer_runtime/ for a fresh start, including pending/committed
    schema history.
"""

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, Body, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.paths import (
    CUSTOMER_SOT_DIR,
    CUSTOMER_KNOWLEDGE_DIR,
    CUSTOMER_RUNTIME_DIR,
    CUSTOMER_PACKAGES_DIR,
    get_customer_paths,
)
from routers.auth_router import get_customer_id
from datetime import datetime

log    = logging.getLogger(__name__)
router = APIRouter()


# ── Token extraction helper ───────────────────────────────────────────────────

def _resolve_token(
    x_session_token: Optional[str] = None,
    token: Optional[str] = None,
) -> str:
    """Pick token from header first, then query param."""
    return x_session_token or token or ""


# ── Per-customer path helpers ─────────────────────────────────────────────────

def _sot_dir(customer_id: str) -> Path:
    return get_customer_paths(customer_id)["SOT_DIR"]

def _knowledge_dir(customer_id: str) -> Path:
    return get_customer_paths(customer_id)["KNOWLEDGE_DIR"]

def _ks_index(customer_id: str) -> Path:
    # Single source of truth — always read from data/canonical_index.json.
    # knowledge_store/canonical_index.json is legacy and may be stale.
    return get_customer_paths(customer_id)["CANONICAL_INDEX"]


def _load_json(p: Path, default=None):
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return default if default is not None else {}


# ── GET /api/customer/numbers ─────────────────────────────────────────────────

@router.get("/api/customer/numbers")
async def get_numbers_graph(
    context: Optional[str] = Query(default=None,
        description="Filter by context: logistics | supply_chain | retail_shopify"),
    x_session_token: Optional[str] = Header(default=None),
    token:           Optional[str] = Query(default=None),
):
    """
    Returns the 'Your Numbers' walk graph for the customer.
    Builds UP/DOWN/sibling edges from canonical_index — no LLM, no execution.
    """
    from customer.numbers_graph import load_and_build

    customer_id = get_customer_id(_resolve_token(x_session_token, token))
    ks_index    = _ks_index(customer_id)
    data_index  = get_customer_paths(customer_id)["CANONICAL_INDEX"]
    index_path  = ks_index if ks_index.exists() else data_index

    graph = load_and_build(index_path, context=context)
    return JSONResponse(graph)


# ── GET /api/customer/trace ───────────────────────────────────────────────────

@router.get("/api/customer/bookmarks")
async def get_bookmarks(
    x_session_token: Optional[str] = Header(default=None),
    token:           Optional[str] = Query(default=None),
):
    """GET /api/customer/bookmarks — returns the list of bookmarked canonical_ids."""
    customer_id = get_customer_id(_resolve_token(x_session_token, token))
    cpaths      = get_customer_paths(customer_id)
    bm_path     = cpaths["DATA_DIR"] / "bookmarks.json"
    if bm_path.exists():
        try:
            return JSONResponse({"bookmarks": json.loads(bm_path.read_text(encoding="utf-8"))})
        except Exception:
            pass
    return JSONResponse({"bookmarks": []})


@router.post("/api/customer/bookmarks")
async def set_bookmarks(
    request:         dict            = Body(...),
    x_session_token: Optional[str]  = Header(default=None),
    token:           Optional[str]  = Query(default=None),
):
    """
    POST /api/customer/bookmarks
    Body: { "bookmarks": ["cid1", "cid2", ...] }
    Replaces the full bookmark list.
    """
    customer_id = get_customer_id(_resolve_token(x_session_token, token))
    cpaths      = get_customer_paths(customer_id)
    bm_path     = cpaths["DATA_DIR"] / "bookmarks.json"
    bookmarks   = request.get("bookmarks", [])
    if not isinstance(bookmarks, list):
        raise HTTPException(400, "bookmarks must be a list")
    bm_path.write_text(json.dumps(bookmarks, ensure_ascii=False), encoding="utf-8")
    return JSONResponse({"ok": True, "bookmarks": bookmarks})


@router.get("/api/customer/trace")
async def get_metric_trace(
    cid:             str             = Query(..., description="canonical_id to trace"),
    page:            int             = Query(default=1,  ge=1,   description="1-based page number"),
    page_size:       int             = Query(default=10, ge=1, le=500, description="rows per page"),
    x_session_token: Optional[str]  = Header(default=None),
    token:           Optional[str]  = Query(default=None),
):
    """
    Returns full traceability for a metric: which table/column/operation it came
    from, row count, column stats, and a sample of rows.
    For compound metrics, recurses into each dependency.
    No LLM. Pure formula parsing + CSV read.
    """
    import csv, math
    from customer.numbers_graph import parse_formula, parse_compound_deps, _label, _fmt

    customer_id = get_customer_id(_resolve_token(x_session_token, token))
    cpaths      = get_customer_paths(customer_id)

    # load canonical index
    ks_index   = _ks_index(customer_id)
    data_index = cpaths["CANONICAL_INDEX"]
    index_path = ks_index if ks_index.exists() else data_index
    try:
        index = _load_json(index_path, {})
    except Exception:
        return JSONResponse({"error": "canonical_index not found"}, status_code=404)

    entry = index.get(cid)
    if not entry:
        return JSONResponse({"error": f"cid not found: {cid}"}, status_code=404)

    formula = entry.get("formula_line", "")

    def _csv_for_table(table: str, context: str) -> Optional[Path]:
        """Find the CSV for a table: real sot_csv first, then mock_data."""
        sot_base  = cpaths["SOT_DIR"] / context
        mock_base = cpaths["MOCK_DATA_DIR"] / context
        for base in (sot_base, mock_base):
            p = base / f"{table}.csv"
            if p.exists():
                return p
        return None

    def _context_for_table(table: str) -> str:
        """Derive context from table prefix (PREFIX IS LAW)."""
        for ctx in ("logistics", "supply_chain", "retail_shopify"):
            if table.startswith(ctx):
                return ctx
        return (entry.get("slots") or {}).get("domain", "")

    def _safe_float(v):
        try:
            f = float(v)
            return None if (math.isnan(f) or math.isinf(f)) else f
        except Exception:
            return None

    def _csv_stats(csv_path: Path, column: Optional[str], operation: str,
                   filters: list, page: int = 1, page_size: int = 10) -> dict:
        """
        Read CSV, apply basic filters.
        - Stats (min/avg/max/sum) are always computed over the FULL filtered dataset.
        - Sample rows are sliced to the requested page window.
        """
        try:
            with open(csv_path, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception as ex:
            return {"error": str(ex), "row_count": None, "stats": None, "sample_rows": []}

        # apply simple equality filters from formula
        for flt in (filters or []):
            if isinstance(flt, dict) and "field" in flt and "value" in flt:
                rows = [r for r in rows if str(r.get(flt["field"], "")) == str(flt["value"])]

        row_count = len(rows)

        # ── stats over FULL dataset (not page-scoped) ─────────────────────────
        stats = None
        if column and column in (rows[0] if rows else {}) and \
                operation not in ("COUNT", "COUNT_DISTINCT"):
            nums = [_safe_float(r.get(column)) for r in rows]
            nums = [n for n in nums if n is not None]
            if nums:
                total = sum(nums)
                n     = len(nums)
                stats = {
                    "sum":    round(total, 4),
                    "mean":   round(total / n, 4),
                    "min":    round(min(nums), 4),
                    "max":    round(max(nums), 4),
                    "n":      n,
                }

        # ── column order: aggregated column first, max 7 cols ─────────────────
        cols = list(rows[0].keys()) if rows else []
        if column and column in cols:
            cols = [column] + [c for c in cols if c != column]
        cols = cols[:7]

        # ── paged sample rows ─────────────────────────────────────────────────
        start  = (page - 1) * page_size
        end    = start + page_size
        sample = [{c: r.get(c) for c in cols} for r in rows[start:end]]

        total_pages = max(1, math.ceil(row_count / page_size)) if row_count else 1

        return {
            "row_count":    row_count,
            "total_pages":  total_pages,
            "page":         page,
            "page_size":    page_size,
            "stats":        stats,
            "columns":      cols,
            "sample_rows":  sample,
            "is_real_data": "sot_csv" in str(csv_path),
        }

    def _build_primitive_trace(table: str, operation: str, column: Optional[str],
                                filters: list, variant: str,
                                req_page: int = 1, req_page_size: int = 10) -> dict:
        context  = _context_for_table(table)
        csv_path = _csv_for_table(table, context)
        result   = {
            "kind":      "primitive",
            "variant":   variant,
            "table":     table,
            "operation": operation,
            "column":    column,
            "filters":   filters,
        }
        if csv_path:
            result.update(_csv_stats(csv_path, column, operation, filters,
                                     page=req_page, page_size=req_page_size))
            result["your_file"] = csv_path.name
        else:
            result["error"] = f"CSV not found for {table}"
        return result

    # ── primitive metric ──────────────────────────────────────────────────────
    parsed = parse_formula(formula)
    if parsed:
        trace = _build_primitive_trace(
            parsed["table"], parsed["operation"], parsed["column"],
            parsed["filters"], parsed["variant"],
            req_page=page, req_page_size=page_size
        )
        trace["cid"]        = cid
        trace["label"]      = _label(cid)
        trace["value"]      = _fmt(entry.get("oracle_value"))
        trace["unit"]       = (entry.get("slots") or {}).get("unit", "")
        trace["source_goal"]= entry.get("source_goal", "")
        return JSONResponse(trace)

    # ── compound metric ───────────────────────────────────────────────────────
    # For compound metrics the client pages each dep independently using the
    # dep's own cid — so page/page_size here apply to the top-level call only
    # (which is unused for compounds). Each dep re-calls the endpoint with its
    # own cid, so we pass page/page_size through for consistency.
    deps = parse_compound_deps(formula, index)
    dep_traces = []
    for dep in deps:
        if dep["primitive"]:
            p = dep["primitive"]
            t = _build_primitive_trace(
                p["table"], p["operation"], p["column"], p["filters"], p["variant"],
                req_page=page, req_page_size=page_size
            )
            t["cid"]   = dep["cid"]
            t["label"] = dep["label"]
            t["value"] = dep["value"]
            t["unit"]  = dep["unit"]
            dep_traces.append(t)
        else:
            dep_traces.append({
                "kind":  "compound",
                "cid":   dep["cid"],
                "label": dep["label"],
                "value": dep["value"],
                "unit":  dep["unit"],
            })

    return JSONResponse({
        "kind":        "compound",
        "cid":         cid,
        "label":       _label(cid),
        "value":       _fmt(entry.get("oracle_value")),
        "unit":        (entry.get("slots") or {}).get("unit", ""),
        "formula":     formula,
        "source_goal": entry.get("source_goal", ""),
        "deps":        dep_traces,
    })



@router.post("/api/customer/upload/add")
async def upload_add(
    files:             List[UploadFile] = File(...),
    vertical:          Optional[str]    = Form(None),
    x_session_token:   Optional[str]    = Header(default=None),
    token:             Optional[str]    = Query(default=None),
):
    """
    Step 1 of onboarding — ingest CSV or Excel files and stage them as "pending".

    Accepted formats:
      .csv         — single table per file (existing behaviour)
      .xlsx / .xls — one table per sheet; empty/header-only sheets are skipped

    Vertical is AUTO-DETECTED per table using filename/sheet prefix + column
    overlap scoring against the factory atom store.  The caller does NOT need
    to pass a vertical — each table is independently routed to the correct domain.

    Returns per-table detection results including confidence scores so the UI
    can surface warnings for low-confidence assignments.
    """
    customer_id     = get_customer_id(_resolve_token(x_session_token, token))
    sot_dir         = _sot_dir(customer_id)
    _cpaths         = get_customer_paths(customer_id)
    _pending_path   = _cpaths["PENDING_SCHEMA"]
    _committed_path = _cpaths["COMMITTED_SCHEMA"]

    from customer.sot_ingestion import ingest_csv, ingest_excel, ingest_sql, update_fingerprints, detect_vertical
    from customer import schema_store

    if not files:
        raise HTTPException(400, "No files uploaded")

    # ── Accepted extensions ────────────────────────────────────────────────────
    _ACCEPTED = {".csv", ".xlsx", ".xls", ".sql"}

    ingest_errors  = []
    ingest_results = []   # [{result, detection}]
    detections     = []   # per-table detection info for the UI (one entry per table)

    for upload in files:
        if not upload.filename:
            continue

        ext = Path(upload.filename).suffix.lower()

        if ext not in _ACCEPTED:
            log.warning(f"[upload_add] Skipping unsupported file type: {upload.filename!r}")
            ingest_errors.append({
                "file":  upload.filename,
                "error": f"Unsupported file type '{ext}'. Accepted: .csv, .xlsx, .xls",
            })
            detections.append({
                "file":   upload.filename,
                "status": "error",
                "error":  f"Unsupported file type '{ext}'.",
            })
            continue

        content = await upload.read()

        # ── SQL branch (.sql) ──────────────────────────────────────────────────
        if ext == ".sql":
            log.info(f"[upload_add] SQL file received: {upload.filename!r} ({len(content)} bytes)")

            table_results = ingest_sql(content, upload.filename, sot_dir, customer_data_dir=_cpaths["DATA_DIR"])

            for result in table_results:
                if result.get("status") != "ok":
                    reason = result.get("reason", "unknown error")
                    tbl_label = result.get("original_table_name") or result.get("table_name", "?")
                    ingest_errors.append({
                        "file":  upload.filename,
                        "table": tbl_label,
                        "error": reason,
                    })
                    detections.append({
                        "file":   upload.filename,
                        "table":  tbl_label,
                        "status": "error",
                        "error":  reason,
                    })
                    continue

                detection  = result.get("detection", {})
                v          = detection.get("vertical", "unknown")
                saved_as   = result.get("saved_as", result.get("table_name", ""))
                table_name = result.get("table_name", "")

                schema_store.upsert_pending(v, result["schema"], _pending_path)
                ingest_results.append({"result": result, "detection": detection})

                # ── Atom creation trigger (SQL table) ──────────────────────────
                _atom_creation_triggered = False
                if detection.get("atom_canonical_id") is None or detection.get("low_confidence", True):
                    try:
                        from customer.atom_queue import enqueue as _aq_enqueue, get_or_create_worker
                        _aq_columns = [
                            {"name": c["name"], "type": c.get("type", "string")}
                            for c in result["schema"].get("columns", [])
                        ]
                        _aq_enqueue(
                            customer_data_dir = _cpaths["DATA_DIR"],
                            customer_id       = customer_id,
                            table_name        = table_name,
                            vertical          = v,
                            columns           = _aq_columns,
                            enums             = result.get("enums", {}),
                            row_count         = result.get("row_count", 0),
                        )
                        get_or_create_worker(customer_id, _cpaths["DATA_DIR"])
                        _atom_creation_triggered = True
                        log.info(
                            f"[upload_add] No atom match for SQL table '{table_name}' — "
                            f"enqueued for factory atom creation"
                        )
                    except Exception as _aq_err:
                        log.warning(f"[upload_add] atom_queue enqueue failed (non-fatal): {_aq_err}")

                detections.append({
                    "file":                    upload.filename,
                    "table":                   result.get("original_table_name", ""),
                    "table_name":              saved_as,
                    "original_name":           table_name,
                    "atom_canonical_id":       detection.get("atom_canonical_id"),
                    "atom_overlap":            detection.get("atom_overlap", 0.0),
                    "vertical":                v,
                    "confidence":              detection.get("confidence", 0.0),
                    "method":                  detection.get("method", ""),
                    "low_confidence":          detection.get("low_confidence", True),
                    "scores":                  detection.get("scores", {}),
                    "source":                  "sql_table",
                    "sql_dialect":             result.get("sql_dialect", "unknown"),
                    "status":                  "ok",
                    "atom_creation_triggered": _atom_creation_triggered,
                })

        # ── Excel branch (.xlsx / .xls) ────────────────────────────────────────
        elif ext in (".xlsx", ".xls"):
            log.info(f"[upload_add] Excel file received: {upload.filename!r} ({len(content)} bytes)")

            sheet_results = ingest_excel(content, upload.filename, sot_dir, customer_data_dir=_cpaths["DATA_DIR"])

            for result in sheet_results:
                if result.get("status") != "ok":
                    reason = result.get("reason", "unknown error")
                    sheet_label = result.get("original_sheet_name") or result.get("table_name", "?")
                    ingest_errors.append({
                        "file":  upload.filename,
                        "sheet": sheet_label,
                        "error": reason,
                    })
                    detections.append({
                        "file":         upload.filename,
                        "sheet":        sheet_label,
                        "status":       "error",
                        "error":        reason,
                    })
                    continue

                detection  = result.get("detection", {})
                v          = detection.get("vertical", "unknown")
                saved_as   = result.get("saved_as", result.get("table_name", ""))
                table_name = result.get("table_name", "")

                schema_store.upsert_pending(v, result["schema"], _pending_path)
                ingest_results.append({"result": result, "detection": detection})

                # ── Atom creation trigger (Excel sheet) ────────────────────────
                # Trigger if no atom matched OR if the vertical confidence is too
                # low to trust the atom assignment (low_confidence = confidence < 0.60).
                _atom_creation_triggered = False
                if detection.get("atom_canonical_id") is None or detection.get("low_confidence", True):
                    try:
                        from customer.atom_queue import enqueue as _aq_enqueue, get_or_create_worker
                        _aq_columns = [
                            {"name": c["name"], "type": c.get("type", "string")}
                            for c in result["schema"].get("columns", [])
                        ]
                        _aq_enqueue(
                            customer_data_dir = _cpaths["DATA_DIR"],
                            customer_id       = customer_id,
                            table_name        = table_name,
                            vertical          = v,
                            columns           = _aq_columns,
                            enums             = result.get("enums", {}),
                            row_count         = result.get("row_count", 0),
                        )
                        get_or_create_worker(customer_id, _cpaths["DATA_DIR"])
                        _atom_creation_triggered = True
                        log.info(
                            f"[upload_add] No atom match for sheet '{table_name}' — "
                            f"enqueued for factory atom creation"
                        )
                    except Exception as _aq_err:
                        log.warning(f"[upload_add] atom_queue enqueue failed (non-fatal): {_aq_err}")

                detections.append({
                    "file":                    upload.filename,
                    "sheet":                   result.get("original_sheet_name", ""),
                    "table_name":              saved_as,
                    "original_name":           table_name,
                    "atom_canonical_id":       detection.get("atom_canonical_id"),
                    "atom_overlap":            detection.get("atom_overlap", 0.0),
                    "vertical":                v,
                    "confidence":              detection.get("confidence", 0.0),
                    "method":                  detection.get("method", ""),
                    "low_confidence":          detection.get("low_confidence", True),
                    "scores":                  detection.get("scores", {}),
                    "source":                  "excel_sheet",
                    "status":                  "ok",
                    "atom_creation_triggered": _atom_creation_triggered,
                })

        # ── CSV branch ─────────────────────────────────────────────────────────
        else:
            stem       = Path(upload.filename).stem.lower().replace(" ", "_").replace("-", "_")
            table_name = stem

            # Write to temp file so we can read headers for detection
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                tmp.write(content)
                tmp_path = Path(tmp.name)

            # Extract column names for detection (read header only)
            try:
                import csv as _csv
                with open(tmp_path, newline="", encoding="utf-8-sig") as f:
                    reader  = _csv.reader(f)
                    headers = next(reader, [])
            except Exception:
                headers = []

            from customer.upload_registry import (
                check_duplicate, lookup_column_cache,
                record_upload as _reg_record, normalise_col,
                content_hash as _csv_content_hash,
            )
            norm_headers = [normalise_col(h) for h in headers]

            # ── Duplicate check ────────────────────────────────────────────────
            # Compute content hash so modified files (same columns, different rows)
            # are NOT treated as duplicates
            _csv_raw_bytes = tmp_path.read_bytes()
            _csv_chash = _csv_content_hash(_csv_raw_bytes)
            dup = check_duplicate(_cpaths["DATA_DIR"], upload.filename, table_name, norm_headers, _csv_chash)
            if dup["is_duplicate"]:
                log.info(f"[upload_add] Duplicate CSV skipped: {upload.filename!r}")
                tmp_path.unlink(missing_ok=True)
                detections.append({
                    "file":        upload.filename,
                    "table_name":  table_name,
                    "status":      "duplicate",
                    "duplicate_of": dup["file_id"],
                    "uploaded_at": dup["uploaded_at"],
                })
                continue

            # ── Pre-filter: column cache fast-track ───────────────────────────
            from customer.pre_filter import check_pre_filter, build_cache_payload, resolve_grain_keys
            from customer.sot_ingestion import build_canonical_map as _build_cmap
            _pre = check_pre_filter(_cpaths["DATA_DIR"], norm_headers)
            if _pre:
                detection      = _pre["detection"]
                _canonical_map = _pre["canonical_map"] or _build_cmap(headers, detection.get("atom_canonical_id"))
                _grain_keys    = _pre["grain_keys"]
                log.info(f"[upload_add] Pre-filter HIT for {upload.filename!r} — skipping detect_vertical()")
            else:
                detection      = detect_vertical(table_name, headers, original_filename=upload.filename)
                _canonical_map = _build_cmap(headers, detection.get("atom_canonical_id"))
                _grain_keys    = resolve_grain_keys(detection.get("atom_canonical_id"))

            v = detection["vertical"]

            log.info(
                f"[upload_add] {upload.filename!r} → vertical={v!r} "
                f"confidence={detection['confidence']:.2f} method={detection['method']}"
            )

            # Temporarily point sot_ingestion at this customer's sot_csv dir
            import customer.sot_ingestion as _sot_mod
            _orig_sot = _sot_mod.CUSTOMER_SOT_DIR
            _sot_mod.CUSTOMER_SOT_DIR = sot_dir
            try:
                result = ingest_csv(
                    tmp_path,
                    table_name,
                    v,
                    atom_canonical_id  = detection.get("atom_canonical_id"),
                    upload_id          = upload.filename,
                    customer_data_dir  = _cpaths["DATA_DIR"],
                    canonical_map      = _canonical_map,
                )
            finally:
                _sot_mod.CUSTOMER_SOT_DIR = _orig_sot
            tmp_path.unlink(missing_ok=True)

            if result["status"] == "ok":
                saved_as = result.get("saved_as", table_name)
                ingest_results.append({"result": result, "detection": detection})
                schema_store.upsert_pending(v, result["schema"], _pending_path)

                # ── Atom creation trigger ──────────────────────────────────────
                # If no atom matched this file, or if the vertical confidence is too
                # low to trust the atom assignment, enqueue for factory atom creation.
                _atom_creation_triggered = False
                if detection.get("atom_canonical_id") is None or detection.get("low_confidence", True):
                    try:
                        from customer.atom_queue import enqueue as _aq_enqueue, get_or_create_worker
                        _aq_columns = [
                            {"name": c["name"], "type": c.get("type", "string")}
                            for c in result["schema"].get("columns", [])
                        ]
                        _aq_enqueue(
                            customer_data_dir = _cpaths["DATA_DIR"],
                            customer_id       = customer_id,
                            table_name        = table_name,
                            vertical          = v,
                            columns           = _aq_columns,
                            enums             = result.get("enums", {}),
                            row_count         = result.get("row_count", 0),
                        )
                        get_or_create_worker(customer_id, _cpaths["DATA_DIR"])
                        _atom_creation_triggered = True
                        log.info(
                            f"[upload_add] No atom match for '{table_name}' — "
                            f"enqueued for factory atom creation"
                        )
                    except Exception as _aq_err:
                        log.warning(f"[upload_add] atom_queue enqueue failed (non-fatal): {_aq_err}")

                det_entry = {
                    "file":                    upload.filename,
                    "table_name":              saved_as,
                    "original_name":           table_name,
                    "atom_canonical_id":       detection.get("atom_canonical_id"),
                    "atom_overlap":            detection.get("atom_overlap", 0.0),
                    "vertical":                v,
                    "confidence":              detection["confidence"],
                    "method":                  detection["method"],
                    "low_confidence":          detection["low_confidence"],
                    "scores":                  detection["scores"],
                    "source":                  "csv",
                    "status":                  "ok",
                    "atom_creation_triggered": _atom_creation_triggered,
                }
                detections.append(det_entry)
                # Record in upload registry — Bug C5 fix: include canonical_map + grain_keys
                _reg_record(_cpaths["DATA_DIR"], upload.filename, [build_cache_payload(
                    sheet_name          = table_name,
                    original_sheet_name = table_name,
                    original_columns    = headers,
                    normalised_columns  = norm_headers,
                    detection           = detection,
                    canonical_map       = result.get("col_map", _canonical_map),
                    grain_keys          = _grain_keys,
                    row_count           = result.get("row_count", 0),
                    content_hash        = _csv_chash,
                )])
            else:
                ingest_errors.append({"file": upload.filename, "error": result.get("reason")})
                detections.append({
                    "file":   upload.filename,
                    "status": "error",
                    "error":  result.get("reason"),
                })

    if not ingest_results:
        raise HTTPException(400, f"No tables ingested. Errors: {ingest_errors}")

    # ── Update fingerprints per vertical ───────────────────────────────────────
    by_vertical: Dict[str, list] = {}
    for item in ingest_results:
        v = item["detection"].get("vertical", "unknown")
        by_vertical.setdefault(v, []).append(item["result"])
    for v, results in by_vertical.items():
        update_fingerprints(results, v)

    # ── Build merged pending view across all affected verticals ────────────────
    from customer.sot_ingestion import infer_relationships
    all_pending = []
    for v in by_vertical:
        merged        = schema_store.merged_view(v, _pending_path, _committed_path)
        table_schemas = schema_store.as_list(merged)
        relationships = infer_relationships(table_schemas)
        all_pending.append({
            "vertical":      v,
            "tables": [
                {
                    "name":    s["table_name"],
                    "rows":    s["row_count"],
                    "columns": len(s["columns"]),
                    "status":  "pending" if s["table_name"] in schema_store.get_pending(v, _pending_path) else "committed",
                }
                for s in table_schemas
            ],
            "relationships": relationships,
        })

    # ── Context detection + Dashboard spec generation ─────────────────────────
    # Mirrors the same block in upload_add_excel_stream.
    # Runs after every successful CSV ingest — non-fatal on any error.
    try:
        from customer.context_detector import detect_context, update_context_schema_cache
        from customer.dashboard_spec_agent import generate_dashboard_spec
        from customer.context_detector import _load_contexts as _lctx_csv

        ds_path_csv      = _cpaths["DATA_DIR"] / "data_store.json"
        affected_contexts_csv: set = set()

        for item in ingest_results:
            res       = item["result"]
            detection = item["detection"]
            tname     = res.get("saved_as") or detection.get("table_name", "unknown")
            vertical  = detection.get("vertical")

            schema_cols  = res.get("schema", {}).get("columns", [])
            sample_enums = res.get("enums", {})
            sample_vals  = {col: vals[:5] for col, vals in sample_enums.items()}
            col_dicts    = [{"name": c["name"], "type": c.get("type", "string")} for c in schema_cols]

            ctx_result = detect_context(
                data_dir      = _cpaths["DATA_DIR"],
                table_name    = tname,
                columns       = col_dicts,
                sample_values = sample_vals,
                vertical      = vertical,
            )
            ctx_id = ctx_result["context_id"]
            affected_contexts_csv.add(ctx_id)
            update_context_schema_cache(
                _cpaths["DATA_DIR"], ctx_id,
                [c["name"] for c in col_dicts],
                sample_vals,
            )
            log.info(
                f"[upload_add] '{tname}' -> context '{ctx_id}' "
                f"({ctx_result['action']}) [{ctx_result['context_name']}]"
            )

        ctx_store_csv = _lctx_csv(_cpaths["DATA_DIR"])
        for ctx_id in affected_contexts_csv:
            ctx = ctx_store_csv.get("contexts", {}).get(ctx_id, {})
            generate_dashboard_spec(
                data_dir        = _cpaths["DATA_DIR"],
                context_id      = ctx_id,
                context_name    = ctx.get("name", ctx_id),
                table_names     = ctx.get("tables", []),
                data_store_path = ds_path_csv,
            )
            log.info(f"[upload_add] Dashboard spec regenerated for context '{ctx_id}'")

    except Exception as _ctx_err_csv:
        log.warning(f"[upload_add] Context detection / dashboard spec failed (non-fatal): {_ctx_err_csv}")

    return JSONResponse({
        "status":              "added",
        "detections":          detections,
        "just_added":          [d["table_name"] for d in detections if d.get("status") == "ok"],
        "errors":              ingest_errors,
        "pending_by_vertical": all_pending,
    })


# ── POST /api/customer/upload/add-excel-stream ──────────────────────────────

@router.post("/api/customer/upload/add-excel-stream")
async def upload_add_excel_stream(
    file:            UploadFile      = File(...),
    x_session_token: Optional[str]  = Header(default=None),
    token:           Optional[str]  = Query(default=None),
):
    """
    Streaming endpoint for Excel uploads.

    Accepts a single .xlsx / .xls file and processes it sheet by sheet,
    yielding one NDJSON line per event so the browser can render a live
    progress bar.

    Event stream format (newline-delimited JSON):
      {"event":"parse_start",  "sheets": N, "filename": "..."}
      {"event":"parse_done",   "usable_sheets": N}
      {"event":"sheet_start",  "sheet_name":"...", "original_sheet_name":"...",
                                "rows": N, "sheet_index": i, "total_sheets": N}
      {"event":"sheet_detected","sheet_name":"...", "vertical":"...",
                                 "confidence": 0.0–1.0, "low_confidence": bool}
      {"event":"sheet_done",   "sheet_name":"...", "saved_as":"...",
                                "row_count": N, "sheet_index": i, "total_sheets": N}
      {"event":"sheet_error",  "sheet_name":"...", "reason":"..."}
      {"event":"enums_done",   "enum_cols": N}
      {"event":"done",         "tables_staged": N, "detections":[...],
                                "errors":[...], "pending_by_vertical":[...]}
      {"event":"error",        "message":"..."}
    """
    import json as _json
    from fastapi.responses import StreamingResponse as _SR

    customer_id     = get_customer_id(_resolve_token(x_session_token, token))
    sot_dir         = _sot_dir(customer_id)
    _cpaths         = get_customer_paths(customer_id)
    _pending_path   = _cpaths["PENDING_SCHEMA"]
    _committed_path = _cpaths["COMMITTED_SCHEMA"]

    ext = Path(file.filename or "").suffix.lower()
    if ext not in (".xlsx", ".xls"):
        raise HTTPException(400, f"Only .xlsx and .xls files accepted by this endpoint. Got: {ext!r}")

    content = await file.read()
    filename = file.filename or "upload.xlsx"

    async def _generate():
        import asyncio
        from customer.excel_parser import parse_excel
        from customer.sot_ingestion import ingest_csv, detect_vertical, update_fingerprints
        from customer.data_cleaner import clean_sheet, write_transformation_log
        from customer import schema_store
        import csv as _csv
        import tempfile

        def _emit(event, **kw):
            return _json.dumps({"event": event, **kw}, ensure_ascii=False) + "\n"

        # ── Step 1: Signal parse start BEFORE blocking parse call ──────────────
        yield _emit("parse_start", filename=filename)
        await asyncio.sleep(0)   # flush bytes to browser now

        try:
            # parse_excel is sync/blocking (pandas) — offload to thread pool so
            # the event loop stays free and the yield above reaches the client
            loop   = asyncio.get_event_loop()
            sheets = await loop.run_in_executor(None, parse_excel, content, filename)
        except ValueError as exc:
            yield _emit("error", message=str(exc))
            return

        # ── Data cleaning pass (offloaded; pure-Python, no I/O except log) ────
        # Note: vertical is not yet known here (detection runs per-sheet below),
        # so we run a fast pre-detection pass using just the sheet name + columns
        # to get the vertical hint for date disambiguation.  Failures are silent.
        def _run_cleaner(sheets_in):
            reports = []
            cleaned = []
            for _s in sheets_in:
                # Quick vertical hint so data_cleaner uses the right date convention
                # (e.g. logistics/supply_chain → DD/MM).  Non-fatal if it fails.
                _vert_hint = None
                try:
                    from customer.sot_ingestion import detect_vertical as _dv_quick
                    _quick = _dv_quick(_s.get("sheet_name", ""), _s.get("columns", []))
                    _vert_hint = _quick.get("vertical") if _quick.get("confidence", 0) >= 0.4 else None
                except Exception:
                    pass
                _c, _r = clean_sheet(_s, options={"vertical": _vert_hint} if _vert_hint else {})
                cleaned.append(_c)
                reports.append(_r)
            return cleaned, reports

        sheets, _clean_reports = await loop.run_in_executor(None, _run_cleaner, sheets)

        # Persist transformation log (non-fatal)
        try:
            _cdr = _cpaths.get("DATA_DIR") if isinstance(_cpaths, dict) else None
            if _cdr:
                await loop.run_in_executor(
                    None, write_transformation_log, _cdr, filename, _clean_reports
                )
        except Exception as _tlog_exc:
            log.warning(f"[excel-stream] transformation_log write failed (non-fatal): {_tlog_exc}")

        # Emit transformation summary badge to client — include per-sheet detail
        # so the UI can show a breakdown modal when the user clicks the badge.
        _badge = "amber" if any(r.get("badge_colour") == "amber" for r in _clean_reports) else "green"
        _total_changed = sum(r.get("total_cells_changed", 0) for r in _clean_reports)
        yield _emit(
            "transformations_applied",
            transformations_count=_total_changed,
            badge_colour=_badge,
            reports=_clean_reports,   # full per-sheet breakdown for the UI modal
        )
        await asyncio.sleep(0)

        yield _emit("parse_done", usable_sheets=len(sheets))
        await asyncio.sleep(0)

        if not sheets:
            yield _emit("error", message=f"'{filename}' contains no usable sheets.")
            return

        # ── Step 2: Per-sheet ingest ───────────────────────────────────────────
        ingest_results = []
        detections_out = []
        ingest_errors  = []
        total = len(sheets)

        from customer.upload_registry import (
            check_duplicate   as _dup_check,
            lookup_column_cache as _cache_lookup,
            record_upload     as _reg_record,
            normalise_col     as _norm_col,
            content_hash      as _sheet_hash,
        )
        registry_sheets = []   # accumulate for record_upload after all sheets done

        for idx, sheet in enumerate(sheets):
            sheet_name          = sheet["sheet_name"]
            original_sheet_name = sheet["original_sheet_name"]
            rows                = sheet["rows"]
            original_cols       = sheet["columns"]
            norm_cols           = [_norm_col(c) for c in original_cols]

            yield _emit("sheet_start",
                        sheet_name=sheet_name,
                        original_sheet_name=original_sheet_name,
                        rows=sheet["row_count"],
                        sheet_index=idx,
                        total_sheets=total)
            await asyncio.sleep(0)

            # ── Duplicate check ────────────────────────────────────────────────
            # Hash this sheet's rows so same-columns-different-rows is NOT a duplicate
            import json as _sheet_json
            _sheet_content_hash = _sheet_hash(
                _sheet_json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()
            )
            dup = _dup_check(_cpaths["DATA_DIR"], filename, sheet_name, norm_cols, _sheet_content_hash)
            if dup["is_duplicate"]:
                log.info(f"[excel-stream] Duplicate sheet skipped: '{sheet_name}' in '{filename}'")
                detections_out.append({
                    "file":        filename,
                    "sheet":       original_sheet_name,
                    "table_name":  sheet_name,
                    "status":      "duplicate",
                    "duplicate_of": dup["file_id"],
                    "uploaded_at": dup["uploaded_at"],
                })
                yield _emit("sheet_duplicate",
                            sheet_name=sheet_name,
                            original_sheet_name=original_sheet_name,
                            duplicate_of=dup["file_id"],
                            uploaded_at=dup["uploaded_at"])
                await asyncio.sleep(0)
                continue

            # ── Pre-filter: column cache fast-track ───────────────────────────
            from customer.pre_filter import check_pre_filter as _pf_check, build_cache_payload as _pf_build, resolve_grain_keys as _pf_gkeys
            from customer.sot_ingestion import build_canonical_map as _build_cmap2
            from customer.canonicalizer import canonicalize_headers as _canonicalize
            _pre2 = _pf_check(_cpaths["DATA_DIR"], norm_cols)
            if _pre2:
                detection       = _pre2["detection"]
                _canonical_map2 = _pre2["canonical_map"] or _build_cmap2(original_cols, detection.get("atom_canonical_id"))
                _grain_keys2    = _pre2["grain_keys"]
                log.info(f"[excel-stream] Pre-filter HIT for sheet '{sheet_name}' — skipping detect_vertical()")
            else:
                # First-pass detection using raw column names
                detection = await asyncio.get_event_loop().run_in_executor(
                    None, detect_vertical, sheet_name, original_cols, None, filename
                )
                vertical_first = detection["vertical"]

                # ── Canonicalize headers ──────────────────────────────────────
                # Maps raw column names to factory canonical names using the
                # vertical's dialect column_hints. Falls back to snake_case
                # silently — never blocks the upload.
                if vertical_first != "unknown":
                    try:
                        _canonical_map2 = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: _canonicalize(
                                original_cols,
                                vertical_first,
                                detection.get("atom_canonical_id"),
                            )
                        )
                        # Re-run detection with canonical column names for
                        # higher-confidence atom matching
                        canonical_col_values = list(_canonical_map2.values())
                        detection2 = await asyncio.get_event_loop().run_in_executor(
                            None, detect_vertical, sheet_name, canonical_col_values, None, filename
                        )
                        # Accept re-run result only if it's equal or better confidence
                        if detection2["confidence"] >= detection["confidence"]:
                            detection = detection2
                            log.info(
                                f"[excel-stream] detect_vertical re-run after canonicalize: "
                                f"confidence {detection2['confidence']:.2f} "
                                f"atom={detection2.get('atom_canonical_id')}"
                            )
                    except Exception as _can_err:
                        log.warning(f"[excel-stream] Canonicalize step failed (non-fatal): {_can_err}")
                        _canonical_map2 = _build_cmap2(original_cols, detection.get("atom_canonical_id"))
                else:
                    _canonical_map2 = _build_cmap2(original_cols, detection.get("atom_canonical_id"))

                _grain_keys2 = _pf_gkeys(detection.get("atom_canonical_id"))
            vertical = detection["vertical"]

            yield _emit("sheet_detected",
                        sheet_name=sheet_name,
                        vertical=vertical,
                        confidence=detection["confidence"],
                        low_confidence=detection["low_confidence"],
                        method=detection["method"])
            await asyncio.sleep(0)

            # ── Write temp CSV → ingest_csv (offload blocking I/O) ────────────
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".csv", delete=False, mode="w",
                    newline="", encoding="utf-8"
                ) as tmp:
                    tmp_path = Path(tmp.name)
                    writer   = _csv.DictWriter(tmp, fieldnames=original_cols, extrasaction="ignore")
                    writer.writeheader()
                    for row in rows:
                        writer.writerow({k: ("" if v is None else str(v)) for k, v in row.items()})

                # Redirect CUSTOMER_SOT_DIR to per-customer path then run in executor
                import customer.sot_ingestion as _sot_mod
                _orig_sot = _sot_mod.CUSTOMER_SOT_DIR
                _sot_mod.CUSTOMER_SOT_DIR = sot_dir
                try:
                    _cm2 = _canonical_map2  # capture for lambda closure
                    result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: ingest_csv(
                            tmp_path,
                            table_name        = sheet_name,
                            vertical          = vertical,
                            atom_canonical_id = detection.get("atom_canonical_id"),
                            upload_id         = filename,
                            customer_data_dir = _cpaths["DATA_DIR"],
                            canonical_map     = _cm2,
                        )
                    )
                finally:
                    _sot_mod.CUSTOMER_SOT_DIR = _orig_sot

                if result["status"] == "ok":
                    saved_as = result.get("saved_as", sheet_name)
                    result["original_sheet_name"] = original_sheet_name
                    result["detection"]           = detection
                    ingest_results.append({"result": result, "detection": detection})
                    schema_store.upsert_pending(vertical, result["schema"], _pending_path)

                    # ── Atom creation trigger (Excel stream) ───────────────────
                    # Trigger if no atom matched OR if the vertical confidence is too
                    # low to trust the atom assignment (low_confidence = confidence < 0.60).
                    _atom_creation_triggered = False
                    if detection.get("atom_canonical_id") is None or detection.get("low_confidence", True):
                        try:
                            from customer.atom_queue import enqueue as _aq_enqueue, get_or_create_worker
                            _aq_columns = [
                                {"name": c["name"], "type": c.get("type", "string")}
                                for c in result["schema"].get("columns", [])
                            ]
                            _aq_enqueue(
                                customer_data_dir = _cpaths["DATA_DIR"],
                                customer_id       = customer_id,
                                table_name        = sheet_name,
                                vertical          = vertical,
                                columns           = _aq_columns,
                                enums             = result.get("enums", {}),
                                row_count         = result.get("row_count", 0),
                            )
                            get_or_create_worker(customer_id, _cpaths["DATA_DIR"])
                            _atom_creation_triggered = True
                            log.info(
                                f"[excel-stream] No atom match for sheet '{sheet_name}' — "
                                f"enqueued for factory atom creation"
                            )
                        except Exception as _aq_err:
                            log.warning(f"[excel-stream] atom_queue enqueue failed (non-fatal): {_aq_err}")

                    det_entry = {
                        "file":                    filename,
                        "sheet":                   original_sheet_name,
                        "table_name":              saved_as,
                        "original_name":           sheet_name,
                        "atom_canonical_id":       detection.get("atom_canonical_id"),
                        "atom_overlap":            detection.get("atom_overlap", 0.0),
                        "vertical":                vertical,
                        "confidence":              detection["confidence"],
                        "method":                  detection["method"],
                        "low_confidence":          detection["low_confidence"],
                        "scores":                  detection["scores"],
                        "source":                  "excel_sheet",
                        "status":                  "ok",
                        "atom_creation_triggered": _atom_creation_triggered,
                    }
                    detections_out.append(det_entry)
                    # Accumulate for record_upload — Bug C5 fix: include canonical_map + grain_keys
                    registry_sheets.append(_pf_build(
                        sheet_name          = sheet_name,
                        original_sheet_name = original_sheet_name,
                        original_columns    = original_cols,
                        normalised_columns  = norm_cols,
                        detection           = detection,
                        canonical_map       = result.get("col_map", _canonical_map2),
                        grain_keys          = _grain_keys2,
                        row_count           = result.get("row_count", 0),
                        content_hash        = _sheet_content_hash,
                    ))

                    yield _emit("sheet_done",
                                sheet_name=sheet_name,
                                saved_as=saved_as,
                                row_count=result["row_count"],
                                sheet_index=idx,
                                total_sheets=total)
                    await asyncio.sleep(0)
                else:
                    reason = result.get("reason", "unknown error")
                    ingest_errors.append({"file": filename, "sheet": original_sheet_name, "error": reason})
                    detections_out.append({"file": filename, "sheet": original_sheet_name, "status": "error", "error": reason})
                    yield _emit("sheet_error", sheet_name=original_sheet_name, reason=reason)

            except Exception as exc:
                reason = str(exc)
                ingest_errors.append({"file": filename, "sheet": original_sheet_name, "error": reason})
                detections_out.append({"file": filename, "sheet": original_sheet_name, "status": "error", "error": reason})
                yield _emit("sheet_error", sheet_name=original_sheet_name, reason=reason)
            finally:
                if tmp_path and tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)

        # ── Step 3: Update fingerprints ────────────────────────────────────────
        by_vertical: Dict[str, list] = {}
        for item in ingest_results:
            v = item["detection"].get("vertical", "unknown")
            by_vertical.setdefault(v, []).append(item["result"])
        for v, res_list in by_vertical.items():
            update_fingerprints(res_list, v)

        # Count total enum columns across all sheets
        total_enum_cols = sum(
            len(item["result"].get("enums", {}))
            for item in ingest_results
        )
        yield _emit("enums_done", enum_cols=total_enum_cols)
        await asyncio.sleep(0)

        # ── Record upload in registry (dup detection + column cache) ──────────
        if registry_sheets:
            try:
                _reg_record(_cpaths["DATA_DIR"], filename, registry_sheets)
            except Exception as exc:
                log.warning(f"[excel-stream] upload_registry.record_upload failed (non-fatal): {exc}")

        # ── Context detection + Dashboard spec generation ─────────────────────
        # Run for each successfully ingested table. Non-fatal on any error.
        try:
            from customer.context_detector import detect_context, update_context_schema_cache
            from customer.dashboard_spec_agent import generate_dashboard_spec

            ds_path = _cpaths["DATA_DIR"] / "data_store.json"
            affected_contexts = set()

            for item in ingest_results:
                res       = item["result"]
                detection = item["detection"]
                tname     = res.get("saved_as") or detection.get("table_name", "unknown")
                vertical  = detection.get("vertical")

                # Extract column info and sample enums for context matching
                schema_cols  = res.get("schema", {}).get("columns", [])
                sample_enums = res.get("enums", {})
                sample_vals  = {col: vals[:5] for col, vals in sample_enums.items()}

                col_dicts = [{"name": c["name"], "type": c.get("type", "string")} for c in schema_cols]

                ctx_result = detect_context(
                    data_dir      = _cpaths["DATA_DIR"],
                    table_name    = tname,
                    columns       = col_dicts,
                    sample_values = sample_vals,
                    vertical      = vertical,
                )
                ctx_id = ctx_result["context_id"]
                affected_contexts.add(ctx_id)

                # Update the context's schema cache for faster matching next time
                update_context_schema_cache(
                    _cpaths["DATA_DIR"], ctx_id,
                    [c["name"] for c in col_dicts],
                    sample_vals,
                )

                log.info(
                    f"[excel-stream] '{tname}' -> context '{ctx_id}' "
                    f"({ctx_result['action']}) [{ctx_result['context_name']}]"
                )

            # Regenerate dashboard spec for each affected context
            from customer.context_detector import _load_contexts as _lctx
            ctx_store = _lctx(_cpaths["DATA_DIR"])
            for ctx_id in affected_contexts:
                ctx = ctx_store.get("contexts", {}).get(ctx_id, {})
                generate_dashboard_spec(
                    data_dir     = _cpaths["DATA_DIR"],
                    context_id   = ctx_id,
                    context_name = ctx.get("name", ctx_id),
                    table_names  = ctx.get("tables", []),
                    data_store_path = ds_path,
                )
                log.info(f"[excel-stream] Dashboard spec regenerated for context '{ctx_id}'")

        except Exception as _ctx_err:
            log.warning(f"[excel-stream] Context detection / dashboard spec failed (non-fatal): {_ctx_err}")

        # ── Step 4: Build pending view ─────────────────────────────────────────
        from customer.sot_ingestion import infer_relationships
        all_pending = []
        for v in by_vertical:
            merged        = schema_store.merged_view(v, _pending_path, _committed_path)
            table_schemas = schema_store.as_list(merged)
            relationships = infer_relationships(table_schemas)
            all_pending.append({
                "vertical":  v,
                "tables": [
                    {
                        "name":    s["table_name"],
                        "rows":    s["row_count"],
                        "columns": len(s["columns"]),
                        "status":  "pending" if s["table_name"] in schema_store.get_pending(v, _pending_path) else "committed",
                    }
                    for s in table_schemas
                ],
                "relationships": relationships,
            })

        yield _emit("done",
                    tables_staged=len(ingest_results),
                    detections=detections_out,
                    errors=ingest_errors,
                    pending_by_vertical=all_pending)

    return _SR(
        _generate(),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


# ── GET /api/customer/upload/pending ─────────────────────────────────────────

@router.get("/api/customer/upload/pending")
async def upload_pending(
    vertical:        str,
    x_session_token: Optional[str] = Header(default=None),
    token:           Optional[str] = Query(default=None),
):
    """List tables currently staged (committed ∪ pending) plus a full
    relationship re-scan, without triggering generation."""
    from customer.sot_ingestion import infer_relationships
    from customer import schema_store
    customer_id     = get_customer_id(_resolve_token(x_session_token, token))
    _cpaths         = get_customer_paths(customer_id)
    _pending_path   = _cpaths["PENDING_SCHEMA"]
    _committed_path = _cpaths["COMMITTED_SCHEMA"]

    merged        = schema_store.merged_view(vertical, _pending_path, _committed_path)
    table_schemas = schema_store.as_list(merged)
    relationships = infer_relationships(table_schemas)
    pending_only  = set(schema_store.get_pending(vertical, _pending_path).keys())

    return JSONResponse({
        "vertical": vertical,
        "tables": [
            {
                "name":    s["table_name"],
                "rows":    s["row_count"],
                "columns": len(s["columns"]),
                "status":  "pending" if s["table_name"] in pending_only else "committed",
            }
            for s in table_schemas
        ],
        "relationships": relationships,
    })


# ── DELETE /api/customer/upload/pending/{table_name} ─────────────────────────

@router.delete("/api/customer/upload/pending/{table_name}")
async def upload_pending_remove(
    table_name:      str,
    vertical:        str,
    x_session_token: Optional[str] = Header(default=None),
    token:           Optional[str] = Query(default=None),
):
    """Remove one table from the pending set (e.g. 'wrong file'). Only
    affects tables that haven't been Generated yet — committed tables from
    a previous Generate can't be removed this way."""
    from customer.sot_ingestion import infer_relationships
    from customer import schema_store
    customer_id     = get_customer_id(_resolve_token(x_session_token, token))
    _cpaths         = get_customer_paths(customer_id)
    _pending_path   = _cpaths["PENDING_SCHEMA"]
    _committed_path = _cpaths["COMMITTED_SCHEMA"]

    existed = schema_store.remove_pending(vertical, table_name, _pending_path)
    if not existed:
        raise HTTPException(404, f"'{table_name}' is not in the pending set for '{vertical}'")

    merged        = schema_store.merged_view(vertical, _pending_path, _committed_path)
    table_schemas = schema_store.as_list(merged)
    relationships = infer_relationships(table_schemas)

    return JSONResponse({
        "status":  "removed",
        "table":   table_name,
        "vertical": vertical,
        "tables": [
            {"name": s["table_name"], "rows": s["row_count"], "columns": len(s["columns"])}
            for s in table_schemas
        ],
        "relationships": relationships,
    })


# ── POST /api/customer/generate ──────────────────────────────────────────────

@router.post("/api/customer/generate")
async def generate(
    vertical:        str          = Form(...),
    factory_url:     str          = Form(None),
    callback_url:    str          = Form(None),
    mock_row_count:  int          = Form(50),
    x_session_token: Optional[str] = Header(default=None),
    token:           Optional[str] = Query(default=None),
):
    """
    Step 2 of onboarding — extract schema + enum metadata from staged tables
    and send it to the factory to run the full pipeline.

    The factory runs: reconcile → relationships → mockdata → enforce_enums →
    seed → vocabulary → goals → compile → verify → package → ship back.

    Returns { status, onboard_id } immediately.
    Runtime polls GET /api/factory/customer/onboard/{onboard_id}/status.
    """
    import json as _json
    from customer.sot_ingestion import infer_relationships, build_customer_schema
    from customer import schema_store

    customer_id     = get_customer_id(_resolve_token(x_session_token, token))
    _cpaths         = get_customer_paths(customer_id)
    _pending_path   = _cpaths["PENDING_SCHEMA"]
    _committed_path = _cpaths["COMMITTED_SCHEMA"]

    pending_before = schema_store.get_pending(vertical, _pending_path)
    if not pending_before and not schema_store.get_committed(vertical, _committed_path):
        raise HTTPException(400, "Nothing to generate — upload at least one CSV first.")

    # Build full merged schema (pending + committed)
    merged        = schema_store.merged_view(vertical, _pending_path, _committed_path)
    table_schemas = schema_store.as_list(merged)
    relationships = infer_relationships(table_schemas)
    customer_schema = build_customer_schema(table_schemas, relationships)

    # Build tables payload
    tables = [
        {
            "name":      s["table_name"],
            "columns":   [{"name": c["name"], "type": c.get("type", "string")} for c in s.get("columns", [])],
            "row_count": s.get("row_count", 0),
        }
        for s in table_schemas
    ]

    # Load enum values from field_values.json (single source of truth)
    # Convert flat {"atom.col": [values]} → nested {"atom": {"col": [values]}} for factory payload
    enums = {}
    try:
        from core.paths import get_customer_paths as _gcpaths
        _fv_path = _gcpaths(customer_id)["FIELD_VALUES_PATH"]
        if _fv_path.exists():
            flat = _json.loads(_fv_path.read_text(encoding="utf-8"))
            for key, values in flat.items():
                if "." in key:
                    atom_key, col = key.split(".", 1)
                    enums.setdefault(atom_key, {})[col] = values
    except Exception as e:
        log.warning(f"[generate] Could not load field_values.json: {e}")

    # Use URLs from request (set by customer in Settings tab) or fall back to config
    from core.config import settings
    _factory_base  = (factory_url  or getattr(settings, "FACTORY_URL",          "http://localhost:8080")).rstrip("/")
    _callback_base = (callback_url or getattr(settings, "RUNTIME_CALLBACK_URL", "http://localhost:8081")).rstrip("/")
    factory_url    = _factory_base  + "/api/factory/customer/onboard"
    callback_url   = _callback_base + "/api/runtime/receive_package"

    try:
        import urllib.request, urllib.error
        _mock_row_count = max(10, min(500, mock_row_count))  # clamp: 10–500
        _payload = _json.dumps({"customer_id": customer_id, "vertical": vertical, "callback_url": callback_url, "tables": tables, "enums": enums, "mock_row_count": _mock_row_count}).encode()
        _req = urllib.request.Request(factory_url, data=_payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(_req, timeout=15) as _r:
            if _r.status not in (200, 201):
                raise RuntimeError(f"Factory returned HTTP {_r.status}")
            data = _json.loads(_r.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Factory returned HTTP {e.code}: {e.read()[:200].decode(errors='replace')}")
    except Exception as e:
        raise HTTPException(502, f"Could not reach factory: {e}")

    return JSONResponse({
        "status":     "pipeline_started",
        "vertical":   vertical,
        "onboard_id": data.get("onboard_id"),
        "tables":     len(tables),
    })


# ── POST /api/customer/query ───────────────────────────────────────────────────

class QueryRequest(BaseModel):
    vertical: str
    question: str
    token:    Optional[str] = None   # session token passed in JSON body


@router.post("/api/customer/query")
async def query(
    req:             QueryRequest,
    x_session_token: Optional[str] = Header(default=None),
):
    """
    Query the customer's knowledge store with a natural language question.

    HIT  → execute formula against real sot_csv/ → return real oracle value
    MISS → send to factory → factory compiles → aterm shipped back
           → stored in knowledge_store → execute → return answer
    """
    customer_id   = get_customer_id(_resolve_token(x_session_token, req.token))
    knowledge_dir = _knowledge_dir(customer_id)
    sot_dir       = _sot_dir(customer_id)

    if not _ks_index(customer_id).exists():
        raise HTTPException(400, (
            "Knowledge store is empty. "
            "Upload your CSV files first via POST /api/customer/upload/add, "
            "then POST /api/customer/generate."
        ))

    from customer.intent_resolver import IntentResolver
    from customer.formula_executor import execute

    resolver = IntentResolver(
        knowledge_store=knowledge_dir,
        vertical=req.vertical,
        canonical_index_path=get_customer_paths(customer_id)["CANONICAL_INDEX"],
    )
    resolution = resolver.resolve(req.question)

    # ── MISS path ──────────────────────────────────────────────────────────────
    if resolution.method == "miss":
        # Try to compile the goal at the factory and ship it back
        miss_result = _handle_miss(req.question, req.vertical, resolver)
        if miss_result.get("status") == "compiled":
            # Now re-resolve with the newly added aterm
            resolution = resolver.resolve(req.question)
            if resolution.method == "miss":
                # Still can't find it — return miss response
                return JSONResponse({
                    "status":       "miss",
                    "question":     req.question,
                    "suggestions":  resolution.suggestions,
                    "factory_note": miss_result.get("note", ""),
                })
        else:
            return JSONResponse({
                "status":       "miss",
                "question":     req.question,
                "suggestions":  resolution.suggestions,
                "factory_note": miss_result.get("error", "Factory could not compile this goal."),
            })

    # ── HIT path — execute formula against real SOT data ──────────────────────
    canonical_index = _load_json(_ks_index(customer_id), {})
    comp_rules      = _load_json(knowledge_dir / "composition_rules.json", {})

    cid = resolution.canonical_id

    # ── Priority 1: sot_results.json — pre-computed real customer data ────────
    # Written by "Execute All Metrics". Always prefer this over live re-execution.
    sot_results_path = get_customer_paths(customer_id)["DATA_DIR"] / "sot_results.json"
    sot_result = None
    if sot_results_path.exists() and cid:
        try:
            sot_data  = _load_json(sot_results_path, {})
            sot_entry = sot_data.get("results", {}).get(cid)
            if sot_entry and sot_entry.get("status") == "ok":
                live_val = sot_entry.get("live_oracle")
                if live_val is not None and live_val != 0.0:
                    sot_result = {
                        "oracle_value": live_val,
                        "note": f"From Execute All Metrics run at {sot_data.get('executed_at', 'unknown')}.",
                    }
        except Exception as e:
            log.warning(f"[query] Could not read sot_results.json: {e}")

    if sot_result:
        return JSONResponse({
            "status":            "hit",
            "question":          req.question,
            "canonical_id":      cid,
            "oracle_value":      sot_result["oracle_value"],
            "resolution_method": resolution.method,
            "executed_live":     True,
            "confidence":        resolution.confidence,
            "source_goal":       resolution.source_goal,
            "wave":              resolution.wave,
            "ai_locked":         resolution.ai_locked,
            "formula_line":      resolution.formula_line,
            "fingerprint":       None,
            "note":              sot_result["note"],
        })

    # ── Priority 2: live execution against sot_csv/ ───────────────────────────
    exec_vertical = getattr(resolution, "slots", {}).get("domain", req.vertical) if hasattr(resolution, "slots") else req.vertical
    if not exec_vertical:
        exec_vertical = req.vertical

    exec_result = execute(
        formula_line=resolution.formula_line or "",
        sot_dir=sot_dir,
        vertical=exec_vertical,
        canonical_index=canonical_index,
        composition_rules=comp_rules,
    )

    if exec_result["status"] == "ok" and exec_result.get("oracle_value") not in (None, 0.0):
        oracle_value  = exec_result["oracle_value"]
        executed_live = True
        exec_note     = "Computed live from your SOT data."
    else:
        # ── Priority 3: canonical_index mock oracle ───────────────────────────
        oracle_value  = resolution.oracle_value
        executed_live = False
        exec_note     = (
            "Showing compiled mock oracle value. "
            "Upload your SOT files and run Execute All Metrics for real data."
        )

    return JSONResponse({
        "status":            "hit",
        "question":          req.question,
        "canonical_id":      cid,
        "oracle_value":      oracle_value,
        "resolution_method": resolution.method,
        "executed_live":     executed_live,
        "confidence":        resolution.confidence,
        "source_goal":       resolution.source_goal,
        "wave":              resolution.wave,
        "ai_locked":         resolution.ai_locked,
        "formula_line":      resolution.formula_line,
        "fingerprint":       exec_result.get("fingerprint"),
        "note":              exec_note,
    })


def _handle_miss(question: str, vertical: str, resolver) -> dict:
    """
    On MISS: build a schema_version from SOT fingerprints, send to the
    factory to compile the goal, receive the aterm, store it in the
    customer knowledge_store, and add it to the resolver index.

    In Option B (same server), this calls the factory pipeline directly.
    In production this would be a TLS call to the GPL Central Platform API.
    """
    try:
        # Build schema_version from SOT fingerprints
        from customer.formula_executor import _fingerprint
        schema_version = {}
        sot_vertical   = CUSTOMER_SOT_DIR / vertical
        if sot_vertical.exists():
            for csv_file in sot_vertical.glob("*.csv"):
                schema_version[csv_file.stem] = _fingerprint(csv_file)

        # Load customer atoms for the factory recompilation context
        from core.paths import CUSTOMER_ATOMS_PATH, CUSTOMER_ATOM_REL_PATH
        import services.atom_registry as ar
        import services.relationship_registry as rr

        # Temporarily switch registries to customer atoms
        orig_atoms_path = ar.ATOMS_PATH
        orig_rel_path   = rr._REL_PATH
        orig_rr_atoms   = rr.ATOMS_PATH
        ar.ATOMS_PATH   = CUSTOMER_ATOMS_PATH
        rr.ATOMS_PATH   = CUSTOMER_ATOMS_PATH
        rr._REL_PATH    = CUSTOMER_ATOM_REL_PATH

        try:
            from compiler.slot_constants import build_canonical_id
            from compiler.compiler_orchestrator import compile_vertical
            from compiler.verifier import verify_vertical
            import core.paths as p

            # Temporarily point compiler at customer data
            orig_aterms  = p.ATERMS_DIR
            orig_index   = p.CANONICAL_INDEX_PATH
            orig_lock    = p.LOCK_REGISTRY_PATH
            orig_mock    = p.MOCK_DATA_DIR

            from core.paths import (
                CUSTOMER_ATERMS_DIR, CUSTOMER_CANONICAL_INDEX,
                CUSTOMER_LOCK_REGISTRY, CUSTOMER_MOCK_DATA_DIR,
            )
            p.ATERMS_DIR           = CUSTOMER_ATERMS_DIR
            p.CANONICAL_INDEX_PATH = CUSTOMER_CANONICAL_INDEX
            p.LOCK_REGISTRY_PATH   = CUSTOMER_LOCK_REGISTRY
            p.MOCK_DATA_DIR        = CUSTOMER_MOCK_DATA_DIR

            # Run compile + verify for this vertical (incremental — adds new goal)
            compile_result = compile_vertical(vertical=vertical)
            verify_result  = verify_vertical(vertical=vertical)

        finally:
            # Restore
            p.ATERMS_DIR           = orig_aterms
            p.CANONICAL_INDEX_PATH = orig_index
            p.LOCK_REGISTRY_PATH   = orig_lock
            p.MOCK_DATA_DIR        = orig_mock
            ar.ATOMS_PATH          = orig_atoms_path
            rr.ATOMS_PATH          = orig_rr_atoms
            rr._REL_PATH           = orig_rel_path

        # Check if the goal is now in customer canonical_index
        from core.paths import CUSTOMER_CANONICAL_INDEX
        if CUSTOMER_CANONICAL_INDEX.exists():
            idx = json.loads(CUSTOMER_CANONICAL_INDEX.read_text(encoding="utf-8"))
            # Find the new entry matching our question (by source_goal or closest match)
            for cid, entry in idx.items():
                if question.lower() in entry.get("source_goal", "").lower():
                    # Add to knowledge_store
                    ks_idx_path = CUSTOMER_KNOWLEDGE_DIR / "canonical_index.json"
                    ks_idx      = _load_json(ks_idx_path, {})
                    ks_idx[cid] = entry
                    ks_idx_path.write_text(
                        json.dumps(ks_idx, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    # Also copy the aterm file
                    aterm_src = CUSTOMER_ATERMS_DIR / f"aterm_{cid}.json"
                    if aterm_src.exists():
                        import shutil as _sh
                        ks_aterms = CUSTOMER_KNOWLEDGE_DIR / "aterms"
                        ks_aterms.mkdir(exist_ok=True)
                        _sh.copy2(aterm_src, ks_aterms / f"aterm_{cid}.json")
                    # Add to resolver
                    resolver.add_to_index(cid, entry)
                    return {"status": "compiled", "cid": cid}

        return {"status": "error", "error": "Goal not found after recompilation"}

    except Exception as e:
        log.error(f"[customer_router] MISS handler failed: {e}")
        return {"status": "error", "error": str(e)}


# ── GET /api/customer/status ───────────────────────────────────────────────────

@router.get("/api/customer/status")
async def get_status(
    x_session_token: Optional[str] = Header(default=None),
    token:           Optional[str] = Query(default=None),
):
    """Return knowledge store stats for the logged-in customer."""
    customer_id   = get_customer_id(_resolve_token(x_session_token, token))
    knowledge_dir = _knowledge_dir(customer_id)
    sot_dir       = _sot_dir(customer_id)
    ks_idx        = _ks_index(customer_id)

    sot_tables = []
    if sot_dir.exists():
        sot_tables = [
            str(f.relative_to(sot_dir))
            for f in sot_dir.rglob("*.csv")
        ]

    if not ks_idx.exists():
        return JSONResponse({
            "status":            "empty",
            "metrics_available": 0,
            "ai_locked":         0,
            "lock_rate":         0.0,
            "sot_tables":        sot_tables,
        })

    idx    = _load_json(ks_idx, {})
    locked = sum(1 for v in idx.values() if v.get("ai_locked"))

    return JSONResponse({
        "status":            "ready",
        "metrics_available": len(idx),
        "ai_locked":         locked,
        "lock_rate":         round(locked / max(len(idx), 1) * 100, 1),
        "sot_tables":        sot_tables,
    })


# ── DELETE /api/customer/reset ─────────────────────────────────────────────────

@router.delete("/api/customer/reset")
async def reset(
    x_session_token: Optional[str] = Header(default=None),
    token:           Optional[str] = Query(default=None),
):
    """Wipe this customer's runtime data AND remove them from customers.json for a fresh start."""
    customer_id = get_customer_id(_resolve_token(x_session_token, token))
    cpaths      = get_customer_paths(customer_id)
    cleared     = []

    # 1. Wipe knowledge store, SOT data, packages
    for key in ["KNOWLEDGE_DIR", "SOT_DIR", "PACKAGES_DIR"]:
        d = cpaths[key]
        if d.exists():
            shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
            cleared.append(str(d))

    # 2. Wipe the entire per-customer data dir (atoms, schema, goals, aterms, etc.)
    data_dir = cpaths["DATA_DIR"]
    if data_dir.exists():
        shutil.rmtree(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
    cleared.append("data/")

    # 3. Recreate expected subdirectories
    for sub in ["mock_data", "seeds", "dialects", "goals", "aterms"]:
        (data_dir / sub).mkdir(parents=True, exist_ok=True)

    # 4. Remove customer from customers.json (wipes login + session)
    _customers_file = Path(__file__).parent.parent / "customers.json"
    try:
        if _customers_file.exists():
            db = json.loads(_customers_file.read_text(encoding="utf-8"))
            # Remove the customer entry
            db.pop(customer_id, None)
            # Also remove any session tokens belonging to this customer
            sessions = db.get("__sessions__", {})
            stale_tokens = [t for t, cid in sessions.items() if cid == customer_id]
            for t in stale_tokens:
                sessions.pop(t, None)
            db["__sessions__"] = sessions
            _customers_file.write_text(
                json.dumps(db, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            cleared.append("customers.json")
    except Exception as e:
        log.warning(f"[reset] Could not update customers.json: {e}")

    return JSONResponse({"status": "reset", "cleared": cleared})


# ═══════════════════════════════════════════════════════════════════════════════
# DEBUG ENDPOINTS — each pipeline stage independently, async via job_manager
# ═══════════════════════════════════════════════════════════════════════════════

from fastapi.responses import StreamingResponse


# ── SSE log stream (customer runtime's own stream on port 8081) ───────────────

@router.get("/api/customer/logs/stream")
async def stream_logs():
    """
    Server-Sent Events — streams live log lines to the customer UI terminal.
    Same format as factory's /api/logs/stream:
        data: level_class|logger_name|message
    """
    from core.log_stream import event_stream
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ── Job polling ───────────────────────────────────────────────────────────────

@router.get("/api/customer/jobs/{job_id}")
async def get_customer_job(job_id: str):
    """Poll a customer-runtime debug job for status + result."""
    from core.job_manager import get_job
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    return JSONResponse(job)


# ── Shared helper: build customer_schema from committed store ─────────────────

def _build_schema_from_committed(vertical: str, pending_path, committed_path):
    from customer.sot_ingestion import infer_relationships, build_customer_schema
    from customer import schema_store
    merged        = schema_store.merged_view(vertical, pending_path, committed_path)
    table_schemas = schema_store.as_list(merged)
    relationships = infer_relationships(table_schemas)
    return build_customer_schema(table_schemas, relationships), table_schemas, relationships


# ── POST /api/customer/debug/reconcile ───────────────────────────────────────

class DebugVerticalRequest(BaseModel):
    vertical: str
    token:    Optional[str] = None   # session token — resolves per-customer paths


@router.post("/api/customer/debug/reconcile")
async def debug_reconcile(req: DebugVerticalRequest):
    """
    Debug Step 1 — Atom Reconciliation.
    Runs VerticalSchemaAgent in CUSTOMER mode against the committed schema.
    Returns job_id immediately; poll /api/customer/jobs/{job_id} for result.
    """
    from core.job_manager import create_job, run_job

    from routers.auth_router import get_customer_id
    _cid  = get_customer_id(_resolve_token(None, req.token))
    _cp   = get_customer_paths(_cid)
    customer_schema, table_schemas, relationships = _build_schema_from_committed(req.vertical, _cp["PENDING_SCHEMA"], _cp["COMMITTED_SCHEMA"])
    if not table_schemas:
        raise HTTPException(400, "No committed tables found. Upload and generate first.")

    job_id = create_job(f"Reconcile — {req.vertical}")

    def _run():
        from customer.onboarding import _customer_path_context
        from agents.vertical_schema_agent import run as vsa_run
        with _customer_path_context(req.vertical):
            result = vsa_run(
                vertical=req.vertical,
                mode="CUSTOMER",
                customer_schema=customer_schema,
            )
        created = len(result.get("atoms_created", []))
        updated = len(result.get("atoms_updated", []))
        return {
            "vertical":      req.vertical,
            "atoms_created": created,
            "atoms_updated": updated,
            "total_atoms":   created + updated,
            "status":        result.get("status", "ok"),
            "mode":          "CUSTOMER",
        }

    run_job(job_id, _run)
    return JSONResponse({"job_id": job_id})


# ── POST /api/customer/debug/relationships ────────────────────────────────────

@router.post("/api/customer/debug/relationships")
async def debug_relationships(req: DebugVerticalRequest):
    """
    Debug Step 2 — Relationship Establishment.
    Re-scans all committed tables for FK relationships.
    """
    from core.job_manager import create_job, run_job

    job_id = create_job(f"Relationships — {req.vertical}")

    from routers.auth_router import get_customer_id
    _cid2 = get_customer_id(_resolve_token(None, req.token))
    _cp2  = get_customer_paths(_cid2)
    _pp2  = _cp2["PENDING_SCHEMA"]
    _ccp2 = _cp2["COMMITTED_SCHEMA"]

    def _run():
        from customer.sot_ingestion import infer_relationships
        from customer import schema_store
        merged        = schema_store.merged_view(req.vertical, _pp2, _ccp2)
        table_schemas = schema_store.as_list(merged)
        relationships = infer_relationships(table_schemas)
        return {
            "vertical":            req.vertical,
            "tables_scanned":      len(table_schemas),
            "relationships_found": len(relationships),
            "relationships":       relationships,
        }

    run_job(job_id, _run)
    return JSONResponse({"job_id": job_id})


# ── POST /api/customer/debug/mockdata ────────────────────────────────────────

@router.post("/api/customer/debug/mockdata")
async def debug_mockdata(req: DebugVerticalRequest):
    """
    Debug Step 3 — Mock Data Generation.
    Generates schema-matched mock CSVs from the reconciled customer atoms.
    """
    from core.job_manager import create_job, run_job

    job_id = create_job(f"Mock Data — {req.vertical}")

    def _run():
        from customer.onboarding import _customer_path_context
        from agents.mock_data_agent import run as mock_run
        with _customer_path_context(req.vertical):
            result = mock_run(vertical=req.vertical, mode="CUSTOMER")
        return {
            "vertical":        req.vertical,
            "atoms_processed": result.get("atoms_processed", 0),
            "total_rows":      result.get("total_rows", 0),
            "reused_atoms":    len(result.get("reused_atoms", [])),
            "status":          result.get("status", "ok"),
        }

    run_job(job_id, _run)
    return JSONResponse({"job_id": job_id})


# ── POST /api/customer/debug/enforce_enums ────────────────────────────────────

@router.post("/api/customer/debug/enforce_enums")
async def debug_enforce_enums(req: DebugVerticalRequest):
    """
    Debug Step 3b — Enum Enforcement.
    Reads field_values.json (real SOT values extracted at upload time)
    and rewrites every mock data CSV so categorical columns use the
    customer's actual values instead of LLM-invented ones.
    Runs across ALL tables in the vertical's mock_data folder at once.
    """
    from core.job_manager import create_job, run_job

    job_id = create_job(f"Enum Enforcement — {req.vertical}")

    def _run():
        import csv as _csv
        import json as _json
        import random as _random
        from pathlib import Path
        from core.paths import CUSTOMER_MOCK_DATA_DIR, CUSTOMER_FIELD_VALUES_PATH

        fv_path = CUSTOMER_FIELD_VALUES_PATH
        if not fv_path.exists():
            return {
                "status":         "skipped",
                "reason":         "field_values.json not found — upload files first",
                "tables_fixed":   0,
                "columns_fixed":  0,
                "details":        [],
            }

        # Convert flat {"atom.col": [values]} → nested {"atom": {"col": [values]}}
        flat_fv = _json.loads(fv_path.read_text(encoding="utf-8"))
        sot_enums: dict = {}
        for key, values in flat_fv.items():
            if "." in key:
                atom_key, col = key.split(".", 1)
                sot_enums.setdefault(atom_key, {})[col] = values
        mock_vertical_dir = CUSTOMER_MOCK_DATA_DIR / req.vertical

        if not mock_vertical_dir.exists():
            return {
                "status":         "skipped",
                "reason":         f"No mock data found for vertical={req.vertical}",
                "tables_fixed":   0,
                "columns_fixed":  0,
                "details":        [],
            }

        tables_fixed  = 0
        columns_fixed = 0
        details       = []

        for csv_file in sorted(mock_vertical_dir.glob("*.csv")):
            table_name  = csv_file.stem
            table_enums = sot_enums.get(table_name, {})
            if not table_enums:
                continue   # No enum constraints for this table

            # Read existing mock rows
            rows = []
            with open(csv_file, newline="", encoding="utf-8") as f:
                reader     = _csv.DictReader(f)
                fieldnames = list(reader.fieldnames or [])
                rows       = list(reader)

            if not rows:
                continue

            table_detail = {"table": table_name, "columns": []}
            did_fix = False

            for col, allowed_values in table_enums.items():
                if col not in fieldnames or not allowed_values:
                    continue

                # Capture before sample (first unique values seen)
                before_sample = list({r[col] for r in rows[:10]})[:3]

                # Replace all values with random samples from real SOT
                for row in rows:
                    row[col] = _random.choice(allowed_values)

                after_sample = list({r[col] for r in rows[:10]})[:3]

                table_detail["columns"].append({
                    "column":  col,
                    "before":  before_sample,
                    "after":   after_sample,
                    "allowed": allowed_values,
                })
                columns_fixed += 1
                did_fix = True

            if did_fix:
                # Rewrite CSV
                with open(csv_file, "w", newline="", encoding="utf-8") as f:
                    writer = _csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                tables_fixed += 1
                details.append(table_detail)
                log.info(
                    f"[debug_enforce_enums] {table_name}: "
                    f"{len(table_detail['columns'])} columns enforced"
                )

        return {
            "status":        "ok",
            "tables_fixed":  tables_fixed,
            "columns_fixed": columns_fixed,
            "details":       details,
        }

    run_job(job_id, _run)
    return JSONResponse({"job_id": job_id})


# ── POST /api/customer/debug/seed ────────────────────────────────────────────

@router.post("/api/customer/debug/seed")
async def debug_seed(req: DebugVerticalRequest):
    """
    Debug Step 4 — Seed Generation.
    Generates the seed JSON file from customer atoms.
    """
    from core.job_manager import create_job, run_job

    job_id = create_job(f"Seed — {req.vertical}")

    def _run():
        from customer.onboarding import _customer_path_context
        from agents.seed_agent import run as seed_run
        with _customer_path_context(req.vertical):
            result = seed_run(vertical=req.vertical)
        seed = result.get("seed", {})
        entities = seed.get("entities", {})
        total_measures = sum(len(e.get("measures", [])) for e in entities.values())
        return {
            "vertical":       req.vertical,
            "entities":       len(entities),
            "total_measures": total_measures,
            "status":         result.get("status", "ok"),
        }

    run_job(job_id, _run)
    return JSONResponse({"job_id": job_id})


# ── POST /api/customer/debug/vocabulary ──────────────────────────────────────

@router.post("/api/customer/debug/vocabulary")
async def debug_vocabulary(req: DebugVerticalRequest):
    """
    Debug Step 5 — Vocabulary / GPL Dialect Generation.
    """
    from core.job_manager import create_job, run_job

    job_id = create_job(f"Vocabulary — {req.vertical}")

    def _run():
        from customer.onboarding import _customer_path_context
        from agents.vocabulary_agent import run as vocab_run
        with _customer_path_context(req.vertical):
            result = vocab_run(vertical=req.vertical)
        return {
            "vertical":      req.vertical,
            "total_entries": result.get("total_entries", 0),
            "status":        result.get("status", "ok"),
        }

    run_job(job_id, _run)
    return JSONResponse({"job_id": job_id})


# ── POST /api/customer/debug/goals ───────────────────────────────────────────

@router.post("/api/customer/debug/goals")
async def debug_goals(req: DebugVerticalRequest):
    """
    Debug Step 6 — Goal Generation (Waves 1-9 + A-E).
    """
    from core.job_manager import create_job, run_job

    job_id = create_job(f"Goals — {req.vertical}")

    def _run():
        from customer.onboarding import _customer_path_context
        from services.goal_generator import generate_goals
        from agents.domain_goals_agent import generate_domain_goals

        with _customer_path_context(req.vertical):
            # Waves 1-9
            result_19 = generate_goals(vertical=req.vertical)
            total_19  = result_19.get("total_goals", 0)
            all_cids  = list(result_19.get("all_cids", []))

            # Waves A-E
            result_ae = generate_domain_goals(
                vertical=req.vertical,
                wave_19_cids=all_cids,
            )
            total_ae = result_ae.get("total_ae_goals", 0)

        waves_19 = result_19.get("waves_19", {})
        return {
            "vertical":    req.vertical,
            "total_goals": total_19 + total_ae,
            "waves_19":    total_19,
            "waves_ae":    total_ae,
            "wave_counts": {k: len(v) for k, v in waves_19.items()},
        }

    run_job(job_id, _run)
    return JSONResponse({"job_id": job_id})


# ── POST /api/customer/debug/compile ─────────────────────────────────────────

@router.post("/api/customer/debug/compile")
async def debug_compile(req: DebugVerticalRequest):
    """
    Debug Step 7 — Compilation (Branch A → C → B).
    Compiles all goals into aterms.
    """
    from core.job_manager import create_job, run_job

    job_id = create_job(f"Compile — {req.vertical}")

    def _run():
        from customer.onboarding import _customer_path_context
        from compiler.compiler_orchestrator import compile_vertical
        with _customer_path_context(req.vertical):
            result = compile_vertical(vertical=req.vertical)
        return {
            "vertical":       req.vertical,
            "total_compiled": result.get("total_compiled", 0),
            "branch_a":       result.get("branch_a", 0),
            "branch_b":       result.get("branch_b", 0),
            "branch_c":       result.get("branch_c", 0),
            "pending":        result.get("pending", 0),
        }

    run_job(job_id, _run)
    return JSONResponse({"job_id": job_id})


# ── POST /api/customer/debug/verify ──────────────────────────────────────────

@router.post("/api/customer/debug/verify")
async def debug_verify(req: DebugVerticalRequest):
    """
    Debug Step 8 — Verification.
    Proves aterms and sets ai_locked flags.
    """
    from core.job_manager import create_job, run_job

    job_id = create_job(f"Verify — {req.vertical}")

    def _run():
        from customer.onboarding import _customer_path_context
        from compiler.verifier import verify_vertical
        with _customer_path_context(req.vertical):
            result = verify_vertical(vertical=req.vertical)
        locked  = result.get("ai_locked_count", 0)
        total   = result.get("total", 0)
        pending = result.get("pending_count", 0)
        return {
            "vertical":        req.vertical,
            "total":           total,
            "ai_locked":       locked,
            "pending":         pending,
            "lock_rate":       round(locked / max(total, 1) * 100, 1),
        }

    run_job(job_id, _run)
    return JSONResponse({"job_id": job_id})


# ── POST /api/customer/debug/deploy ──────────────────────────────────────────

@router.post("/api/customer/debug/deploy")
async def debug_deploy(
    req: DebugVerticalRequest,
    x_session_token: Optional[str] = Header(default=None),
):
    """
    Debug Step 9 — Deploy.
    Packages aterms into a zip and unpacks to customer knowledge_store/.
    """
    from core.job_manager import create_job, run_job

    job_id = create_job(f"Deploy — {req.vertical}")

    def _run():
        from customer.onboarding import _customer_path_context, _unpack_package
        from compiler.deployment_engine import package_vertical
        with _customer_path_context(req.vertical):
            result   = package_vertical(vertical=req.vertical)
        pkg_path = result.get("package_path", "")
        ks_path  = _unpack_package(pkg_path)

        # Count metrics in knowledge store — use per-customer path
        import json
        from routers.auth_router import get_customer_id
        _cid     = get_customer_id(_resolve_token(x_session_token, req.token))
        _ks_dir  = _knowledge_dir(_cid)
        ks_index = _ks_dir / "canonical_index.json"
        metrics  = 0
        locked   = 0
        if ks_index.exists():
            idx     = json.loads(ks_index.read_text(encoding="utf-8"))
            metrics = len(idx)
            locked  = sum(1 for v in idx.values() if v.get("ai_locked"))

        # Commit pending schemas — move from pending_schema.json to
        # committed_schema.json so the Upload tab shows tables as generated.
        from customer import schema_store
        _dp = get_customer_paths(_cid)
        schema_store.commit_pending(req.vertical, _dp["PENDING_SCHEMA"], _dp["COMMITTED_SCHEMA"])
        log.info(f"[debug_deploy] Committed pending schemas for vertical={req.vertical}")

        from pathlib import Path
        return {
            "vertical":          req.vertical,
            "package_name":      Path(pkg_path).name if pkg_path else "",
            "metric_count":      result.get("metric_count", metrics),
            "locked_count":      result.get("locked_count", locked),
            "knowledge_store":   ks_path,
        }

    run_job(job_id, _run)
    return JSONResponse({"job_id": job_id})


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTE ALL — run every compiled metric against real SOT data
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/api/customer/execute/all")
async def execute_all(
    req: DebugVerticalRequest,
    x_session_token: Optional[str] = Header(default=None),
):
    """
    Runs every entry in the knowledge_store canonical_index through
    formula_executor against the real sot_csv/ files.

    Runs across ALL verticals that have SOT CSV data — not just the
    vertical passed in the request body. The request vertical is used
    only as a fallback label.

    Writes results to customer_runtime/data/sot_results.json.
    Returns job_id immediately — poll /api/customer/jobs/{job_id}.
    """
    from core.job_manager import create_job, run_job

    job_id = create_job(f"Execute All")

    # Resolve customer paths before spawning background thread
    from routers.auth_router import get_customer_id
    from core.paths import get_customer_paths
    _cid           = get_customer_id(_resolve_token(x_session_token, req.token))
    _cpaths        = get_customer_paths(_cid)
    _ks_dir        = _cpaths["KNOWLEDGE_DIR"]
    _sot_dir       = _cpaths["SOT_DIR"]
    _data_dir      = _cpaths["DATA_DIR"]
    _results_dir   = _cpaths["METRIC_RESULTS_DIR"]
    _results_index = _cpaths["METRIC_RESULTS_INDEX"]
    _results_dir.mkdir(parents=True, exist_ok=True)

    def _run():
        import json
        from datetime import datetime, timezone
        from pathlib import Path
        from customer.formula_executor import execute, _fingerprint

        ks_dir  = _ks_dir
        sot_dir = _sot_dir

        idx_path  = ks_dir / "canonical_index.json"
        comp_path = ks_dir / "composition_rules.json"

        out_path = _data_dir / "sot_results.json"

        if not idx_path.exists():
            raise RuntimeError("Knowledge store is empty — generate first.")

        idx  = json.loads(idx_path.read_text(encoding="utf-8"))
        comp = json.loads(comp_path.read_text(encoding="utf-8")) if comp_path.exists() else {}

        # Run ALL entries in the canonical index — no vertical filter
        entries = dict(idx)

        # SOT fingerprints across all sot_csv subdirs
        sot_fp = {}
        if sot_dir.exists():
            for v_dir in sot_dir.iterdir():
                if v_dir.is_dir():
                    for csv_file in sorted(v_dir.glob("*.csv")):
                        sot_fp[f"{v_dir.name}/{csv_file.stem}"] = _fingerprint(csv_file)

        TOLERANCE = 0.05   # 5% relative tolerance for divergence check

        results  = {}
        executed = 0
        failed   = 0
        diverged = 0

        total = len(entries)
        log.info(f"[execute_all] Starting — {total} total metrics across all verticals")

        for i, (cid, entry) in enumerate(entries.items(), 1):
            formula = entry.get("formula_line", "")
            mock_oracle = entry.get("oracle_value")
            # Use the domain from the entry's slots as the vertical for SOT lookup
            entry_vertical = entry.get("slots", {}).get("domain") or req.vertical

            if not formula:
                results[cid] = {
                    "live_oracle":  None,
                    "mock_oracle":  mock_oracle,
                    "method":       None,
                    "status":       "skipped",
                    "diverged":     False,
                    "error":        "No formula_line",
                }
                failed += 1
                continue

            exec_result = execute(
                formula_line=formula,
                sot_dir=sot_dir,
                vertical=entry_vertical,
                canonical_index=idx,
                composition_rules=comp,
            )

            if exec_result["status"] == "ok":
                live = exec_result["oracle_value"]
                executed += 1

                # Divergence check — relative tolerance
                div = False
                if mock_oracle is not None and live is not None:
                    try:
                        denom = abs(float(mock_oracle)) if float(mock_oracle) != 0 else 1.0
                        div   = abs(float(live) - float(mock_oracle)) / denom > TOLERANCE
                    except (TypeError, ValueError):
                        div = live != mock_oracle

                if div:
                    diverged += 1

                results[cid] = {
                    "live_oracle":  live,
                    "mock_oracle":  mock_oracle,
                    "method":       exec_result.get("method"),
                    "status":       "ok",
                    "diverged":     div,
                    "error":        None,
                    "source_goal":  entry.get("source_goal", ""),
                    "wave":         entry.get("wave", ""),
                    "ai_locked":    entry.get("ai_locked", False),
                }
            else:
                failed += 1
                results[cid] = {
                    "live_oracle":  None,
                    "mock_oracle":  mock_oracle,
                    "method":       None,
                    "status":       "error",
                    "diverged":     False,
                    "error":        exec_result.get("error", "Unknown error"),
                    "source_goal":  entry.get("source_goal", ""),
                    "wave":         entry.get("wave", ""),
                    "ai_locked":    entry.get("ai_locked", False),
                }

            if i % 20 == 0:
                log.info(f"[execute_all] {i}/{total} done — {executed} ok, {failed} failed")

        summary = {
            "total":    total,
            "executed": executed,
            "failed":   failed,
            "diverged": diverged,
        }

        output = {
            "vertical":        "all",
            "executed_at":     datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sot_fingerprint": sot_fp,
            "summary":         summary,
            "results":         results,
        }

        # ── Persist this run ──────────────────────────────────────────────────
        # 1. Overwrite sot_results.json (latest run — backward compat)
        out_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # 2. Save a timestamped copy in metric_results/
        run_ts      = output["executed_at"].replace(":", "-").replace("T", "_")
        run_file    = _results_dir / f"run_{run_ts}.json"
        run_file.write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # 3. Update _index.json — append entry for this run
        index_path = _results_index
        try:
            index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else []
        except Exception:
            index = []
        index.append({
            "run_id":     run_ts,
            "file":       run_file.name,
            "executed_at": output["executed_at"],
            "summary":    summary,
        })
        # Keep newest first, cap at 100 entries
        index = sorted(index, key=lambda x: x["executed_at"], reverse=True)[:100]
        index_path.write_text(
            json.dumps(index, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        log.info(
            f"[execute_all] Done — {executed} executed, {failed} failed, {diverged} diverged. "
            f"Saved to {run_file.name} | index has {len(index)} runs."
        )

        return {
            "summary":     summary,
            "run_id":      run_ts,
            "output_path": str(run_file),
        }

    run_job(job_id, _run)
    return JSONResponse({"job_id": job_id})


@router.get("/api/customer/execute/results")
async def get_sot_results(
    vertical: str = "all",
    x_session_token: Optional[str] = Header(default=None),
    token:           Optional[str] = Query(default=None),
):
    """
    Return the latest execution results (sot_results.json — most recent run).
    The vertical param is accepted but ignored — results always cover all verticals.
    """
    import json
    customer_id = get_customer_id(_resolve_token(x_session_token, token))
    out_path    = get_customer_paths(customer_id)["DATA_DIR"] / "sot_results.json"
    if not out_path.exists():
        raise HTTPException(404, "No SOT results yet — run Execute All first.")
    data = json.loads(out_path.read_text(encoding="utf-8"))
    return JSONResponse(data)


@router.get("/api/customer/execute/history")
async def get_execution_history(
    x_session_token: Optional[str] = Header(default=None),
    token:           Optional[str] = Query(default=None),
):
    """
    Return the index of all Execute All runs for this customer.
    Each entry: { run_id, file, executed_at, summary }
    Sorted newest first, capped at 100 entries.
    """
    import json
    customer_id  = get_customer_id(_resolve_token(x_session_token, token))
    index_path   = get_customer_paths(customer_id)["METRIC_RESULTS_INDEX"]
    if not index_path.exists():
        return JSONResponse([])
    index = json.loads(index_path.read_text(encoding="utf-8"))
    return JSONResponse(index)


@router.get("/api/customer/execute/history/{run_id}")
async def get_execution_run(
    run_id: str,
    x_session_token: Optional[str] = Header(default=None),
    token:           Optional[str] = Query(default=None),
):
    """
    Return the full results for a specific historical run.
    run_id is the timestamp string from the history index (e.g. '2026-08-15_01-31-39').
    """
    import json
    customer_id  = get_customer_id(_resolve_token(x_session_token, token))
    results_dir  = get_customer_paths(customer_id)["METRIC_RESULTS_DIR"]
    run_file     = results_dir / f"run_{run_id}.json"
    if not run_file.exists():
        raise HTTPException(404, f"Run '{run_id}' not found.")
    data = json.loads(run_file.read_text(encoding="utf-8"))
    return JSONResponse(data)


@router.get("/api/customer/execute/download")
async def download_sot_results(
    x_session_token: Optional[str] = Header(default=None),
    token:           Optional[str] = Query(default=None),
):
    """
    Stream sot_results.json as a file download.
    """
    from fastapi.responses import FileResponse
    customer_id = get_customer_id(_resolve_token(x_session_token, token))
    out_path    = get_customer_paths(customer_id)["DATA_DIR"] / "sot_results.json"
    if not out_path.exists():
        raise HTTPException(404, "No SOT results yet — run Execute All first.")
    return FileResponse(
        path=str(out_path),
        filename="sot_results.json",
        media_type="application/json",
    )


# ── GET /api/customer/data/history ────────────────────────────────────────────

@router.get("/api/customer/data/history")
async def get_data_history(
    atom_id:         Optional[str] = Query(default=None),
    x_session_token: Optional[str] = Header(default=None),
    token:           Optional[str] = Query(default=None),
):
    """
    Return upload history for a customer.

    GET /api/customer/data/history             → all atoms, newest first
    GET /api/customer/data/history?atom_id=xxx → one atom only

    Each entry:
      {
        upload_id, atom_id, timestamp,
        rows_added, rows_updated, rows_unchanged, rows_deleted,
        additions: [{primary_key, new_row}],
        changes:   [{primary_key, old, new}],
        deletions: [{primary_key, deleted_row}]
      }
    """
    customer_id = get_customer_id(_resolve_token(x_session_token, token))
    cpaths      = get_customer_paths(customer_id)
    data_dir    = cpaths["DATA_DIR"]

    from customer.data_history import get_history_for_atom, get_all_history

    if atom_id:
        history = get_history_for_atom(data_dir, atom_id)
    else:
        history = get_all_history(data_dir)

    return JSONResponse({"history": history, "count": len(history)})


@router.get("/api/customer/data/store")
async def get_data_store_info(
    x_session_token: Optional[str] = Header(default=None),
    token:           Optional[str] = Query(default=None),
):
    """
    Return summary of what is stored in data_store.json for this customer.

    Returns one entry per atom:
      { atom_id, primary_key, total_rows, last_updated, column_hash }
    """
    customer_id = get_customer_id(_resolve_token(x_session_token, token))
    cpaths      = get_customer_paths(customer_id)
    data_dir    = cpaths["DATA_DIR"]

    from customer.data_store import list_atoms, get_atom_info

    atoms   = list_atoms(data_dir)
    summary = [get_atom_info(data_dir, a) for a in atoms]
    summary = [s for s in summary if s]  # drop None

    return JSONResponse({"atoms": summary, "count": len(summary)})


# ══════════════════════════════════════════════════════════════════════════════
# Presentation Compiler — Marketplace + Install + Generate
# ══════════════════════════════════════════════════════════════════════════════

def _installed_dir(customer_id: str) -> Path:
    """Directory where a customer's installed ptem blueprints are stored."""
    d = get_customer_paths(customer_id)["DATA_DIR"] / "installed_ptems"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _installed_index_path(customer_id: str) -> Path:
    return _installed_dir(customer_id) / "_index.json"


def _load_installed_index(customer_id: str) -> dict:
    p = _installed_index_path(customer_id)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_installed_index(customer_id: str, index: dict) -> None:
    _installed_index_path(customer_id).write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
    )


@router.get("/api/customer/reports/marketplace")
async def list_marketplace_reports(
    x_session_token: Optional[str] = Header(default=None),
    token:           Optional[str] = Query(default=None),
):
    """
    GET /api/customer/reports/marketplace
    Fetches the lightweight marketplace list from the factory,
    enriched with:
      - per-customer oracle coverage (using their actual data)
      - which templates are already installed
      - missing data requirements per template
    """
    import re as _re
    from customer.ptem_oracle_builder import build_oracle_values

    customer_id  = get_customer_id(_resolve_token(x_session_token, token))
    cpaths       = get_customer_paths(customer_id)

    # ── Data-compatibility helpers ────────────────────────────────────────────
    # Human-readable labels for SOT table names shown in "missing data" warnings
    _TABLE_LABELS = {
        "supply_chain_shipments_WMS_record":          "Shipments",
        "supply_chain_invoices_ERP_record":           "Invoices",
        "supply_chain_purchase_orders_ERP_record":    "Purchase Orders",
        "supply_chain_suppliers_ERP_dimension":       "Suppliers",
        "supply_chain_inventory_levels_WMS_state":    "Inventory",
        "logistics_incidents_ERP_record":             "Incidents",
        "logistics_vehicle_trips_WMS_event":          "Vehicle Trips",
        "logistics_vehicles_MANUAL_dimension":        "Vehicles",
        "retail_shopify_order_status_history_ERP_state": "Orders",
        "retail_shopify_customers_CRM_dimension":     "Customers",
    }

    # Build oracle_id → SOT table map from aterm files (factory side)
    _oracle_to_table: dict = {}
    _aterm_dir = Path(__file__).resolve().parents[1] / "data" / "aterms"
    if _aterm_dir.exists():
        for _af in _aterm_dir.glob("aterm_*.json"):
            try:
                _ad = json.loads(_af.read_text(encoding="utf-8"))
                _cid = _ad.get("canonical_id", "")
                _formula = _ad.get("formula_line", "")
                # Match both "FROM table_name" and "DEDUP table_name"
                _m = _re.search(r'(?:FROM|DEDUP)\s+([\w_]+)', _formula)
                if _m and _cid:
                    _oracle_to_table[_cid] = _m.group(1).rstrip(")")
            except Exception:
                pass

    def _get_missing_tables(required_oracle_ids: list, customer_tables: set) -> list:
        """Return list of SOT table names needed but not uploaded by this customer."""
        needed = set()
        for oid in required_oracle_ids:
            tbl = _oracle_to_table.get(oid)
            if tbl and tbl not in customer_tables:
                needed.add(tbl)
        return sorted(needed)

    # Collect the atom_canonical_ids the customer has actually uploaded
    _upload_reg_path = cpaths["DATA_DIR"] / "upload_registry.json"
    _customer_tables: set = set()
    if _upload_reg_path.exists():
        try:
            _reg = json.loads(_upload_reg_path.read_text(encoding="utf-8"))
            _uploads = _reg if isinstance(_reg, list) else _reg.get("uploads", [])
            for _u in _uploads:
                for _sh in _u.get("sheets", []):
                    _atom_cid = _sh.get("atom_canonical_id", "")
                    if _atom_cid:
                        _customer_tables.add(_atom_cid)
        except Exception:
            pass

    # Load marketplace list directly from local catalog (no external factory needed)
    try:
        from compiler.ptem_compiler_service import list_ptems
        factory_ptems = list_ptems()
    except Exception as e:
        raise HTTPException(500, f"Could not load ptem catalog: {e}")

    # Build customer's oracle pool for coverage computation
    built_oracles = build_oracle_values(cpaths["DATA_DIR"])
    ci_path = cpaths["CANONICAL_INDEX"]
    ci = json.loads(ci_path.read_text(encoding="utf-8")) if ci_path.exists() else {}
    oracle_pool = {**built_oracles, **{k: v.get("oracle_value") for k, v in ci.items()}}

    # Load installed index
    installed_index = _load_installed_index(customer_id)

    # Enrich each ptem with customer-specific coverage
    result = []
    for p in factory_ptems:
        cid           = p["canonical_id"]
        req_oracles   = p.get("coverage", {}).get("missing_oracles", []) +                         [{"canonical_id": oid} for oid in p.get("coverage", {}).get("available_ids", [])]

        # Recompute coverage against THIS customer's actual data
        # Use the factory's required_oracles list embedded in coverage
        all_oracle_ids = (
            [o["canonical_id"] for o in p.get("coverage", {}).get("missing_oracles", [])] +
            p.get("coverage", {}).get("available_ids", [])
        )
        cust_available = sum(1 for oid in all_oracle_ids if oracle_pool.get(oid) is not None)
        cust_total     = len(all_oracle_ids)
        cust_pct       = int(cust_available / cust_total * 100) if cust_total else 100

        # Which oracles are missing for THIS customer
        missing_for_customer = [
            o for o in p.get("coverage", {}).get("missing_oracles", [])
            if oracle_pool.get(o["canonical_id"]) is None
        ]

        # ── Data compatibility check ──────────────────────────────────────────
        # Resolve which SOT tables this PTEM's required_oracles need
        req_oracle_ids = [o["canonical_id"] for o in p.get("required_oracles", [])]
        missing_tables = _get_missing_tables(req_oracle_ids, _customer_tables)
        data_compatible = len(missing_tables) == 0
        # Build human-readable labels for missing tables
        missing_data_labels = [_TABLE_LABELS.get(t, t.replace("_", " ").title()) for t in missing_tables]

        result.append({
            "canonical_id":        cid,
            "title":               p["meta"]["title"],
            "description":         p["meta"]["description"],
            "audience_label":      p["meta"]["audience_label"],
            "verticals":           p["meta"].get("verticals", []),
            "tags":                p["meta"].get("tags", []),
            "slots":               p.get("slots", {}),
            "coverage_pct":        cust_pct,
            "oracle_coverage":     f"{cust_available}/{cust_total}",
            "can_generate":        cust_pct >= 30,
            "missing_oracles":     missing_for_customer,
            "installed":           cid in installed_index,
            "installed_at":        installed_index.get(cid, {}).get("installed_at"),
            "formats":             p.get("delivery", {}).get("formats", ["html"]),
            "sections":            p.get("sections", []),
            "data_compatible":     data_compatible,
            "missing_tables":      missing_tables,
            "missing_data_labels": missing_data_labels,
            "star_rating":         p.get("star_rating", 3),
        })

    return JSONResponse({
        "customer_id": customer_id,
        "total":       len(result),
        "installed":   sum(1 for r in result if r["installed"]),
        "reports":     result,
    })


@router.post("/api/customer/reports/install")
async def install_report(
    request:         dict,
    x_session_token: Optional[str] = Header(default=None),
    token:           Optional[str] = Query(default=None),
):
    """
    POST /api/customer/reports/install
    Body: { "canonical_id": "report_ops_logistics_monthly_detailed_lastmonth" }

    Downloads the full ptem blueprint from the factory and saves it
    to the customer's installed_ptems/ directory.
    The ptem is now available for Generate.
    """
    from datetime import datetime

    customer_id  = get_customer_id(_resolve_token(x_session_token, token))
    canonical_id = request.get("canonical_id")
    if not canonical_id:
        raise HTTPException(400, "canonical_id is required")

    # Fetch full blueprint directly from local catalog (no external factory needed)
    try:
        from compiler.ptem_compiler_service import get_ptem
        ptem = get_ptem(canonical_id)
        if ptem is None:
            raise HTTPException(404, f"Template '{canonical_id}' not found in catalog")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Could not load ptem catalog: {e}")

    # Save blueprint to customer's installed_ptems/
    install_dir  = _installed_dir(customer_id)
    ptem_path    = install_dir / f"{canonical_id}.json"
    ptem_path.write_text(json.dumps(ptem, indent=2, ensure_ascii=False), encoding="utf-8")

    # Update installed index
    index = _load_installed_index(customer_id)
    index[canonical_id] = {
        "canonical_id":  canonical_id,
        "title":         ptem.get("meta", {}).get("title", canonical_id),
        "description":   ptem.get("meta", {}).get("description", ""),
        "version":       ptem.get("version", 1),
        "installed_at":  datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _save_installed_index(customer_id, index)

    log.info("[install] %s installed ptem: %s", customer_id, canonical_id)
    return JSONResponse({
        "success":      True,
        "canonical_id": canonical_id,
        "title":        ptem.get("meta", {}).get("title"),
        "installed_at": index[canonical_id]["installed_at"],
    })


@router.delete("/api/customer/reports/install/{canonical_id}")
async def uninstall_report(
    canonical_id:    str,
    x_session_token: Optional[str] = Header(default=None),
    token:           Optional[str] = Query(default=None),
):
    """
    DELETE /api/customer/reports/install/{canonical_id}
    Removes an installed ptem from the customer's runtime.
    """
    customer_id = get_customer_id(_resolve_token(x_session_token, token))
    install_dir = _installed_dir(customer_id)
    ptem_path   = install_dir / f"{canonical_id}.json"

    if ptem_path.exists():
        ptem_path.unlink()

    index = _load_installed_index(customer_id)
    index.pop(canonical_id, None)
    _save_installed_index(customer_id, index)

    return JSONResponse({"success": True, "canonical_id": canonical_id})


@router.get("/api/customer/reports/installed")
async def list_installed_reports(
    x_session_token: Optional[str] = Header(default=None),
    token:           Optional[str] = Query(default=None),
):
    """
    GET /api/customer/reports/installed
    Lists all installed ptem templates for this customer.
    Each entry is enriched with current oracle coverage from live data.
    """
    from customer.ptem_oracle_builder import build_oracle_values

    customer_id = get_customer_id(_resolve_token(x_session_token, token))
    cpaths      = get_customer_paths(customer_id)
    index       = _load_installed_index(customer_id)
    install_dir = _installed_dir(customer_id)

    # Build customer oracle pool
    built_oracles = build_oracle_values(cpaths["DATA_DIR"])
    ci_path = cpaths["CANONICAL_INDEX"]
    ci = json.loads(ci_path.read_text(encoding="utf-8")) if ci_path.exists() else {}
    oracle_pool = {**built_oracles, **{k: v.get("oracle_value") for k, v in ci.items()}}

    result = []
    for cid, meta in index.items():
        ptem_path = install_dir / f"{cid}.json"
        if not ptem_path.exists():
            continue
        try:
            ptem = json.loads(ptem_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        req_oracles    = ptem.get("required_oracles", [])
        available      = sum(1 for o in req_oracles if oracle_pool.get(o["canonical_id"]) is not None)
        total          = len(req_oracles)
        pct            = int(available / total * 100) if total else 100
        missing        = [
            {"canonical_id": o["canonical_id"], "label": o.get("label", o["canonical_id"])}
            for o in req_oracles if oracle_pool.get(o["canonical_id"]) is None
        ]

        result.append({
            "canonical_id":   cid,
            "title":          meta.get("title", cid),
            "description":    meta.get("description", ""),
            "installed_at":   meta.get("installed_at"),
            "version":        meta.get("version", 1),
            "coverage_pct":   pct,
            "oracle_coverage": f"{available}/{total}",
            "can_generate":   pct >= 30,
            "missing_oracles": missing,
        })

    return JSONResponse({
        "customer_id": customer_id,
        "total":       len(result),
        "reports":     sorted(result, key=lambda x: x.get("installed_at", ""), reverse=True),
    })


@router.post("/api/customer/reports/generate")
async def generate_report(
    request:         dict,
    x_session_token: Optional[str] = Header(default=None),
    token:           Optional[str] = Query(default=None),
):
    """
    POST /api/customer/reports/generate
    Body: { "canonical_id": "report_ops_logistics_monthly_detailed_lastmonth" }

    Generates a report from an INSTALLED ptem using live customer data.
    Always uses the freshest oracle values — data updates automatically.
    """
    customer_id  = get_customer_id(_resolve_token(x_session_token, token))
    canonical_id   = request.get("canonical_id")
    chart_overrides = request.get("chart_types", [])  # e.g. ["pie_chart", "line_chart"]
    if not canonical_id:
        raise HTTPException(400, "canonical_id is required")

    # Verify template is installed
    index = _load_installed_index(customer_id)
    if canonical_id not in index:
        raise HTTPException(400, f"Template '{canonical_id}' is not installed. Install it first from the marketplace.")

    from customer.ptem_executor import PtemExecutor
    executor = PtemExecutor(customer_id=customer_id)
    result   = executor.execute(canonical_id, chart_overrides=chart_overrides or None)

    if not result.get("success"):
        raise HTTPException(400, result.get("error", "Report generation failed"))

    return JSONResponse({
        "success":        True,
        "canonical_id":   result["canonical_id"],
        "title":          result["title"],
        "html":           result["html"],
        "html_path":      result["html_path"],
        "pdf_path":       result.get("pdf_path"),
        "narrative":      result["narrative"],
        "oracle_summary": result["oracle_summary"],
        "generated_at":   result["generated_at"],
    })


@router.get("/api/customer/reports/files")
async def list_generated_report_files(
    x_session_token: Optional[str] = Header(default=None),
    token:           Optional[str] = Query(default=None),
):
    """
    GET /api/customer/reports/files
    Lists all previously generated report files for this customer.
    """
    from datetime import datetime
    customer_id = get_customer_id(_resolve_token(x_session_token, token))
    cpaths      = get_customer_paths(customer_id)
    report_dir  = cpaths["BASE"] / "reports"

    if not report_dir.exists():
        return JSONResponse({"reports": [], "total": 0})

    files = []
    for f in sorted(report_dir.iterdir(), key=lambda x: -x.stat().st_mtime):
        if f.suffix in (".html", ".pdf"):
            files.append({
                "filename":     f.name,
                "size_kb":      round(f.stat().st_size / 1024, 1),
                "generated_at": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                "format":       f.suffix.lstrip(".").upper(),
                "download_url": f"/api/customer/reports/download/{f.name}",
            })
    return JSONResponse({"reports": files, "total": len(files)})


@router.get("/api/customer/reports/download/{filename}")
async def download_report_file(
    filename:        str,
    x_session_token: Optional[str] = Header(default=None),
    token:           Optional[str] = Query(default=None),
):
    """
    GET /api/customer/reports/download/{filename}
    Download a generated report file (HTML or PDF).
    """
    from fastapi.responses import FileResponse
    customer_id = get_customer_id(_resolve_token(x_session_token, token))
    cpaths      = get_customer_paths(customer_id)
    file_path   = cpaths["BASE"] / "reports" / filename

    if not file_path.exists():
        raise HTTPException(404, f"Report file '{filename}' not found")

    media_type = "application/pdf" if filename.endswith(".pdf") else "text/html"
    return FileResponse(path=str(file_path), filename=filename, media_type=media_type)


# ── Custom Report Request Endpoints ──────────────────────────────────────────

@router.post("/api/customer/reports/request")
async def request_custom_report(
    request:         dict,
    x_session_token: Optional[str] = Header(default=None),
    token:           Optional[str] = Query(default=None),
):
    """
    POST /api/customer/reports/request
    Customer submits a plain-English custom report request.
    Sends it to the factory which compiles a custom ptem and ships it back.

    Body: {
      "request_text": "Show me my top carriers by cost with delivery rate",
      "factory_url":  "http://localhost:8080"   (optional, uses settings default)
    }

    Returns: { job_id, status: "pending" }
    """
    import urllib.request, urllib.error
    from core.config import settings

    customer_id  = get_customer_id(_resolve_token(x_session_token, token))
    request_text = request.get("request_text", "").strip()
    if not request_text:
        raise HTTPException(400, "request_text is required")

    factory_base = (request.get("factory_url") or getattr(settings, "FACTORY_URL", "http://localhost:8080")).rstrip("/")
    callback_url = getattr(settings, "RUNTIME_CALLBACK_URL", "http://localhost:8081").rstrip("/") + "/api/runtime/receive_custom_ptem"

    # Collect oracle IDs this customer already has (so factory builds a relevant ptem)
    cpaths = get_customer_paths(customer_id)
    available_oracle_ids = []
    try:
        from customer.ptem_oracle_builder import build_oracle_values
        built = build_oracle_values(cpaths["DATA_DIR"])
        available_oracle_ids = list(built.keys())
    except Exception:
        pass

    try:
        import urllib.request, urllib.error
        _payload2 = json.dumps({
            "customer_id":          customer_id,
            "request_text":         request_text,
            "callback_url":         callback_url,
            "available_oracle_ids": available_oracle_ids,
        }).encode()
        _req2 = urllib.request.Request(
            factory_base + "/api/ptems/custom/start",
            data=_payload2,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(_req2, timeout=15) as _r2:
            if _r2.status not in (200, 201):
                raise RuntimeError(f"Factory returned HTTP {_r2.status}")
            data = json.loads(_r2.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Factory returned HTTP {e.code}: {e.read()[:200].decode(errors='replace')}")
    except Exception as e:
        raise HTTPException(502, f"Could not reach factory: {e}")

    # Store job reference locally so runtime can poll
    jobs_path = cpaths["DATA_DIR"] / "custom_ptem_jobs.json"
    jobs = {}
    if jobs_path.exists():
        try:
            jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
        except Exception:
            jobs = {}

    job_id = data.get("job_id")
    jobs[job_id] = {
        "job_id":       job_id,
        "request_text": request_text,
        "status":       "pending",
        "factory_url":  factory_base,
        "created_at":   datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    jobs_path.write_text(json.dumps(jobs, indent=2, ensure_ascii=False), encoding="utf-8")

    return JSONResponse({"job_id": job_id, "status": "pending", "message": "Request sent to factory — poll /api/customer/reports/request/{job_id} for status"})


@router.get("/api/customer/reports/request/{job_id}")
async def poll_custom_report_job(
    job_id:          str,
    x_session_token: Optional[str] = Header(default=None),
    token:           Optional[str] = Query(default=None),
):
    """
    GET /api/customer/reports/request/{job_id}
    Poll factory for status of a custom report request.
    When status=complete, also checks if custom ptem has been received locally.

    Returns: { job_id, status, request_text, ptem_ready, canonical_id }
    """
    import urllib.request, urllib.error
    from core.config import settings

    # Load local job record
    jobs_path = cpaths["DATA_DIR"] / "custom_ptem_jobs.json"
    jobs = {}
    if jobs_path.exists():
        try:
            jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    local_job = jobs.get(job_id, {})
    factory_base = local_job.get("factory_url") or getattr(settings, "FACTORY_URL", "http://localhost:8080").rstrip("/")

    # Poll factory for current status
    factory_status = {}
    try:
        import urllib.request, urllib.error
        with urllib.request.urlopen(factory_base + f"/api/ptems/custom/{job_id}", timeout=10) as _rg:
            if _rg.status == 200:
                factory_status = _json.loads(_rg.read().decode())
    except Exception:
        pass

    status = factory_status.get("status", local_job.get("status", "unknown"))

    # Check if ptem was already received locally
    custom_dir = cpaths["DATA_DIR"] / "custom_ptems"
    index_path = custom_dir / "_index.json"
    ptem_ready    = False
    canonical_id  = None

    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            for cid, entry in index.items():
                if entry.get("job_id") == job_id:
                    ptem_ready   = True
                    canonical_id = cid
                    break
        except Exception:
            pass

    # Update local record
    jobs[job_id] = {**local_job, "status": status}
    try:
        jobs_path.write_text(json.dumps(jobs, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

    return JSONResponse({
        "job_id":       job_id,
        "status":       status,
        "request_text": local_job.get("request_text", ""),
        "ptem_ready":   ptem_ready,
        "canonical_id": canonical_id,
        "error":        factory_status.get("error"),
        "created_at":   local_job.get("created_at"),
        "completed_at": factory_status.get("completed_at"),
    })


@router.get("/api/customer/reports/custom")
async def list_custom_reports(
    x_session_token: Optional[str] = Header(default=None),
    token:           Optional[str] = Query(default=None),
):
    """
    GET /api/customer/reports/custom
    Lists all custom ptems received for this customer.
    """
    customer_id = get_customer_id(_resolve_token(x_session_token, token))
    cpaths      = get_customer_paths(customer_id)
    custom_dir  = cpaths["DATA_DIR"] / "custom_ptems"
    index_path  = custom_dir / "_index.json"

    if not index_path.exists():
        return JSONResponse({"custom_reports": [], "total": 0})

    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        reports = list(index.values())
    except Exception:
        reports = []

    return JSONResponse({"custom_reports": reports, "total": len(reports)})


@router.post("/api/customer/reports/generate/custom")
async def generate_custom_report(
    request:         dict,
    x_session_token: Optional[str] = Header(default=None),
    token:           Optional[str] = Query(default=None),
):
    """
    POST /api/customer/reports/generate/custom
    Generate a report from a received custom ptem.
    Body: { "canonical_id": "custom_customer_xxx_abc_my_report" }
    """
    customer_id  = get_customer_id(_resolve_token(x_session_token, token))
    canonical_id = request.get("canonical_id")
    if not canonical_id:
        raise HTTPException(400, "canonical_id is required")

    cpaths     = get_customer_paths(customer_id)
    ptem_path  = cpaths["DATA_DIR"] / "custom_ptems" / f"{canonical_id}.json"

    if not ptem_path.exists():
        raise HTTPException(404, f"Custom ptem '{canonical_id}' not found — has it been received yet?")

    # Load custom ptem and inject into executor's catalog temporarily
    ptem = json.loads(ptem_path.read_text(encoding="utf-8"))

    from customer.ptem_executor import PtemExecutor
    executor = PtemExecutor(customer_id=customer_id)
    # Inject custom ptem into the executor catalog
    executor._catalog[canonical_id] = ptem
    result = executor.execute(canonical_id)

    if not result.get("success"):
        raise HTTPException(400, result.get("error", "Report generation failed"))

    return JSONResponse({
        "success":        True,
        "canonical_id":   result["canonical_id"],
        "title":          result["title"],
        "html":           result["html"],
        "html_path":      result["html_path"],
        "pdf_path":       result.get("pdf_path"),
        "narrative":      result["narrative"],
        "oracle_summary": result["oracle_summary"],
        "generated_at":   result["generated_at"],
    })


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD - Generic spec-driven engine (no hardcoded domain logic)
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/api/customer/contexts")
async def get_contexts(
    x_session_token: Optional[str] = Header(default=None),
    token:           Optional[str] = Query(default=None),
):
    """
    GET /api/customer/contexts
    Returns the list of available business contexts for this customer,
    plus which one is the default. Used by the context switcher dropdown.
    """
    customer_id = get_customer_id(_resolve_token(x_session_token, token))
    cpaths      = get_customer_paths(customer_id)
    ctx_path    = cpaths["DATA_DIR"] / "contexts.json"

    if not ctx_path.exists():
        return JSONResponse({"contexts": [], "default_context": None})

    try:
        store = json.loads(ctx_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(500, f"Failed to read contexts: {e}")

    contexts_list = [
        {
            "id":   cid,
            "name": ctx.get("name", cid),
            "tables": ctx.get("tables", []),
        }
        for cid, ctx in store.get("contexts", {}).items()
    ]
    return JSONResponse({
        "contexts":        contexts_list,
        "default_context": store.get("default_context"),
    })


@router.get("/api/customer/dashboard")
async def get_dashboard(
    context_id:      Optional[str] = Query(default=None),
    period:          str            = Query(default="30d"),
    from_date:       Optional[str] = Query(default=None, alias="from"),
    to_date:         Optional[str] = Query(default=None, alias="to"),
    x_session_token: Optional[str] = Header(default=None),
    token:           Optional[str] = Query(default=None),
):
    """
    GET /api/customer/dashboard?context_id=logistics&period=30d
    Generic spec-driven dashboard executor.
    Loads dashboard.json for the requested context, executes KPIs and chart
    aggregations against data_store.json, returns computed values.
    No domain-specific logic — fully driven by the spec.
    """
    from datetime import date, timedelta
    from collections import defaultdict

    customer_id = get_customer_id(_resolve_token(x_session_token, token))
    cpaths      = get_customer_paths(customer_id)
    base_dir    = cpaths["BASE"]
    data_dir    = cpaths["DATA_DIR"]

    # ── Resolve context ───────────────────────────────────────────────────────
    if not context_id:
        ctx_path = data_dir / "contexts.json"
        if ctx_path.exists():
            try:
                store      = json.loads(ctx_path.read_text(encoding="utf-8"))
                context_id = store.get("default_context")
            except Exception:
                pass

    if not context_id:
        return JSONResponse({"error": "no_context", "message": "No business context found. Please upload your data first."})

    # ── Load dashboard spec ───────────────────────────────────────────────────
    spec_path = base_dir / "context" / context_id / "dashboard.json"
    if not spec_path.exists():
        return JSONResponse({"error": "no_spec", "context_id": context_id,
                             "message": "Dashboard spec not yet generated for this context."})

    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(500, f"Failed to read dashboard spec: {e}")

    # ── Load data store ───────────────────────────────────────────────────────
    ds_path = data_dir / "data_store.json"
    if not ds_path.exists():
        return JSONResponse({"error": "no_data", "context_id": context_id})

    try:
        raw = json.loads(ds_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(500, f"Failed to read data store: {e}")

    # ── Date filter setup ─────────────────────────────────────────────────────
    today    = date.today()
    days_map = {"1d": 1, "7d": 7, "30d": 30, "90d": 90, "365d": 365}
    cutoff    = None
    cutoff_to = None

    if from_date and to_date:
        try:
            cutoff    = date.fromisoformat(from_date)
            cutoff_to = date.fromisoformat(to_date)
        except ValueError:
            pass
    else:
        days   = days_map.get(period)
        cutoff = (today - timedelta(days=days)) if days else None

    def _parse_date(val) -> Optional[date]:
        if not val:
            return None
        s = str(val).strip()
        # Strip timezone offset (+05:30, -08:00, Z) before parsing
        # ISO format with offset: "2026-06-17T16:05:11+05:30" → take first 10 chars
        if len(s) >= 10 and (s[10:11] == 'T' or s[4:5] == '-'):
            try:
                return date.fromisoformat(s[:10])
            except ValueError:
                pass
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%m/%d/%Y"):
            try:
                return datetime.strptime(s[:10], fmt).date()
            except ValueError:
                continue
        return None

    def _in_period(row: dict, date_col: Optional[str]) -> bool:
        if cutoff is None or not date_col:
            return True
        d = _parse_date(row.get(date_col, ""))
        if d is None:
            return True
        return d >= cutoff and (cutoff_to is None or d <= cutoff_to)

    def _fval(v) -> float:
        try:
            return float(str(v).replace(",", "") or 0)
        except (TypeError, ValueError):
            return 0.0

    # ── Table cache: fuzzy-match spec table names to data_store keys ──────────
    def _resolve_table(table_name: str) -> list:
        """Return rows for a table, fuzzy-matching against data_store keys."""
        if table_name in raw:
            return raw[table_name].get("rows", [])
        tl = table_name.lower()
        best_key, best_score = None, 0
        for key in raw:
            kl = key.lower()
            score = sum(1 for w in tl.split("_") if w and w in kl)
            if tl in kl or kl in tl:
                score += 3
            if score > best_score:
                best_key, best_score = key, score
        if best_key and best_score > 0:
            return raw[best_key].get("rows", [])
        return []

    # ── Primary date column from spec ─────────────────────────────────────────
    pdate = spec.get("primary_date_field") or {}
    primary_date_table  = pdate.get("table")
    primary_date_column = pdate.get("column")

    # ── Auto-heal: if spec's date column doesn't exist in actual rows, find one ─
    # This happens when data is uploaded from an external system (e.g. ODH Shopify)
    # whose column names differ from the atom canonical names used to build the spec.
    _DATE_SUFFIXES = ("_at", "_date", "_time", "created_at", "processed_at",
                      "updated_at", "order_date", "invoice_date", "event_date", "trip_date")

    def _find_real_date_col(rows: list) -> Optional[str]:
        """Find the best date column actually present in the rows."""
        if not rows:
            return None
        sample = rows[0]
        # Priority 1: exact known good names
        for preferred in ("created_at", "order_created_at", "event_date", "trip_date",
                          "invoice_date", "transaction_date", "processed_at", "updated_at"):
            if preferred in sample and sample.get(preferred):
                return preferred
        # Priority 2: any col ending in _at or _date that has a parseable value
        for col, val in sample.items():
            if any(col.endswith(sfx) for sfx in ("_at", "_date")) and val:
                if _parse_date(val):
                    return col
        return None

    if primary_date_table and primary_date_column:
        sample_rows = _resolve_table(primary_date_table)[:5]
        if sample_rows and not sample_rows[0].get(primary_date_column):
            # Spec's column missing from actual data — auto-detect
            real_col = _find_real_date_col(sample_rows)
            if real_col:
                log.info(
                    f"[dashboard] Auto-healing primary_date_column: "
                    f"{primary_date_column!r} → {real_col!r} for table {primary_date_table!r}"
                )
                primary_date_column = real_col


    # ── Execute KPIs ──────────────────────────────────────────────────────────
    computed_kpis = []
    for kpi in spec.get("kpis", []):
        table_name = kpi.get("table") or primary_date_table
        rows       = _resolve_table(table_name) if table_name else []
        date_col   = primary_date_column if table_name == primary_date_table else None
        rows       = [r for r in rows if _in_period(r, date_col)]

        kpi_type = kpi.get("type", "simple")
        try:
            if kpi_type == "simple":
                val = _exec_simple_agg(rows, kpi)
            elif kpi_type == "ratio":
                val = _exec_ratio(rows, kpi)
            else:
                val = None

            computed_kpis.append({
                "id":     kpi["id"],
                "title":  kpi["title"],
                "value":  val,
                "format": kpi.get("format", "number"),
            })
        except Exception as e:
            log.warning(f"[dashboard] KPI '{kpi['id']}' failed: {e}")
            computed_kpis.append({"id": kpi["id"], "title": kpi["title"], "value": None, "format": "number"})

    # ── Execute charts ────────────────────────────────────────────────────────
    computed_charts = []
    for chart in spec.get("charts", []):
        table_name = chart.get("table") or primary_date_table
        rows       = _resolve_table(table_name) if table_name else []
        date_col   = primary_date_column if table_name == primary_date_table else None
        rows       = [r for r in rows if _in_period(r, date_col)]

        try:
            chart_data = _exec_chart(rows, chart)
            computed_charts.append({
                "id":     chart["id"],
                "title":  chart["title"],
                "type":   chart["type"],
                "format": chart.get("format", "number"),
                "data":   chart_data,
            })
        except Exception as e:
            log.warning(f"[dashboard] Chart '{chart['id']}' failed: {e}")
            computed_charts.append({"id": chart["id"], "title": chart["title"], "type": chart["type"], "format": chart.get("format", "number"), "data": None})

    # -- Build per-table filtered row snapshot for client-side cross-filtering --
    # Strip internal _meta field; cap each table at 2000 rows to keep payload sane
    tables_in_spec = set()
    for kpi in spec.get("kpis", []):
        if kpi.get("table"): tables_in_spec.add(kpi["table"])
    for chart in spec.get("charts", []):
        if chart.get("table"): tables_in_spec.add(chart["table"])
    if primary_date_table: tables_in_spec.add(primary_date_table)

    raw_tables = {}
    for tname in tables_in_spec:
        trows = _resolve_table(tname)
        date_col = primary_date_column if tname == primary_date_table else None
        trows = [r for r in trows if _in_period(r, date_col)]
        # Strip _meta, cap at 2000 rows
        raw_tables[tname] = [{k: v for k, v in r.items() if k != "_meta"} for r in trows[:2000]]

    return JSONResponse({
        "context_id":   context_id,
        "context_name": spec.get("context_name", context_id),
        "period":       period,
        "computed_at":  datetime.utcnow().isoformat() + "Z",
        "kpis":         computed_kpis,
        "charts":       computed_charts,
        "raw_tables":   raw_tables,
        "spec":         {
            "kpis":    spec.get("kpis", []),
            "charts":  spec.get("charts", []),
            "filters": spec.get("filters", []),
            "primary_date_field": pdate,
        },
    })


# ── Generic aggregation helpers ───────────────────────────────────────────────

def _exec_simple_agg(rows: list, kpi: dict):
    """Execute a simple aggregation (count, sum, avg, count_where)."""
    col  = kpi.get("column")
    agg  = kpi.get("aggregation", "count")

    if agg == "count":
        return len(rows)

    if agg == "count_where":
        fcol = kpi.get("filter_column")
        fval = kpi.get("filter_value", "")
        if not fcol:
            return 0
        return sum(1 for r in rows if str(r.get(fcol, "")).lower() == str(fval).lower())

    if agg == "sum":
        return round(sum(_fval_safe(r.get(col)) for r in rows), 2)

    if agg == "avg":
        vals = [_fval_safe(r.get(col)) for r in rows if r.get(col) is not None]
        return round(sum(vals) / len(vals), 2) if vals else 0

    if agg == "max":
        vals = [_fval_safe(r.get(col)) for r in rows if r.get(col) is not None]
        return max(vals) if vals else 0

    if agg == "min":
        vals = [_fval_safe(r.get(col)) for r in rows if r.get(col) is not None]
        return min(vals) if vals else 0

    return None


def _exec_ratio(rows: list, kpi: dict):
    """Execute a numerator/denominator ratio KPI."""
    num_spec = kpi.get("numerator", {})
    den_spec = kpi.get("denominator", {})

    # Build temporary simple KPI specs
    num_kpi = {**num_spec, "type": "simple", "id": "_num", "title": "_num", "format": "number"}
    den_kpi = {**den_spec, "type": "simple", "id": "_den", "title": "_den", "format": "number"}

    num_val = _exec_simple_agg(rows, num_kpi)
    den_val = _exec_simple_agg(rows, den_kpi)

    if not den_val:
        return 0
    ratio = (num_val or 0) / den_val * 100
    return round(ratio, 1)


def _exec_chart(rows: list, chart: dict):
    """Execute a chart aggregation, returning labels + values."""
    from collections import defaultdict

    dim_col  = chart.get("dimension")
    dim_type = chart.get("dimension_type", "category")
    meas_col = chart.get("measure")
    agg      = chart.get("aggregation", "count")

    if not dim_col:
        return None

    if dim_type == "date":
        return _exec_time_chart(rows, dim_col, meas_col, agg)

    # Scatter: X = dim_col (numeric), Y = meas_col (numeric) — return paired points
    if dim_type == "numeric" and chart.get("type") == "scatter":
        points = []
        for r in rows:
            x = _fval_safe(r.get(dim_col))
            y = _fval_safe(r.get(meas_col)) if meas_col else 1
            if x is not None and y is not None:
                points.append({"x": x, "y": y})
        points = points[:500]  # cap for rendering
        return {"points": points, "labels": [], "values": []}

    # Histogram: bucket a numeric column into N bins
    if dim_type == "numeric" and chart.get("type") == "histogram":
        raw_vals = [_fval_safe(r.get(dim_col)) for r in rows if _fval_safe(r.get(dim_col)) is not None]
        if not raw_vals:
            return None
        mn, mx = min(raw_vals), max(raw_vals)
        n_bins = 10
        if mx == mn:
            return {"labels": [str(mn)], "values": [len(raw_vals)]}
        bin_size = (mx - mn) / n_bins
        bins = [0] * n_bins
        labels = []
        for i in range(n_bins):
            lo = mn + i * bin_size
            hi = lo + bin_size
            labels.append(f"{lo:.1f}–{hi:.1f}")
        for v in raw_vals:
            idx = min(int((v - mn) / bin_size), n_bins - 1)
            bins[idx] += 1
        return {"labels": labels, "values": bins}

    # Category / status / donut
    buckets: dict = defaultdict(list)
    for r in rows:
        key = str(r.get(dim_col) or "Unknown")
        val = _fval_safe(r.get(meas_col)) if meas_col else 1
        buckets[key].append(val)

    result = {}
    for k, vals in buckets.items():
        if agg == "count":
            result[k] = len(vals)
        elif agg == "count_where":
            fcol = chart.get("filter_column")
            fval = chart.get("filter_value", "")
            result[k] = sum(1 for r in rows if str(r.get(dim_col, "")) == k
                            and str(r.get(fcol, "")).lower() == str(fval).lower())
        elif agg == "sum":
            result[k] = round(sum(vals), 2)
        elif agg == "avg":
            result[k] = round(sum(vals) / len(vals), 2) if vals else 0
        else:
            result[k] = len(vals)

    # Sort by value descending, cap at 15 items
    sorted_items = sorted(result.items(), key=lambda x: -x[1])[:15]
    return {
        "labels": [i[0] for i in sorted_items],
        "values": [i[1] for i in sorted_items],
    }


def _exec_time_chart(rows: list, date_col: str, meas_col: Optional[str], agg: str):
    """Execute a time-series chart with auto-selected bucket granularity.

    Bucket rules based on data span:
      span <= 2 days   -> hourly   (Last 24 hrs)
      span <= 60 days  -> daily    (Last 30 days)
      span <= 400 days -> weekly   (Last 12 months / quarter)
      span > 400 days  -> monthly  (all time)
    """
    from collections import defaultdict
    from datetime import timedelta

    parsed = []
    for r in rows:
        raw_date = r.get(date_col, "")
        if not raw_date:
            continue
        s = str(raw_date).strip()
        d = None
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
            try:
                d = datetime.strptime(s, fmt)
                break
            except ValueError:
                # Also try trimming to 10 chars for datetime strings
                try:
                    d = datetime.strptime(s[:10], fmt)
                    break
                except ValueError:
                    continue
        if not d:
            continue
        val = _fval_safe(r.get(meas_col)) if meas_col else 1
        parsed.append((d, val))

    if not parsed:
        return None

    dates     = [p[0] for p in parsed]
    span_days = (max(dates) - min(dates)).days

    if span_days <= 2:
        bucket_fn = lambda d: d.strftime("%d %b %H:00")
    elif span_days <= 60:
        bucket_fn = lambda d: d.strftime("%d %b")
    elif span_days <= 400:
        # Start-of-week (Monday) label
        bucket_fn = lambda d: (d - timedelta(days=d.weekday())).strftime("%d %b")
    else:
        bucket_fn = lambda d: d.strftime("%b %Y")

    buckets: dict      = defaultdict(list)
    bucket_anchor: dict = {}
    for d, val in parsed:
        key = bucket_fn(d)
        buckets[key].append(val)
        if key not in bucket_anchor:
            bucket_anchor[key] = d

    ordered_keys = sorted(buckets.keys(), key=lambda k: bucket_anchor[k])

    values = []
    for mk in ordered_keys:
        vals = buckets[mk]
        if agg in ("count", "count_where"):
            values.append(len(vals))
        elif agg == "sum":
            values.append(round(sum(vals), 2))
        elif agg == "avg":
            values.append(round(sum(vals) / len(vals), 2) if vals else 0)
        else:
            values.append(len(vals))

    return {"labels": ordered_keys, "values": values}


def _fval_safe(v) -> float:
    try:
        return float(str(v).replace(",", "") or 0)
    except (TypeError, ValueError):
        return 0.0


# ── Dashboard Backfill ────────────────────────────────────────────────────────
# One-time endpoint: reads existing data_store.json for a customer,
# detects contexts and generates dashboard.json for each.
# Call once for customers who already had data before the new code was deployed.

@router.post("/api/customer/dashboard/backfill")
async def backfill_dashboard(
    x_session_token: Optional[str] = Header(default=None),
    token:           Optional[str] = Query(default=None),
):
    """
    POST /api/customer/dashboard/backfill
    Reads the customer's existing data_store.json, runs context detection
    and dashboard spec generation for every table found.
    Safe to call multiple times — overwrites existing contexts/specs.
    """
    customer_id = get_customer_id(_resolve_token(x_session_token, token))
    cpaths      = get_customer_paths(customer_id)
    data_dir    = cpaths["DATA_DIR"]
    ds_path     = data_dir / "data_store.json"

    if not ds_path.exists():
        raise HTTPException(404, "No data_store.json found — upload data first.")

    try:
        raw = json.loads(ds_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(500, f"Failed to read data_store: {e}")

    if not raw:
        return JSONResponse({"ok": False, "message": "data_store.json is empty."})

    # Wipe existing contexts.json so stale wrong assignments don't block re-detection
    ctx_path = data_dir / "contexts.json"
    if ctx_path.exists():
        ctx_path.write_text(json.dumps({"contexts": {}, "default_context": None}), encoding="utf-8")

    from customer.dashboard_spec_agent import generate_dashboard_spec, _extract_columns, _extract_enums
    from customer.context_detector import _load_contexts, _save_contexts, update_context_schema_cache
    from datetime import datetime, timezone

    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    # PREFIX MAP: table name prefix -> (context_id, human name)
    # Table naming law: {vertical}_{table}_{source}
    # Prefix is ground truth — no AI, no scoring in backfill.
    PREFIX_MAP = [
        ("supply_chain_",      "supply_chain",  "Supply Chain"),
        ("logistics_",         "logistics",     "Logistics Operations"),
        ("retail_shopify_",    "retail_shopify","Retail / Shopify"),
        ("retail_",            "retail_shopify","Retail / Shopify"),
        ("hr_",                "hr",            "HR Analytics"),
        ("human_resources_",   "hr",            "HR Analytics"),
        ("finance_",           "finance",       "Finance Analytics"),
        ("sales_",             "sales",         "Sales Analytics"),
    ]

    def _resolve_prefix(table_key: str):
        tk = table_key.lower()
        for prefix, ctx_id, ctx_name in PREFIX_MAP:
            if tk.startswith(prefix):
                return ctx_id, ctx_name
        # Unknown prefix: use first segment
        seg = tk.split("_")[0] if "_" in tk else tk
        return seg, seg.replace("_", " ").title() + " Analytics"

    # Build fresh ctx_store directly from prefix — no detect_context, no AI
    ctx_store = {"contexts": {}, "default_context": None}
    results = []
    affected_contexts: set = set()

    for table_key, table_data in raw.items():
        rows = table_data.get("rows", [])
        if not rows:
            continue

        ctx_id, ctx_name = _resolve_prefix(table_key)
        columns   = _extract_columns(rows)
        enums     = _extract_enums(rows, columns)
        sample_vals = {col: vals[:5] for col, vals in enums.items()}
        col_names = [c["name"] for c in columns]

        if ctx_id not in ctx_store["contexts"]:
            # Create context entry
            ctx_store["contexts"][ctx_id] = {
                "id":            ctx_id,
                "name":          ctx_name,
                "tables":        [],
                "column_names":  [],
                "sample_values": [],
                "detected_at":   _now_iso(),
                "updated_at":    _now_iso(),
            }
            if not ctx_store["default_context"]:
                ctx_store["default_context"] = ctx_id
            # Ensure context directory exists
            ctx_dir = data_dir.parent / "context" / ctx_id
            ctx_dir.mkdir(parents=True, exist_ok=True)
            action = "created"
        else:
            action = "joined"

        ctx = ctx_store["contexts"][ctx_id]
        if table_key not in ctx["tables"]:
            ctx["tables"].append(table_key)
        ctx["updated_at"] = _now_iso()

        # Update column/value cache inline
        existing_cols = set(ctx.get("column_names", []))
        existing_cols.update(col_names)
        ctx["column_names"] = list(existing_cols)[:200]

        existing_samples = set(ctx.get("sample_values", []))
        for vals in sample_vals.values():
            for v in (vals or [])[:5]:
                existing_samples.add(str(v)[:50])
        ctx["sample_values"] = list(existing_samples)[:500]

        affected_contexts.add(ctx_id)
        results.append({"table": table_key, "context": ctx_id, "action": action})
        log.info(f"[backfill] '{table_key}' -> '{ctx_id}' ({action})")

    # Persist the freshly-built ctx_store
    _save_contexts(data_dir, ctx_store)

    # Generate dashboard spec for every affected context
    from customer.context_detector import _load_contexts as _lctx
    ctx_store    = _lctx(data_dir)
    specs_built  = []

    for ctx_id in affected_contexts:
        ctx = ctx_store.get("contexts", {}).get(ctx_id, {})
        try:
            spec = generate_dashboard_spec(
                data_dir        = data_dir,
                context_id      = ctx_id,
                context_name    = ctx.get("name", ctx_id),
                table_names     = ctx.get("tables", []),
                data_store_path = ds_path,
            )
            specs_built.append({
                "context_id":   ctx_id,
                "context_name": ctx.get("name", ctx_id),
                "kpis":  len(spec.get("kpis", []))  if spec else 0,
                "charts": len(spec.get("charts", [])) if spec else 0,
            })
            log.info(f"[backfill] Dashboard spec built for '{ctx_id}'")
        except Exception as e:
            log.warning(f"[backfill] spec generation failed for '{ctx_id}': {e}")
            specs_built.append({"context_id": ctx_id, "error": str(e)})

    return JSONResponse({
        "ok":       True,
        "tables_processed": len(results),
        "contexts_detected": list(affected_contexts),
        "table_results": results,
        "specs_built": specs_built,
    })


# ── Alert Thresholds ──────────────────────────────────────────────────────────

_DEFAULT_THRESHOLDS = {
    "delivery_rate_min":        {"value": 80,  "label": "Min delivery rate (%)",       "unit": "%",  "description": "Alert when delivery rate falls below this %"},
    "late_shipments_max":       {"value": 0,   "label": "Max late shipments",           "unit": "",   "description": "Alert when late shipments exceed this count (0 = any)"},
    "in_transit_notify":        {"value": 1,   "label": "In-transit notify threshold",  "unit": "",   "description": "Alert when in-transit shipments reach this count"},
    "pending_shipments_notify": {"value": 1,   "label": "Pending shipments threshold",  "unit": "",   "description": "Alert when pending shipments reach this count"},
    "overdue_invoices_max":     {"value": 0,   "label": "Max overdue invoices",         "unit": "",   "description": "Alert when overdue invoices exceed this count (0 = any)"},
    "pending_invoices_notify":  {"value": 1,   "label": "Pending invoices threshold",   "unit": "",   "description": "Alert when pending invoices reach this count"},
}

def _load_thresholds(data_dir: Path) -> dict:
    path = data_dir / "alert_thresholds.json"
    if path.exists():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            # Merge with defaults so new keys always appear
            result = {}
            for k, meta in _DEFAULT_THRESHOLDS.items():
                result[k] = dict(meta)
                if k in stored:
                    result[k]["value"] = stored[k].get("value", meta["value"])
            return result
        except Exception:
            pass
    return {k: dict(v) for k, v in _DEFAULT_THRESHOLDS.items()}

def _save_thresholds(data_dir: Path, thresholds: dict):
    path = data_dir / "alert_thresholds.json"
    path.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")


@router.get("/api/customer/alert-thresholds")
async def get_alert_thresholds(
    x_session_token: Optional[str] = Header(default=None),
    token:           Optional[str] = Query(default=None),
):
    customer_id = get_customer_id(_resolve_token(x_session_token, token))
    cpaths      = get_customer_paths(customer_id)
    thresholds  = _load_thresholds(cpaths["DATA_DIR"])
    return JSONResponse({"thresholds": thresholds})


@router.post("/api/customer/alert-thresholds")
async def save_alert_thresholds(
    request:         dict,
    x_session_token: Optional[str] = Header(default=None),
    token:           Optional[str] = Query(default=None),
):
    customer_id = get_customer_id(_resolve_token(x_session_token, token))
    cpaths      = get_customer_paths(customer_id)
    thresholds  = _load_thresholds(cpaths["DATA_DIR"])

    key   = request.get("key")
    value = request.get("value")

    if key not in _DEFAULT_THRESHOLDS:
        raise HTTPException(400, f"Unknown threshold key: {key}")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise HTTPException(400, "Value must be a number")

    thresholds[key]["value"] = value
    _save_thresholds(cpaths["DATA_DIR"], thresholds)
    return JSONResponse({"ok": True, "key": key, "value": value, "thresholds": thresholds})


@router.get("/api/customer/alerts")
async def get_alerts(
    period:          str                  = Query(default="30d"),
    context_id:      Optional[str]        = Query(default=None),
    x_session_token: Optional[str]        = Header(default=None),
    token:           Optional[str]        = Query(default=None),
):
    """
    GET /api/customer/alerts
    Evaluates alert thresholds against live data, filtered by the active vertical (context_id).
    Returns alerts ordered by severity (error > warning > info), each with threshold_key for editing.
    """
    customer_id = get_customer_id(_resolve_token(x_session_token, token))
    cpaths      = get_customer_paths(customer_id)
    thresholds  = _load_thresholds(cpaths["DATA_DIR"])

    ds_path = cpaths["DATA_DIR"] / "data_store.json"
    if not ds_path.exists():
        return JSONResponse({"alerts": [], "evaluated_at": datetime.utcnow().isoformat() + "Z"})
    try:
        data_store = json.loads(ds_path.read_text(encoding="utf-8"))
    except Exception:
        return JSONResponse({"alerts": [], "evaluated_at": datetime.utcnow().isoformat() + "Z"})

    # Determine active vertical from context_id (e.g. "logistics", "supply_chain", "retail_shopify")
    vertical = (context_id or "").lower().strip()

    def _get_rows(table_name: str) -> list:
        return data_store.get(table_name, {}).get("rows", [])

    def _try_tables(candidates: list) -> tuple:
        """Return (rows, tname) for first candidate that has rows."""
        for tname in candidates:
            rows = _get_rows(tname)
            if rows:
                return rows, tname
        return [], ""

    alerts = []

    # ── Helper: resolve table candidates by vertical ──────────────────────────
    # Logistics vertical
    def _shipment_tables():
        if vertical == "logistics":
            return ["logistics_shipments_ERP_record"]
        if vertical == "supply_chain":
            return ["supply_chain_shipments_WMS_record"]
        return ["logistics_shipments_ERP_record", "supply_chain_shipments_WMS_record"]

    def _invoice_tables():
        if vertical in ("supply_chain", ""):
            return ["supply_chain_invoices_ERP_record"]
        return []

    def _incident_tables():
        if vertical == "logistics":
            return ["logistics_incidents_ERP_record"]
        return []

    def _vehicle_tables():
        if vertical == "logistics":
            return ["logistics_vehicle_trips_WMS_event"]
        return []

    # ── 1. Delivery rate ──────────────────────────────────────────────────────
    min_dr = thresholds.get("delivery_rate_min", {}).get("value", 70)
    rows, tname = _try_tables(_shipment_tables())
    if rows:
        flag_col = next((c for c in rows[0] if "on_time" in c.lower() or "delivered" in c.lower()), None)
        status_col = next((c for c in rows[0] if "status" in c.lower()), None)
        if flag_col:
            delivered = sum(1 for r in rows if str(r.get(flag_col, "")).lower() in ("1","true","yes","delivered"))
            rate      = round(delivered / len(rows) * 100, 1)
        elif status_col:
            delivered = sum(1 for r in rows if "delivered" in str(r.get(status_col,"")).lower())
            rate      = round(delivered / len(rows) * 100, 1)
        else:
            rate = 100
        if rate < min_dr:
            alerts.append({
                "id": "delivery_rate_low", "severity": "error",
                "title": "Delivery Rate Below Threshold",
                "message": f"Current delivery rate is {rate}%, below the minimum of {min_dr}%.",
                "metric": f"{rate}%", "threshold": f"{min_dr}%",
                "threshold_key": "delivery_rate_min",
                "threshold_value": min_dr, "threshold_unit": "%",
                "threshold_label": "Min delivery rate (%)",
                "vertical": vertical or "all", "table": tname,
            })

    # ── 2. Late shipments ─────────────────────────────────────────────────────
    max_late = thresholds.get("late_shipments_max", {}).get("value", 0)
    rows, tname = _try_tables(_shipment_tables())
    if rows:
        late_col   = next((c for c in rows[0] if "is_late" in c.lower()), None)
        status_col = next((c for c in rows[0] if "status" in c.lower()), None)
        if late_col:
            late_count = sum(1 for r in rows if str(r.get(late_col,"")).lower() in ("1","true","yes"))
        elif status_col:
            late_count = sum(1 for r in rows if "late" in str(r.get(status_col,"")).lower())
        else:
            late_count = 0
        if late_count > max_late:
            alerts.append({
                "id": "late_shipments_exceeded",
                "severity": "error" if late_count > max_late * 2 + 2 else "warning",
                "title": "Late Shipments Exceed Limit",
                "message": f"{late_count} late shipment{'s' if late_count!=1 else ''} detected (limit: {max_late}).",
                "metric": str(late_count), "threshold": str(max_late),
                "threshold_key": "late_shipments_max",
                "threshold_value": max_late, "threshold_unit": "",
                "threshold_label": "Max late shipments",
                "vertical": vertical or "all", "table": tname,
            })

    # ── 3. In-transit notify ──────────────────────────────────────────────────
    notify_transit = thresholds.get("in_transit_notify", {}).get("value", 1)
    rows, tname = _try_tables(_shipment_tables())
    if rows:
        status_col = next((c for c in rows[0] if "status" in c.lower()), None)
        if status_col:
            in_transit = sum(1 for r in rows if "transit" in str(r.get(status_col,"")).lower())
            if in_transit >= notify_transit:
                alerts.append({
                    "id": "in_transit_count", "severity": "info",
                    "title": "Shipments In Transit",
                    "message": f"{in_transit} shipment{'s' if in_transit!=1 else ''} currently in transit.",
                    "metric": str(in_transit), "threshold": str(notify_transit),
                    "threshold_key": "in_transit_notify",
                    "threshold_value": notify_transit, "threshold_unit": "",
                    "threshold_label": "In-transit notify threshold",
                    "vertical": vertical or "all", "table": tname,
                })

    # ── 4. Pending shipments ──────────────────────────────────────────────────
    notify_pending = thresholds.get("pending_shipments_notify", {}).get("value", 1)
    rows, tname = _try_tables(_shipment_tables())
    if rows:
        status_col = next((c for c in rows[0] if "status" in c.lower()), None)
        if status_col:
            pending = sum(1 for r in rows if "pending" in str(r.get(status_col,"")).lower())
            if pending >= notify_pending:
                alerts.append({
                    "id": "pending_shipments", "severity": "warning",
                    "title": "Pending Shipments",
                    "message": f"{pending} shipment{'s' if pending!=1 else ''} pending dispatch.",
                    "metric": str(pending), "threshold": str(notify_pending),
                    "threshold_key": "pending_shipments_notify",
                    "threshold_value": notify_pending, "threshold_unit": "",
                    "threshold_label": "Pending shipments threshold",
                    "vertical": vertical or "all", "table": tname,
                })

    # ── 5. Incidents (logistics only) ─────────────────────────────────────────
    if vertical in ("logistics", ""):
        rows, tname = _try_tables(_incident_tables())
        if rows:
            open_col = next((c for c in rows[0] if "status" in c.lower()), None)
            open_cnt = sum(1 for r in rows if "open" in str(r.get(open_col or "", "")).lower()) if open_col else len(rows)
            if open_cnt > 0:
                alerts.append({
                    "id": "open_incidents", "severity": "warning",
                    "title": "Open Logistics Incidents",
                    "message": f"{open_cnt} open incident{'s' if open_cnt!=1 else ''} require attention.",
                    "metric": str(open_cnt), "threshold": "0",
                    "threshold_key": None,
                    "threshold_value": None, "threshold_unit": "",
                    "threshold_label": None,
                    "vertical": "logistics", "table": tname,
                })

    # ── 6. Overdue invoices (supply_chain) ────────────────────────────────────
    max_overdue = thresholds.get("overdue_invoices_max", {}).get("value", 0)
    rows, tname = _try_tables(_invoice_tables())
    if rows:
        status_col = next((c for c in rows[0] if "status" in c.lower()), None)
        if status_col:
            overdue = sum(1 for r in rows if
                         "overdue" in str(r.get(status_col,"")).lower() or
                         "disputed" in str(r.get(status_col,"")).lower())
            if overdue > max_overdue:
                alerts.append({
                    "id": "overdue_invoices",
                    "severity": "error" if overdue > max_overdue + 3 else "warning",
                    "title": "Overdue Invoices",
                    "message": f"{overdue} invoice{'s' if overdue!=1 else ''} overdue or disputed (limit: {max_overdue}).",
                    "metric": str(overdue), "threshold": str(max_overdue),
                    "threshold_key": "overdue_invoices_max",
                    "threshold_value": max_overdue, "threshold_unit": "",
                    "threshold_label": "Max overdue invoices",
                    "vertical": "supply_chain", "table": tname,
                })

    # ── 7. Pending invoices ───────────────────────────────────────────────────
    notify_inv = thresholds.get("pending_invoices_notify", {}).get("value", 1)
    rows, tname = _try_tables(_invoice_tables())
    if rows:
        status_col = next((c for c in rows[0] if "status" in c.lower()), None)
        if status_col:
            pending_inv = sum(1 for r in rows if "pending" in str(r.get(status_col,"")).lower())
            if pending_inv >= notify_inv:
                alerts.append({
                    "id": "pending_invoices", "severity": "warning",
                    "title": "Pending Invoices",
                    "message": f"{pending_inv} invoice{'s' if pending_inv!=1 else ''} awaiting approval.",
                    "metric": str(pending_inv), "threshold": str(notify_inv),
                    "threshold_key": "pending_invoices_notify",
                    "threshold_value": notify_inv, "threshold_unit": "",
                    "threshold_label": "Pending invoices threshold",
                    "vertical": "supply_chain", "table": tname,
                })

    # Sort: error > warning > info
    sev_order = {"error": 0, "warning": 1, "info": 2}
    alerts.sort(key=lambda a: sev_order.get(a["severity"], 3))

    return JSONResponse({
        "alerts":       alerts,
        "count":        len(alerts),
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
    })

