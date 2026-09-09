"""
routers/agents_router.py
========================
All API endpoints for the GPL agents platform.

PRE-FLIGHT STRATEGY (two calls before the main agent run):

  Call 1 — Estimate:
    "How many atoms does {vertical} need?"
    → atom_count (int)
    → strategy: single_low / single_high / batch

  Call 2 — Enumerate:
    "List every entity name for {vertical}"
    → ["orders", "invoices", "shipments", ...]
    This explicit list is passed to the agent as a checklist.
    Claude cannot declare done until every item on the list is created.

  For batch strategy, Call 2 is split by category:
    core / operational / reference / financial — each gets its own list.

  Customer mode skips both pre-flight calls (schema is already scoped).
"""

import json
import logging
import re
from typing import List, Optional

from anthropic import Anthropic
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from core.config import settings

log    = logging.getLogger(__name__)
router = APIRouter()
def _get_client():
    """Build a fresh client per call so a runtime API-key update takes effect immediately."""
    return Anthropic(api_key=settings.ANTHROPIC_API_KEY)

_VERTICALS = [
    "supply_chain", "finance", "sales", "hr", "logistics",
    "warehouse", "procurement", "revenue", "marketing", "customer",
    "legal", "compliance", "operations", "production", "quality",
    "risk", "treasury", "maintenance", "project_management", "property",
    "academic", "actuarial", "advertising", "agriculture", "call_center",
    "claims", "distribution", "ediscovery", "engineering", "enrollment",
    "clinical", "construction", "construction_ops", "corporate_tx",
    "cybersecurity", "data_governance", "esg", "fleet", "food_beverage",
    "fulfillment", "generation", "government", "grid", "growth",
    "hospitality", "ip", "it_ops", "leasing", "legal_ops", "lending",
    "litigation", "media_content", "merchandising", "mining", "nonprofit",
    "personal_finance", "personal_health", "pharma_commercial", "pharma_rd",
    "product", "real_estate", "regulatory", "restaurant", "revenue_cycle",
    "student_success", "tax_preparation", "telecom_network",
    "telecom_subscriber", "trading", "underwriting",
]

_BATCH_SEQUENCE    = ["core", "operational", "reference", "financial"]
_THRESHOLD_LOW     = 20   # < 20  → single_low
_THRESHOLD_HIGH    = 50   # 20-50 → single_high  |  > 50 → batch

_BATCH_DESCRIPTIONS = {
    "core":        "primary fact tables that record main business transactions (orders, invoices, payments, shipments)",
    "operational": "day-to-day operations and status tracking (inventory, schedules, logs, quality checks, production runs)",
    "reference":   "master data and lookup tables (customers, products, suppliers, employees, locations, chart of accounts)",
    "financial":   "money, budgets, forecasts, and accounting (budgets, forecasts, ledger entries, bank transactions, cost allocations)",
}


# ── Pydantic models ────────────────────────────────────────────────────────────

class CustomerColumn(BaseModel):
    name:   str
    type:   str
    values: Optional[List[str]] = None

class CustomerTable(BaseModel):
    name:    str
    columns: List[CustomerColumn]

class CustomerRelationship(BaseModel):
    from_table:  str
    from_column: str
    to_table:    str
    to_column:   str

class CustomerSchema(BaseModel):
    tables:        List[CustomerTable]
    relationships: Optional[List[CustomerRelationship]] = []

class MockDataRequest(BaseModel):
    vertical:     str
    mode:         str = "GENERIC"
    canonical_id: str = ""   # if set, regenerate only this one atom


class DeclareRelationshipsRequest(BaseModel):
    vertical: str


class VerticalSchemaRequest(BaseModel):
    vertical:        str
    mode:            str = "GENERIC"
    customer_schema: Optional[CustomerSchema] = None


# ── Grain key auto-fix ────────────────────────────────────────────────────────

def _unused_placeholder():
    pass


# ── Pre-flight helpers ─────────────────────────────────────────────────────────

def _preflight_estimate(vertical: str) -> dict:
    """
    Call 1: ask Claude how many atoms this vertical needs.
    Returns {"atom_count": int, "reasoning": str}
    Falls back to 25 on any error.
    """
    json_example = '{"atom_count": <integer>, "reasoning": "<one sentence>"}'
    prompt = (
        f"How many atom definitions (data table definitions) does a comprehensive "
        f"'{vertical}' business vertical typically need in a GPL data model? "
        f"Consider all main entity categories: core transactions, operational tables, "
        f"reference/dimension tables, and financial tables. "
        f"Reply with ONLY a JSON object, no other text, no markdown: {json_example}"
    )
    try:
        resp = _get_client().messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip().replace("```json","").replace("```","").strip()
        data = json.loads(text)
        count = max(3, min(int(data.get("atom_count", 25)), 200))
        return {"atom_count": count, "reasoning": data.get("reasoning", "")}
    except Exception as e:
        return {"atom_count": 25, "reasoning": f"Fallback — pre-flight error: {e}"}


def _preflight_enumerate(vertical: str, category: str = "all") -> list:
    """
    Call 2: ask Claude to list every specific entity for this vertical.
    For batch mode, category filters to one group (core/operational/reference/financial).
    Returns a list of entity name strings e.g. ["orders", "invoices", "shipments"]
    Falls back to empty list on any error (agent will still run, just without checklist).
    """
    if category == "all":
        scope = (
            f"List every entity (data table) that a comprehensive '{vertical}' "
            f"business vertical needs in a GPL data model. Include all categories: "
            f"core transactions, operational tables, reference/dimension tables, "
            f"and financial tables."
        )
    else:
        desc = _BATCH_DESCRIPTIONS.get(category, category)
        scope = (
            f"List only the '{category}' entities for a '{vertical}' business vertical. "
            f"These are: {desc}. "
            f"Do not include entities from other categories."
        )

    prompt = (
        f"{scope} "
        f"Reply with ONLY a JSON array of snake_case entity names, no other text, no markdown: "
        f'["entity_one", "entity_two", ...]'
    )
    try:
        resp = _get_client().messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip().replace("```json","").replace("```","").strip()
        entities = json.loads(text)
        if isinstance(entities, list):
            # Clean and deduplicate
            return list(dict.fromkeys(
                str(e).strip().lower().replace(" ", "_")
                for e in entities if e
            ))
        return []
    except Exception as e:
        return []  # agent runs without checklist rather than failing


def _decide_strategy(atom_count: int) -> str:
    if atom_count < _THRESHOLD_LOW:
        return "single_low"
    elif atom_count <= _THRESHOLD_HIGH:
        return "single_high"
    else:
        return "batch"


def _build_schema_dict(req: VerticalSchemaRequest) -> Optional[dict]:
    if not req.customer_schema:
        return None
    return {
        "tables": [
            {"name": t.name, "columns": [
                {"name": c.name, "type": c.type,
                 **({"values": c.values} if c.values else {})}
                for c in t.columns
            ]}
            for t in req.customer_schema.tables
        ],
        "relationships": [
            {"from_table": r.from_table, "from_column": r.from_column,
             "to_table":   r.to_table,   "to_column":   r.to_column}
            for r in (req.customer_schema.relationships or [])
        ],
    }


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/api/health")
async def health():
    return JSONResponse({"status": "ok"})


# ── Column canonicalization ───────────────────────────────────────────────────

class CanonicalizeRequest(BaseModel):
    columns:           List[str]
    vertical:          str
    atom_canonical_id: Optional[str] = None


@router.post("/api/factory/canonicalize")
async def canonicalize_columns(req: CanonicalizeRequest):
    """
    Map raw column headers to factory canonical field names.

    Uses the vertical's dialect file (column_hints vocabulary section) as the
    primary lookup, with the matched atom's field list as a secondary layer.
    Plain snake_case is the fallback — this endpoint never returns an error
    for individual columns.

    Request body:
      { "columns": ["Carrier Name", "Rate/Km", ...],
        "vertical": "logistics",
        "atom_canonical_id": "logistics_carriers_MANUAL_dimension" }

    Response:
      { "canonical_map": {"Carrier Name": "carrier_name", "Rate/Km": "contract_rate_per_km", ...},
        "mapped_count": 2,
        "total_count": 3 }
    """
    from customer.canonicalizer import canonicalize_headers
    try:
        canon_map = canonicalize_headers(
            columns           = req.columns,
            vertical          = req.vertical,
            atom_canonical_id = req.atom_canonical_id,
        )
        mapped = sum(
            1 for orig, canon in canon_map.items()
            if canon != re.sub(r"[^a-z0-9_]", "_", orig.lower().strip()).strip("_")
        )
        return JSONResponse({
            "canonical_map": canon_map,
            "mapped_count":  mapped,
            "total_count":   len(req.columns),
            "vertical":      req.vertical,
            "atom":          req.atom_canonical_id,
        })
    except Exception as exc:
        log.error(f"[canonicalize] Unexpected error: {exc}")
        raise HTTPException(500, f"Canonicalization failed: {exc}")


# ── Vertical detection ────────────────────────────────────────────────────────

class DetectVerticalRequest(BaseModel):
    table_name:         str
    columns:            List[str]
    candidate_domains:  List[str]   # tied domains, or all known domains for low/zero signal


@router.post("/api/factory/detect-vertical")
async def detect_vertical_endpoint(req: DetectVerticalRequest):
    """
    LLM-powered vertical detection for cases the keyword dictionary couldn't resolve:
      - Score = 0  (no keyword signal at all)
      - Score < 2  (signal too weak to trust)
      - Tie        (two or more domains share the top keyword score)

    Receives only table name + normalised column names — no customer row data.
    candidate_domains is pre-scoped by the caller:
      - On a tie:          only the tied domains  (tight decision space)
      - On low/no signal:  all known domains       (open decision)

    Request:
      { "table_name": "fleet_ops",
        "columns": ["carrier", "ship_date", "tracking_no"],
        "candidate_domains": ["logistics", "supply_chain"] }

    Response:
      { "vertical": "logistics" }
    """
    domains_list = ", ".join(f'"{d}"' for d in req.candidate_domains)
    prompt = f"""You are a business data classifier. A customer has uploaded a table.
Decide which business domain it belongs to.

Table name : {req.table_name}
Columns    : {req.columns}

Choose exactly one domain from this list: [{domains_list}]

Rules:
- Reply with ONLY the domain string, nothing else — no explanation, no punctuation.
- If genuinely ambiguous, pick the closest match. Never reply with "unknown".
- Valid replies: {domains_list}"""

    try:
        client = _get_client()
        resp   = client.messages.create(
            model      = settings.ANTHROPIC_MODEL,
            max_tokens = 20,
            messages   = [{"role": "user", "content": prompt}],
        )
        raw    = resp.content[0].text.strip().strip('"').strip("'").lower()
        chosen = raw if raw in req.candidate_domains else None

        if chosen is None:
            log.warning(
                f"[detect-vertical] LLM returned unexpected value {raw!r} "
                f"(candidates={req.candidate_domains}) — defaulting to first candidate"
            )
            chosen = req.candidate_domains[0]

        log.info(
            f"[detect-vertical] '{req.table_name}' → '{chosen}' "
            f"(candidates={req.candidate_domains})"
        )
        return JSONResponse({"vertical": chosen})

    except Exception as exc:
        log.error(f"[detect-vertical] LLM call failed: {exc}")
        raise HTTPException(500, f"Vertical detection failed: {exc}")


@router.get("/api/logs/stream")
async def stream_logs():
    """
    Server-Sent Events endpoint — streams live log lines to the browser terminal.
    Each event: data: level_class|logger_name|message
    """
    from core.log_stream import event_stream
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        }
    )


# ── Job polling ───────────────────────────────────────────────────────────────

@router.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Poll a long-running job for status + result."""
    from core.job_manager import get_job
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    return JSONResponse(job)


@router.get("/api/agents/vertical-schema/verticals")
async def list_verticals():
    return JSONResponse({"verticals": _VERTICALS, "count": len(_VERTICALS)})


@router.post("/api/agents/vertical-schema")
async def run_vertical_schema(req: VerticalSchemaRequest):
    from core.job_manager import create_job, run_job
    from agents.vertical_schema_agent import run as _run_schema
    job_id = create_job(f"Vertical Schema — {req.vertical}")
    def _job():
        result = _run_schema(
            vertical=req.vertical,
            mode=req.mode,
            customer_schema=req.customer_schema.dict() if req.customer_schema else None
        )
        return result
    run_job(job_id, _job)
    return JSONResponse({"job_id": job_id, "status": "pending"})


@router.post("/api/agents/declare-relationships")
async def run_declare_relationships(req: DeclareRelationshipsRequest):
    """
    Run RelationshipDeclarationAgent — declares FK relationships between
    atoms that already exist for req.vertical. Does not create or modify atoms.
    Separate step from the Vertical Schema Agent (atoms only) and from
    Relationship Discovery Phase 1 (deterministic validation/gap-filling).
    """
    from core.job_manager import create_job, run_job
    from agents.relationship_declaration_agent import run as _run_declare
    job_id = create_job(f"Relationship Declaration — {req.vertical}")
    def _job():
        return _run_declare(vertical=req.vertical)
    run_job(job_id, _job)
    return JSONResponse({"job_id": job_id, "status": "pending"})


@router.post("/api/agents/vertical-schema-sync")
async def run_vertical_schema_sync(req: VerticalSchemaRequest):
    """Sync version kept for internal use."""
    """
    Run VerticalSchemaAgent with two-stage pre-flight and automatic batching.

    GENERIC mode:
      Pre-flight 1 → estimate atom count → decide strategy
      Pre-flight 2 → enumerate specific entities → pass as checklist to agent
      Agent runs with explicit list — cannot stop early

    CUSTOMER mode:
      No pre-flight — schema already defines the scope
      Single run always
    """
    from agents.vertical_schema_agent import run

    if req.mode not in ("GENERIC", "CUSTOMER"):
        raise HTTPException(400, "mode must be GENERIC or CUSTOMER")
    if req.mode == "CUSTOMER" and not req.customer_schema:
        raise HTTPException(400, "customer_schema required for CUSTOMER mode")

    schema_dict = _build_schema_dict(req)

    # ── Customer mode ──────────────────────────────────────────────────────────
    if req.mode == "CUSTOMER":
        try:
            result = run(
                vertical=req.vertical,
                mode="CUSTOMER",
                customer_schema=schema_dict,
                entity_checklist=[],
                batch_focus="all",
            )
            result["batches_run"]   = 1
            result["atom_estimate"] = None
            result["strategy"]      = "single_customer"
            return JSONResponse(result)
        except Exception as e:
            raise HTTPException(500, str(e))

    # ── GENERIC mode — pre-flight ──────────────────────────────────────────────
    estimate   = _preflight_estimate(req.vertical)
    atom_count = estimate["atom_count"]
    strategy   = _decide_strategy(atom_count)

    # ── Single run (small < 20) ────────────────────────────────────────────────
    if strategy == "single_low":
        checklist = _preflight_enumerate(req.vertical, "all")
        try:
            result = run(
                vertical=req.vertical,
                mode="GENERIC",
                entity_checklist=checklist,
                batch_focus="all",
                max_iterations=120,
            )
            result["batches_run"]   = 1
            result["atom_estimate"] = atom_count
            result["strategy"]      = "single_low"
            result["reasoning"]     = estimate["reasoning"]
            result["checklist"]     = checklist
            return JSONResponse(result)
        except Exception as e:
            raise HTTPException(500, str(e))

    # ── Single run (medium 20-50) ──────────────────────────────────────────────
    if strategy == "single_high":
        checklist = _preflight_enumerate(req.vertical, "all")
        try:
            result = run(
                vertical=req.vertical,
                mode="GENERIC",
                entity_checklist=checklist,
                batch_focus="all",
                max_iterations=300,
            )
            result["batches_run"]   = 1
            result["atom_estimate"] = atom_count
            result["strategy"]      = "single_high"
            result["reasoning"]     = estimate["reasoning"]
            result["checklist"]     = checklist
            return JSONResponse(result)
        except Exception as e:
            raise HTTPException(500, str(e))

    # ── Auto-batch (large > 50) ────────────────────────────────────────────────
    combined = {
        "status":                "success",
        "mode":                  req.mode,
        "vertical":              req.vertical,
        "atom_estimate":         atom_count,
        "strategy":              "auto_batch",
        "reasoning":             estimate["reasoning"],
        "batches_run":           0,
        "atoms_created":         [],
        "atoms_updated":         [],
        "iterations":            0,
        "checklist":             {},
        "summary":               "",
    }

    for batch in _BATCH_SEQUENCE:
        checklist = _preflight_enumerate(req.vertical, batch)
        combined["checklist"][batch] = checklist
        try:
            result = run(
                vertical=req.vertical,
                mode="GENERIC",
                entity_checklist=checklist,
                batch_focus=batch,
                max_iterations=400,
            )
            combined["batches_run"]           += 1
            combined["atoms_created"]         += result.get("atoms_created", [])
            combined["atoms_updated"]         += result.get("atoms_updated", [])
            combined["iterations"]            += result.get("iterations", 0)
            if result.get("summary"):
                combined["summary"] += f"[{batch}] {result['summary']} "
        except Exception as e:
            combined["summary"] += f"[{batch}] ERROR: {str(e)} "
            continue

    if not combined["atoms_created"] and not combined["atoms_updated"]:
        combined["status"] = "no_changes"

    # Auto-run relationship declaration, then relationship discovery,
    # after schema agent completes (atoms-only agent no longer does this itself)
    try:
        from agents.relationship_declaration_agent import run as _declare
        declaration = _declare(vertical=req.vertical)
        combined["relationship_declaration"] = declaration
    except Exception as e:
        combined["relationship_declaration"] = {"status": "error", "error": str(e)}

    try:
        from agents.relationship_discovery_agent import run as _discover
        discovery = _discover(vertical=req.vertical)
        combined["relationship_discovery"] = discovery
    except Exception as e:
        combined["relationship_discovery"] = {"status": "error", "error": str(e)}

    return JSONResponse(combined)


@router.post("/api/agents/discover-enums")
async def run_discover_enums(vertical: str = None):
    """
    Manually trigger enum discovery for all verticals or a specific one.
    Reads mock_data CSVs via DuckDB and writes field_values.json.
    MockDataAgent calls this automatically — this endpoint is for manual re-runs.
    """
    from services.discover_enums import discover_enums
    try:
        result = discover_enums(vertical=vertical)
        return JSONResponse({
            "status":     "success",
            "enums_found": len(result),
            "vertical":   vertical or "all",
        })
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Relationship discovery ────────────────────────────────────────────────────

@router.post("/api/agents/discover-relationships/{vertical}")
async def discover_relationships(vertical: str):
    """
    Run the Relationship Discovery Agent (Phase 1 — no CSV needed).
    5 algorithms: column name matching, functional dependency,
    hidden dimension extraction, grain verification, temporal patterns.
    Pure Python — zero AI calls, runs instantly.
    """
    from agents.relationship_discovery_agent import run as _run
    result = _run(vertical=vertical)
    return JSONResponse(result)


@router.post("/api/agents/fix-relationship-errors/{vertical}")
async def fix_relationship_errors(vertical: str):
    """
    Remove structurally broken relationships for a vertical.
    - ATOM_NOT_FOUND: removes rels pointing to atoms from other verticals
    - FIELD_NOT_FOUND: removes rels with non-existent fields
    - DUPLICATE: removes duplicate rels
    Pure Python — no AI calls.
    """
    import json
    from core.paths import ATOMS_PATH, PROJECT_ROOT

    raw   = json.loads(ATOMS_PATH.read_text(encoding="utf-8"))
    atoms = list(raw.get("_default", {}).values())
    atom_ids = {a["canonical_id"] for a in atoms
                if (a.get("domain") or a.get("vertical")) == vertical}
    atom_fields = {
        cid: {f["name"] for f in next(
            a for a in atoms if a["canonical_id"] == cid
        ).get("fields", [])}
        for cid in atom_ids
    }

    rels_path = PROJECT_ROOT / "data" / "atom_relationships.json"
    all_rels  = json.loads(rels_path.read_text(encoding="utf-8"))

    kept    = []
    removed = []
    seen    = set()

    for r in all_rels:
        fa, ff = r.get("from_atom",""), r.get("from_field","")
        ta, tf = r.get("to_atom",""),   r.get("to_field","")

        # Keep rels not involving this vertical at all
        if fa not in atom_ids and ta not in atom_ids:
            kept.append(r)
            continue

        # Remove if either atom not found in this vertical
        if fa not in atom_ids or ta not in atom_ids:
            removed.append({"reason":"ATOM_NOT_FOUND", "rel": r})
            continue

        # Remove if fields don't exist
        if ff not in atom_fields.get(fa, set()) or tf not in atom_fields.get(ta, set()):
            removed.append({"reason":"FIELD_NOT_FOUND", "rel": r})
            continue

        # Remove duplicates
        key = (fa, ff, ta, tf)
        if key in seen:
            removed.append({"reason":"DUPLICATE", "rel": r})
            continue
        seen.add(key)
        kept.append(r)

    rels_path.write_text(json.dumps(kept, indent=2, ensure_ascii=False), encoding="utf-8")

    # Re-run discovery to confirm
    from agents.relationship_discovery_agent import run as _run
    result = _run(vertical=vertical)

    return JSONResponse({
        "status":   "ok",
        "removed":  len(removed),
        "kept":     len(kept),
        "removed_details": removed,
        "discovery_after": result["stats"],
        "errors_remaining": len(result["errors"]),
    })


@router.get("/api/agents/discover-relationships/{vertical}")
async def get_relationship_discovery(vertical: str):
    """GET alias — same as POST for convenience."""
    from agents.relationship_discovery_agent import run as _run
    result = _run(vertical=vertical)
    return JSONResponse(result)


@router.post("/api/agents/discover-relationships-p2/{vertical}")
@router.get("/api/agents/discover-relationships-p2/{vertical}")
async def discover_relationships_phase2(vertical: str):
    """
    Run Relationship Discovery Phase 2 — needs CSV mock data.
    Algorithm 2: value inclusion analysis (set intersection on actual CSV data).
    Algorithm 3: cardinality analysis (field role vs actual unique value counts).
    Pure Python — zero AI calls.
    """
    from agents.relationship_discovery_agent import run_phase2 as _run2
    result = _run2(vertical=vertical)
    return JSONResponse(result)


# ── Fix inclusion failures ────────────────────────────────────────────────────

@router.post("/api/agents/fix-compile-failures/{vertical}")
async def fix_compile_failures(vertical: str):
    """
    Auto-fix compile failures and recompile the vertical.

    Fixes applied before recompile:
    1. SUM of percent fields across rows (oracle > 100%) — the range check now
       allows unbounded values for SUM/total aggregations on percent fields,
       since SUM(completion_percent) over 30 rows CAN legitimately be 1852.
    2. Negative MIN/MAX values for days/count — already fixed in _shared.py.
    3. Complement identity on state atoms — recompile resolves after grain fix.

    The recompile runs as a background job.
    """
    from core.job_manager import create_job, run_job

    job_id = create_job(f"Fix & Recompile — {vertical}")
    log.info(f"[fix_compile] Starting fix+recompile for {vertical}")

    def _job():
        from compiler.compiler_orchestrator import compile_vertical
        result = compile_vertical(vertical=vertical)
        return {
            "status":         "ok",
            "total_compiled": result.get("total_compiled", 0),
            "failed_after":   result.get("failed", 0),
            "compiled_a":     result.get("compiled_a", 0),
            "compiled_b":     result.get("compiled_b", 0),
            "compiled_c":     result.get("compiled_c", 0),
            "errors":         result.get("errors", []),
        }

    run_job(job_id, _job)
    return JSONResponse({"job_id": job_id, "status": "pending"})


@router.post("/api/agents/fix-cardinality/{vertical}/{canonical_id}")
async def fix_cardinality_issue(vertical: str, canonical_id: str):
    """
    Fix a CRITICAL cardinality issue by regenerating mock data for one atom.
    Used when an atom has genuine duplicate PKs in its CSV.
    Passes only_atom_ids so only this one atom is regenerated.
    """
    from core.job_manager import create_job, run_job
    from agents.mock_data_agent import run as _mock_run

    job_id = create_job(f"Fix Cardinality — {canonical_id}")

    def _job():
        result = _mock_run(
            vertical=vertical,
            mode="GENERIC",
            only_atom_ids={canonical_id},
        )
        # Re-run Phase 2 to confirm fix
        from agents.relationship_discovery_agent import run_phase2
        phase2 = run_phase2(vertical)
        remaining = [
            c for c in phase2.get("cardinality_results", [])
            if c.get("severity") == "CRITICAL"
        ]
        return {
            "status":             "ok",
            "atom_regenerated":   canonical_id,
            "mock_result":        result,
            "criticals_remaining": len(remaining),
        }

    run_job(job_id, _job)
    return JSONResponse({"job_id": job_id, "status": "pending"})


@router.post("/api/agents/fix-inclusion/{vertical}")
async def fix_inclusion_failures(vertical: str):
    """Fix FK inclusion failures as a background job."""
    from core.job_manager import create_job, run_job
    from agents.relationship_discovery_agent import get_inclusion_failures, run_phase2
    from agents.mock_data_agent import run as _mock_run

    failures = get_inclusion_failures(vertical)
    if not failures:
        return JSONResponse({"status": "ok", "message": "No inclusion failures found.", "fixed": []})

    atoms_to_fix = list({f["canonical_id"] for f in failures})
    job_id = create_job(f"Fix Inclusion — {vertical} ({len(atoms_to_fix)} atoms)")

    def _job():
        result = _mock_run(vertical=vertical, mode="GENERIC", only_atom_ids=set(atoms_to_fix))
        phase2 = run_phase2(vertical)
        return {
            "status":          "ok",
            "atoms_fixed":     atoms_to_fix,
            "mock_result":     result,
            "phase2_after":    phase2["stats"],
            "remaining_fails": [
                r["message"] for r in phase2.get("inclusion_results", [])
                if r.get("status") == "FAIL"
            ],
        }
    run_job(job_id, _job)
    return JSONResponse({"job_id": job_id, "status": "pending"})


# ── Seed endpoints ────────────────────────────────────────────────────────────

class SeedRequest(BaseModel):
    vertical: str


@router.post("/api/agents/seed")
async def run_seed_agent(req: SeedRequest):
    from core.job_manager import create_job, run_job
    job_id = create_job(f"Seed File — {req.vertical}")
    def _job():
        from agents.seed_agent import run as _run
        return _run(vertical=req.vertical)
    run_job(job_id, _job)
    return JSONResponse({"job_id": job_id, "status": "pending"})

@router.post("/api/agents/seed-sync")
async def run_seed_agent_sync(req: SeedRequest):
    """
    Run SeedAgent for a vertical.
    Generates data/seeds/{vertical}_seed.json.

    Prerequisites:
      - atoms.json must have atoms for this vertical (VerticalSchemaAgent)
      - field_values.json must exist (MockDataAgent + discover_enums)
    """
    from agents.seed_agent import run
    try:
        result = run(vertical=req.vertical)
        # Don't return the full seed in the API response (too large)
        return JSONResponse({k: v for k, v in result.items() if k != "seed"})
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/api/agents/seed/{vertical}")
async def get_seed(vertical: str):
    """Return the seed file for a vertical if it exists."""
    from core.paths import SEEDS_DIR
    seed_path = SEEDS_DIR / f"{vertical}_seed.json"
    if not seed_path.exists():
        raise HTTPException(404, f"No seed file for vertical '{vertical}'")
    try:
        return JSONResponse(json.loads(seed_path.read_text(encoding="utf-8")))
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Vocabulary endpoints ───────────────────────────────────────────────────────

class VocabRequest(BaseModel):
    vertical: str
    skip_ai:  bool = False


@router.post("/api/agents/vocabulary")
async def run_vocabulary_agent(req: VocabRequest):
    from core.job_manager import create_job, run_job
    job_id = create_job(f"Vocabulary — {req.vertical}")
    def _job():
        from agents.vocabulary_agent import run as _run
        return _run(vertical=req.vertical, skip_ai=req.skip_ai)
    run_job(job_id, _job)
    return JSONResponse({"job_id": job_id, "status": "pending"})

@router.post("/api/agents/vocabulary-sync")
async def run_vocabulary_agent_sync(req: VocabRequest):
    """
    Run VocabularyAgent for a vertical.
    Generates data/dialects/{vertical}_dialect.json.

    Prerequisites:
      - data/seeds/{vertical}_seed.json must exist (SeedAgent)
      - data/field_values.json must exist (MockDataAgent + discover_enums)

    Set skip_ai=true to skip Phase C (AI synonym expansion).
    Faster and cheaper but lower vocabulary coverage.
    """
    from agents.vocabulary_agent import run
    try:
        result = run(vertical=req.vertical, skip_ai=req.skip_ai)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/api/agents/vocabulary/{vertical}")
async def get_dialect(vertical: str):
    """Return the dialect file for a vertical if it exists."""
    from core.paths import DIALECTS_DIR
    path = DIALECTS_DIR / f"{vertical}_dialect.json"
    if not path.exists():
        raise HTTPException(404, f"No dialect file for '{vertical}'")
    try:
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")))
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Goal generation endpoints (Waves 1-9) ───────────────────────────────────────

class GoalsRequest(BaseModel):
    vertical: str


@router.post("/api/agents/goals")
async def run_goal_generator(req: GoalsRequest):
    from core.job_manager import create_job, run_job
    job_id = create_job(f"Goal Generator — {req.vertical}")
    def _job():
        from services.goal_generator import generate_goals
        from agents.domain_goals_agent import generate_domain_goals
        r1 = generate_goals(vertical=req.vertical)
        r2 = generate_domain_goals(
            vertical=req.vertical,
            wave_19_cids=r1.get("all_cids", []),
        )
        return {**r1, "domain_goals": r2}
    run_job(job_id, _job)
    return JSONResponse({"job_id": job_id, "status": "pending"})

@router.post("/api/agents/goals-sync")
async def run_goal_generator_sync(req: GoalsRequest):
    """
    Generate all goals (Waves 1-9 algebraic + Waves A-E domain expert) for a vertical.

    Prerequisites:
      - data/seeds/{vertical}_seed.json must exist (SeedAgent)
    """
    from services.goal_generator import generate_goals
    from agents.domain_goals_agent import generate_domain_goals
    try:
        # Step 1 — Waves 1-9: pure Python, instant, $0
        result_19 = generate_goals(vertical=req.vertical)

        # Step 2 — Waves A-E: single Claude call, domain expert goals
        result_ae = generate_domain_goals(
            vertical=req.vertical,
            wave_19_cids=result_19.get("all_cids", []),
        )

        return JSONResponse({
            "status":         "success",
            "vertical":       req.vertical,
            "domain":         result_19.get("domain"),
            "total_goals_19": result_19.get("total_goals", 0),
            "total_goals_ae": result_ae.get("total_ae_goals", 0),
            "total_goals":    result_19.get("total_goals", 0) + result_ae.get("total_ae_goals", 0),
            "waves_19":       result_19.get("waves", {}),
            "waves_ae":       result_ae.get("waves_ae", {}),
            "dropped_ae":     len(result_ae.get("dropped", [])),
            "calls_made":     result_ae.get("calls_made", 1),
        })
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/api/agents/goals/{vertical}")
async def get_goals_summary(vertical: str):
    """Return the generation summary for a vertical if goals exist."""
    from services.goal_generator import get_goals_summary as _get_summary
    summary = _get_summary(vertical)
    if summary is None:
        raise HTTPException(404, f"No goals generated for '{vertical}'")
    return JSONResponse(summary)


@router.get("/api/agents/goals/{vertical}/{wave}")
async def get_goals_wave(vertical: str, wave: str):
    """Return a specific wave file (e.g. wave='wave_01') for a vertical."""
    from services.goal_generator import get_wave
    data = get_wave(vertical, wave)
    if data is None:
        raise HTTPException(404, f"No '{wave}' file for vertical '{vertical}'")
    return JSONResponse(data)


# ── Compilation endpoints (Branch A + Branch C) ─────────────────────────────

class CompileRequest(BaseModel):
    vertical: str


@router.post("/api/agents/compile")
async def run_compiler(req: CompileRequest):
    from core.job_manager import create_job, run_job
    job_id = create_job(f"Compiler — {req.vertical}")
    def _job():
        from compiler.compiler_orchestrator import compile_vertical
        return compile_vertical(vertical=req.vertical)
    run_job(job_id, _job)
    return JSONResponse({"job_id": job_id, "status": "pending"})

@router.post("/api/agents/compile-sync")
async def run_compiler_sync(req: CompileRequest):
    """
    Compile all goals for a vertical (Branch A algebraic + Branch C composition).

    Prerequisites:
      - Goals must be generated first (POST /api/agents/goals)
      - Atoms, mock data, and field_values must exist
    """
    from compiler.compiler_orchestrator import compile_vertical
    try:
        result = compile_vertical(vertical=req.vertical)
        return JSONResponse(result)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/api/agents/compile/{vertical}")
async def get_compile_status(vertical: str):
    """Return canonical_index summary for a vertical if compilation has run."""
    from core.paths import CANONICAL_INDEX_PATH, LOCK_REGISTRY_PATH
    if not CANONICAL_INDEX_PATH.exists():
        raise HTTPException(404, "No compilation results found. Run POST /api/agents/compile first.")
    try:
        index    = json.loads(CANONICAL_INDEX_PATH.read_text(encoding="utf-8"))
        registry = json.loads(LOCK_REGISTRY_PATH.read_text(encoding="utf-8")) \
            if LOCK_REGISTRY_PATH.exists() else {}
        locked   = sum(1 for v in index.values() if v.get("ai_locked"))
        verified = sum(1 for v in index.values() if v.get("verified"))
        return JSONResponse({
            "vertical":        vertical,
            "total_compiled":  len(index),
            "ai_locked":       locked,
            "verified":        verified,
            "lock_rate":       round(locked / len(index) * 100, 1) if index else 0,
        })
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Verification endpoints ────────────────────────────────────────────────────

class VerifyRequest(BaseModel):
    vertical: str



class WorkflowAnalyzeRequest(BaseModel):
    vertical:      str
    canonical_ids: List[str]
    model_keys:    List[str] = ["claude"]

class WaDecisionRequest(BaseModel):
    vertical:     str
    canonical_id: str
    status:       str   # "approved" | "rejected"
    reviewed_by:  str
    note:         str = ""

class WaKeyRequest(BaseModel):
    provider: str   # "openai" | "google"
    api_key:  str


@router.post("/api/agents/verify")
async def run_verifier(req: VerifyRequest):
    from core.job_manager import create_job, run_job
    job_id = create_job(f"Verifier — {req.vertical}")
    def _job():
        from compiler.verifier import verify_vertical
        from core.paths import LOCK_REGISTRY_PATH
        import json
        result = verify_vertical(vertical=req.vertical)
        # Add pending_details to result
        reason_labels = {
            "STATIC_SUSPICIOUS":    "Formula returns same value at all data sizes",
            "LLM_WIZARD_execution":   "Wizard compiled — exec_formula unavailable for re-execution",
            "VERIFY_wizard_execution":"Wizard compiled and re-executed successfully",
            "COMPLEMENT_FAILED":    "Filter math does not add up (state atom DEDUP issue)",
            "VERIFY_execution":     "Formula ran but stability check pending",
            "ORACLE_DRIFT":         "Formula result drifted from stored oracle",
        }
        if LOCK_REGISTRY_PATH.exists():
            lock   = json.loads(LOCK_REGISTRY_PATH.read_text(encoding="utf-8"))
            aterms = lock.get("aterms", lock)
            pending_details = [
                {
                    "canonical_id":  cid,
                    "verify_reason": entry.get("verify_reason", "UNKNOWN"),
                    "reason_label":  reason_labels.get(entry.get("verify_reason",""), entry.get("verify_reason","")),
                    "formula":       (entry.get("formula","") or "")[:100],
                }
                for cid, entry in aterms.items()
                if cid.startswith(req.vertical) and not entry.get("ai_locked")
            ]
            result["pending_details"] = pending_details
            result["vertical"] = req.vertical
        return result
    run_job(job_id, _job)
    return JSONResponse({"job_id": job_id, "status": "pending"})

@router.post("/api/agents/verify-sync")
async def run_verifier_sync(req: VerifyRequest):
    """
    Post-compilation verification pass (no LLM, pure Python).
    Re-executes all formulas, runs complement identity and stability tests,
    updates ai_locked status in canonical_index and lock_registry.

    Prerequisites: compilation must have run first.
    """
    from compiler.verifier import verify_vertical
    try:
        result = verify_vertical(vertical=req.vertical)
        return JSONResponse(result)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/api/agents/verify/{vertical}")
async def get_verify_status(vertical: str):
    """Return verification summary for a vertical."""
    from core.paths import CANONICAL_INDEX_PATH
    if not CANONICAL_INDEX_PATH.exists():
        raise HTTPException(404, "No compilation results. Run compile first.")
    try:
        index  = json.loads(CANONICAL_INDEX_PATH.read_text(encoding="utf-8"))
        locked = sum(1 for v in index.values() if v.get("ai_locked"))
        return JSONResponse({
            "vertical":       vertical,
            "total_compiled": len(index),
            "ai_locked":      locked,
            "lock_rate":      round(locked / len(index) * 100, 1) if index else 0,
        })
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Deployment endpoints ──────────────────────────────────────────────────────

class DeployRequest(BaseModel):
    vertical: str
    version:  Optional[str] = None


@router.post("/api/agents/deploy")
async def run_deployment_engine(req: DeployRequest):
    """
    Build a deployment package for a vertical.
    Runs dependency resolution, schema validation, and package assembly.
    Produces a zip in data/packages/ ready for the customer runtime.

    Prerequisites: compilation + verification must have run first.
    """
    from compiler.deployment_engine import package_vertical
    try:
        result = package_vertical(vertical=req.vertical, version=req.version)
        return JSONResponse(result)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/api/agents/deploy/{vertical}")
async def get_deployments(vertical: str):
    """List all deployment packages built for a vertical."""
    from compiler.deployment_engine import get_packages
    try:
        packages = get_packages(vertical)
        return JSONResponse({"vertical": vertical, "packages": packages})
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/api/factory/atom/{canonical_id}/questions")
async def atom_questions(canonical_id: str):
    """
    Return questions (source_goal) that reference this atom, each annotated
    with the columns it uses — so the Verticals tab can filter questions by
    column when the user clicks a column pill.

    Each entry: { "text": "...", "columns": ["col_a", "col_b", ...] }
    """
    import json as _json
    import re as _re
    from core.paths import ATERMS_DIR

    def _extract_columns(formula_line: str) -> list:
        """Extract every column name referenced in a formula_line."""
        cols = set()
        fl = formula_line

        # SQL style: AGG(col) and WHERE col=val and GROUP BY col and BUCKET(col,...)
        cols.update(_re.findall(
            r'(?:COUNT_DISTINCT|SUM|MAX|MIN|AVG|BUCKET)\s*\(\s*([a-z][a-z0-9_]*)\s*[\),]',
            fl, _re.IGNORECASE
        ))
        cols.update(_re.findall(r'WHERE\s+([a-z][a-z0-9_]*)\s*=', fl, _re.IGNORECASE))
        cols.update(_re.findall(r'GROUP\s+BY\s+([a-z][a-z0-9_]*)', fl, _re.IGNORECASE))

        # Python-callable: MEASURE('atom', 'AGG', 'col', [{'field': 'filter_col'...}])
        # 3rd positional arg is the aggregation column
        m = _re.match(
            r"""MEASURE[A-Z_]*\s*\(\s*'[^']+'\s*,\s*'[^']+'\s*,\s*'([^']+)'""",
            fl
        )
        if m:
            cols.add(m.group(1))
        # filter fields
        cols.update(_re.findall(r"'field'\s*:\s*'([^']+)'", fl))
        # date_col arg (time-scoped operators)
        cols.update(_re.findall(r"date_col\s*=\s*'([^']+)'", fl))
        # group_by_col arg
        cols.update(_re.findall(r"group_by_col\s*=\s*'([^']+)'", fl))

        # Remove atom canonical_id tokens that leak in (they contain underscores)
        # Keep only lowercase words that look like column names (no more than 4 parts)
        cleaned = {
            c.lower() for c in cols
            if c and not c.startswith(canonical_id.split("_")[0])
            and len(c.split("_")) <= 5
            and not c.endswith("_record")
            and not c.endswith("_dimension")
            and not c.endswith("_snapshot")
            and not c.endswith("_event")
        }

        # For composition formulas {dep_cid} extract the measure/column
        # from the dependency canonical_id (e.g. weight_kg_count → weight_kg)
        if not cleaned:
            for dep in _re.findall(r'\{([a-z][a-z0-9_]+)\}', formula_line):
                # strip domain prefix (e.g. 'logistics_cargo_items_') and unit suffix
                parts = dep.split('_')
                # find the atom prefix length by matching canonical_id prefix
                prefix = canonical_id.rsplit('_', 1)[0]  # e.g. logistics_cargo_items
                prefix_parts = prefix.split('_')
                remaining = parts[len(prefix_parts):]
                # last part is usually unit (count/currency/percent) — drop it
                col_parts = remaining[:-1] if len(remaining) > 1 else remaining
                if col_parts:
                    cleaned.add('_'.join(col_parts))

        return sorted(cleaned)

    questions = []
    seen      = set()

    if ATERMS_DIR.exists():
        for f in sorted(ATERMS_DIR.glob("aterm_*.json")):
            try:
                data         = _json.loads(f.read_text(encoding="utf-8"))
                formula_line = data.get("formula_line", "")
                source_goal  = data.get("source_goal", "").strip()
                if (canonical_id.lower() in formula_line.lower()
                        and source_goal and source_goal not in seen):
                    seen.add(source_goal)
                    questions.append({
                        "text":    source_goal,
                        "columns": _extract_columns(formula_line),
                    })
            except Exception:
                continue

    return JSONResponse({
        "canonical_id": canonical_id,
        "questions":    questions,
        "count":        len(questions),
    })


@router.get("/api/factory/verticals")
async def factory_verticals():
    """
    Return all verticals that have atoms in the factory atom registry,
    along with their atom canonical_ids and field names per atom.
    Used by the customer runtime Verticals tab.
    """
    from services.atom_registry import get_all_atoms
    atoms = get_all_atoms()

    # Group by domain
    verticals: Dict[str, Dict] = {}
    for atom in atoms:
        domain = atom.get("domain", "")
        if not domain:
            continue
        if domain not in verticals:
            verticals[domain] = {"name": domain, "atoms": []}
        # Derive system tag from canonical_id suffix
        cid    = atom.get("canonical_id", "")
        parts  = cid.split("_")
        system = parts[-2].upper() if len(parts) >= 2 and parts[-2].upper() in {"ERP","WMS","CRM","MANUAL","API","MRP","TMS","WES"} else ""
        verticals[domain]["atoms"].append({
            "canonical_id": cid,
            "name":         atom.get("name", ""),
            "description":  atom.get("description", ""),
            "system":       system,
            "fields":       [f.get("name", "") for f in atom.get("fields", [])],
        })

    result = sorted(verticals.values(), key=lambda v: v["name"])
    for v in result:
        v["atoms"].sort(key=lambda a: a["canonical_id"])

    return JSONResponse({"verticals": result, "count": len(result)})


@router.get("/api/atoms")
async def get_atoms():
    from services.atom_registry import get_all_atoms
    atoms = get_all_atoms()
    return JSONResponse({"atoms": atoms, "count": len(atoms)})


@router.get("/api/atoms/relationships")
async def get_relationships():
    from services.relationship_registry import list_relationships
    rels = list_relationships()
    return JSONResponse({"relationships": rels, "count": len(rels)})


# ── Database Tab endpoints ─────────────────────────────────────────────────────

@router.get("/api/db/verticals")
async def db_verticals():
    """List all verticals that have data available."""
    import core.paths as _p
    verticals = set()
    # From atoms
    if _p.ATOMS_PATH.exists():
        import json as _j
        raw = _j.loads(_p.ATOMS_PATH.read_text())
        entries = raw.get("_default", raw) if isinstance(raw, dict) and "_default" in raw else raw
        if isinstance(entries, dict):
            for atom in entries.values():
                cid = atom.get("canonical_id", "")
                if "_" in cid:
                    verticals.add(cid.split("_")[0])
    # From aterms dir
    if _p.ATERMS_DIR.exists():
        for f in _p.ATERMS_DIR.iterdir():
            name = f.stem.replace("aterm_", "")
            if "_" in name:
                verticals.add(name.split("_")[0])
    # From goals dir
    goals_dir = _p.DATA_DIR / "goals"
    if goals_dir.exists():
        for d in goals_dir.iterdir():
            if d.is_dir():
                verticals.add(d.name)
    return JSONResponse({"verticals": sorted(verticals)})


@router.get("/api/db/atoms")
async def db_atoms(
    vertical: str = "",
    page: int = 1,
    limit: int = 50,
    search: str = "",
    record_type: str = "",
):
    """Paginated atoms for the DB viewer."""
    import json as _j
    import core.paths as _p

    raw = _j.loads(_p.ATOMS_PATH.read_text()) if _p.ATOMS_PATH.exists() else {}
    entries = raw.get("_default", raw) if "_default" in raw else raw
    atoms = list(entries.values()) if isinstance(entries, dict) else raw if isinstance(raw, list) else []

    # Filter
    if vertical:
        atoms = [a for a in atoms if a.get("canonical_id", "").startswith(vertical + "_")]
    if search:
        s = search.lower()
        atoms = [a for a in atoms if s in a.get("canonical_id", "").lower()]
    if record_type:
        atoms = [a for a in atoms if a.get("record_type", "") == record_type]

    total = len(atoms)
    atoms.sort(key=lambda a: a.get("canonical_id", ""))
    start = (page - 1) * limit
    page_items = atoms[start:start + limit]

    # Slim payload — just what the table needs, full detail on expand
    slim = []
    for a in page_items:
        slim.append({
            "canonical_id":  a.get("canonical_id", ""),
            "record_type":   a.get("record_type", ""),
            "field_count":   len(a.get("fields", [])),
            "grain_keys":    a.get("grain_keys", []),
            "fields":        a.get("fields", []),
            "source":        a.get("source", ""),
        })

    return JSONResponse({
        "atoms":      slim,
        "total":      total,
        "page":       page,
        "limit":      limit,
        "pages":      max(1, (total + limit - 1) // limit),
    })


@router.get("/api/db/aterms")
async def db_aterms(
    vertical: str = "",
    page: int = 1,
    limit: int = 50,
    search: str = "",
    wave: str = "",
    lock_method: str = "",
    ai_locked: str = "",
    verify_reason: str = "",
):
    """Paginated aterms for the DB viewer.
    Reads canonical_index.json + lock_registry.json (2 file reads total)
    instead of iterating individual aterm files — much faster at scale.
    Full aterm detail is fetched per-row only when user expands a row.
    """
    import json as _j
    import core.paths as _p

    if not _p.CANONICAL_INDEX_PATH.exists():
        return JSONResponse({"aterms": [], "total": 0, "page": 1, "limit": limit, "pages": 1})

    ci = _j.loads(_p.CANONICAL_INDEX_PATH.read_text())
    lr_raw = _j.loads(_p.LOCK_REGISTRY_PATH.read_text()) if _p.LOCK_REGISTRY_PATH.exists() else {}
    lr = lr_raw.get("aterms", lr_raw)

    # Build merged list
    aterms = []
    for cid, entry in ci.items():
        if vertical and not cid.startswith(vertical + "_"):
            continue
        lock = lr.get(cid, {})
        w = entry.get("wave", "")
        if isinstance(w, int) or (isinstance(w, str) and w.isdigit()):
            wave_str = f"wave_{int(w):02d}"
        elif isinstance(w, str) and w.upper() in "ABCDE" and len(w) == 1:
            wave_str = f"wave_{w.upper()}"
        elif isinstance(w, str) and w.startswith("wave_"):
            wave_str = w
        else:
            wave_str = f"wave_{w}" if w else ""
        aterms.append({
            "canonical_id":  cid,
            "source_goal":   entry.get("source_goal", ""),
            "wave":          wave_str,
            "oracle_value":  entry.get("oracle_value"),
            "formula_line":  entry.get("formula_line", ""),
            "lock_method":   lock.get("lock_method", ""),
            "verify_reason": lock.get("verify_reason", entry.get("verify_reason", "")),
            "ai_locked":     lock.get("ai_locked", entry.get("ai_locked", False)),
            "verified_at":   lock.get("verified_at", ""),
            "slots":         entry.get("slots", {}),
            "via_llm":       lock.get("lock_method", "") == "llm_wizard",
            "wizard_steps":  0,
        })

    # Filters
    if search:
        s = search.lower()
        aterms = [a for a in aterms if s in a["canonical_id"].lower()
                  or s in a["source_goal"].lower()]
    if wave:
        aterms = [a for a in aterms if a["wave"] == wave]
    if lock_method:
        aterms = [a for a in aterms if a["lock_method"] == lock_method]
    if ai_locked != "":
        locked = ai_locked.lower() == "true"
        aterms = [a for a in aterms if bool(a["ai_locked"]) == locked]
    if verify_reason:
        aterms = [a for a in aterms if a["verify_reason"] == verify_reason]

    total = len(aterms)
    start = (page - 1) * limit
    page_items = aterms[start:start + limit]

    return JSONResponse({
        "aterms": page_items,
        "total":  total,
        "page":   page,
        "limit":  limit,
        "pages":  max(1, (total + limit - 1) // limit),
    })


@router.get("/api/db/relationships")
async def db_relationships(
    vertical: str = "",
    page: int = 1,
    limit: int = 50,
    search: str = "",
):
    """Paginated relationships for the DB viewer."""
    import json as _j
    import core.paths as _p

    rel_path = _p.DATA_DIR / "atom_relationships.json"
    rels = _j.loads(rel_path.read_text()) if rel_path.exists() else []

    if vertical:
        rels = [r for r in rels if vertical in r.get("from_atom", "") or vertical in r.get("to_atom", "")]
    if search:
        s = search.lower()
        rels = [r for r in rels if s in r.get("from_atom", "").lower()
                or s in r.get("to_atom", "").lower()
                or s in r.get("from_field", "").lower()
                or s in r.get("to_field", "").lower()]

    total = len(rels)
    rels.sort(key=lambda r: r.get("from_atom", ""))
    start = (page - 1) * limit
    page_items = rels[start:start + limit]

    return JSONResponse({
        "relationships": page_items,
        "total":         total,
        "page":          page,
        "limit":         limit,
        "pages":         max(1, (total + limit - 1) // limit),
    })


@router.get("/api/db/goals")
async def db_goals(
    vertical: str = "",
    page: int = 1,
    limit: int = 50,
    search: str = "",
    wave: str = "",
):
    """Paginated goals for the DB viewer."""
    import json as _j
    import core.paths as _p

    goals_root = _p.DATA_DIR / "goals"
    if not goals_root.exists():
        return JSONResponse({"goals": [], "total": 0, "page": 1, "limit": limit, "pages": 1})

    all_goals = []
    dirs = [goals_root / vertical] if vertical else [d for d in goals_root.iterdir() if d.is_dir()]

    for vdir in dirs:
        vname = vdir.name
        for wf in sorted(vdir.iterdir()):
            if not wf.name.endswith(".json") or wf.name in ("generation_summary.json", "pending_branch_b.json"):
                continue
            wave_name = wf.stem
            try:
                data = _j.loads(wf.read_text())
                glist = data.get("goals", [])
                for g in glist:
                    all_goals.append({
                        "vertical":     vname,
                        "wave":         wave_name,
                        "goal":         g.get("goal", ""),
                        "canonical_id": g.get("canonical_id", ""),
                        "slots":        g.get("slots", {}),
                        "complexity":   g.get("complexity", ""),
                    })
            except Exception:
                continue

    if wave:
        all_goals = [g for g in all_goals if g["wave"] == wave]
    if search:
        s = search.lower()
        all_goals = [g for g in all_goals if s in g["goal"].lower() or s in g["canonical_id"].lower()]

    total = len(all_goals)
    start = (page - 1) * limit
    page_items = all_goals[start:start + limit]

    return JSONResponse({
        "goals": page_items,
        "total": total,
        "page":  page,
        "limit": limit,
        "pages": max(1, (total + limit - 1) // limit),
    })


@router.get("/api/db/aterm-detail/{cid:path}")
async def db_aterm_detail(cid: str):
    """Fetch full aterm detail for row expansion — called only when user expands a row."""
    import json as _j
    import core.paths as _p
    aterm_path = _p.ATERMS_DIR / f"aterm_{cid}.json"
    if not aterm_path.exists():
        raise HTTPException(404, f"Aterm not found: {cid}")
    return JSONResponse(_j.loads(aterm_path.read_text()))


@router.get("/api/db/field-values")
async def db_field_values(vertical: str = "", search: str = ""):
    """Field values grouped by atom — load all (reasonable size per vertical)."""
    import json as _j
    import core.paths as _p

    fv_path = _p.DATA_DIR / "field_values.json"
    if not fv_path.exists():
        return JSONResponse({"atoms": [], "total_keys": 0})

    fv = _j.loads(fv_path.read_text())

    # Group by atom
    grouped = {}
    for key, values in fv.items():
        parts = key.split(".")
        if len(parts) < 2:
            continue
        atom_cid = parts[0]
        field    = ".".join(parts[1:])
        if vertical and not atom_cid.startswith(vertical + "_"):
            continue
        if search and search.lower() not in atom_cid.lower() and search.lower() not in field.lower():
            continue
        if atom_cid not in grouped:
            grouped[atom_cid] = []
        grouped[atom_cid].append({"field": field, "values": values})

    result = [{"atom": k, "fields": v} for k, v in sorted(grouped.items())]
    return JSONResponse({"atoms": result, "total_keys": len(fv)})


@router.get("/api/db/dialects")
async def db_dialects(vertical: str = ""):
    """Dialect files — load all (one per vertical, always small)."""
    import json as _j
    import core.paths as _p

    dialects_dir = _p.DATA_DIR / "dialects"
    if not dialects_dir.exists():
        return JSONResponse({"dialects": []})

    result = []
    for f in sorted(dialects_dir.iterdir()):
        if not f.name.endswith("_dialect.json"):
            continue
        vname = f.stem.replace("_dialect", "")
        if vertical and vname != vertical:
            continue
        try:
            data = _j.loads(f.read_text())
            result.append({"vertical": vname, "data": data})
        except Exception:
            continue

    return JSONResponse({"dialects": result})


@router.get("/api/db/seeds")
async def db_seeds(vertical: str = ""):
    """Seed files — load all (one per vertical, always small)."""
    import json as _j
    import core.paths as _p

    seeds_dir = _p.DATA_DIR / "seeds"
    if not seeds_dir.exists():
        return JSONResponse({"seeds": []})

    result = []
    for f in sorted(seeds_dir.iterdir()):
        if not f.name.endswith("_seed.json"):
            continue
        vname = f.stem.replace("_seed", "")
        if vertical and vname != vertical:
            continue
        try:
            data = _j.loads(f.read_text())
            result.append({"vertical": vname, "data": data})
        except Exception:
            continue

    return JSONResponse({"seeds": result})




@router.post("/api/agents/mock-data")
async def run_mock_data(req: MockDataRequest):
    """Run MockDataAgent as a background job — returns job_id immediately.
    If canonical_id is set, regenerates only that one atom's CSV.
    """
    from core.job_manager import create_job, run_job
    from agents.mock_data_agent import run as _run, run_single_atom as _run_single
    if req.mode not in ("GENERIC", "CUSTOMER"):
        raise HTTPException(400, "mode must be GENERIC or CUSTOMER")

    if req.canonical_id:
        job_id = create_job(f"Mock Data — {req.canonical_id}")
        cid = req.canonical_id
        vert = req.vertical
        def _job():
            return _run_single(vertical=vert, canonical_id=cid)
        run_job(job_id, _job)
    else:
        job_id = create_job(f"Mock Data — {req.vertical}")
        vert = req.vertical
        mode = req.mode
        def _job():
            return _run(vertical=vert, mode=mode)
        run_job(job_id, _job)

    return JSONResponse({"job_id": job_id, "status": "pending"})


@router.get("/api/agents/mock-data/{vertical}")
async def get_mock_data_status(vertical: str):
    """
    Check mock data completeness for a vertical — atoms vs CSVs generated.
    Returns counts, missing atom list, and incomplete flag.
    """
    from core.paths import MOCK_DATA_DIR, ATOMS_PATH
    import json

    # All atoms for this vertical
    raw   = json.loads(ATOMS_PATH.read_text(encoding="utf-8"))
    atoms = [a for a in raw.get("_default", {}).values() if a.get("domain") == vertical]

    # CSVs on disk
    vertical_dir = MOCK_DATA_DIR / vertical
    csvs = []
    if vertical_dir.exists():
        for csv_file in sorted(vertical_dir.glob("*.csv")):
            try:
                rows = sum(1 for _ in open(csv_file, encoding="utf-8")) - 1
            except Exception:
                rows = -1
            csvs.append({"file": csv_file.name, "stem": csv_file.stem, "rows": rows})

    covered    = {c["stem"] for c in csvs}
    atom_ids   = {a["canonical_id"] for a in atoms}
    missing    = [
        {"canonical_id": cid, "record_type": next(
            (a.get("record_type","?") for a in atoms if a["canonical_id"]==cid), "?"
        )}
        for cid in sorted(atom_ids - covered)
    ]
    incomplete = len(missing) > 0

    return JSONResponse({
        "vertical":      vertical,
        "atoms_total":   len(atoms),
        "csvs_generated": len(csvs),
        "atoms_missing": len(missing),
        "missing_atoms": missing,
        "incomplete":    incomplete,
        "csvs":          csvs,
    })


@router.get("/api/stats")
async def get_stats():
    """Aggregate dashboard stats — single call for the factory UI."""
    import json
    from core.paths import (
        CANONICAL_INDEX_PATH, LOCK_REGISTRY_PATH, GOALS_DIR, PACKAGES_DIR
    )
    from compiler.deployment_engine import get_packages

    result = {
        "verticals": {},
        "totals": {"verticals": 0, "atoms": 0, "goals": 0, "compiled": 0, "proven": 0, "pending": 0},
        "branches": {"A": 0, "B": 0, "C": 0},
        "waves": {},
        "packages": [],
    }

    # ── Atoms (read directly from TinyDB JSON — no import needed) ──
    from core.paths import ATOMS_PATH
    if ATOMS_PATH.exists():
        raw = json.loads(ATOMS_PATH.read_text(encoding="utf-8"))
        atoms_data = list(raw.get("_default", {}).values())
        for a in atoms_data:
            v = a.get("domain") or a.get("vertical") or "unknown"
            if v not in result["verticals"]:
                result["verticals"][v] = {"atoms": 0, "aterms": 0, "proven": 0}
            result["verticals"][v]["atoms"] += 1
            result["totals"]["atoms"] += 1
    result["totals"]["verticals"] = len(result["verticals"])

    # ── Lock registry — source of truth for proven + branches ──
    if LOCK_REGISTRY_PATH.exists():
        registry = json.loads(LOCK_REGISTRY_PATH.read_text(encoding="utf-8"))
        aterms_reg = registry.get("aterms", registry)
        proven = 0
        pending_details = []
        reason_labels = {
            "STATIC_SUSPICIOUS":      "Formula returns same value regardless of data size",
            "LLM_WIZARD_execution":   "Wizard compiled — exec_formula unavailable for re-execution",
            "VERIFY_wizard_execution":"Wizard compiled and re-executed successfully",
            "NO_FORMULA":             "No formula was generated during compilation",
            "ORACLE_DRIFT":           "Formula result drifted from stored oracle value",
            "SKIP_NO_EXEC":           "Formula could not be executed during verification",
        }
        for cid, entry in aterms_reg.items():
            if entry.get("ai_locked"):
                proven += 1
            else:
                reason = entry.get("verify_reason", "UNKNOWN")
                # Derive vertical and entity from CID prefix
                parts = cid.split("_")
                vertical = parts[0] if parts else ""
                entity   = parts[1] if len(parts) > 1 else ""
                pending_details.append({
                    "canonical_id":  cid,
                    "verify_reason": reason,
                    "lock_method":   entry.get("lock_method", ""),
                    "reason_label":  reason_labels.get(reason, reason),
                    "formula":       (entry.get("formula") or "")[:120],
                    "vertical":      vertical,
                    "entity":        entity,
                    "oracle_value":  entry.get("oracle_value"),
                    "verified_at":   entry.get("verified_at", ""),
                })
            method = entry.get("lock_method", "")
            if "combinatoric" in method or method == "algebraic":
                result["branches"]["A"] += 1
            elif "domain_expert" in method or "deterministic_b" in method or method.startswith("branch_b"):
                result["branches"]["B"] += 1
            elif "composition" in method:
                result["branches"]["C"] += 1
        result["totals"]["proven"]  = proven
        result["totals"]["pending"] = len(aterms_reg) - proven
        result["pending_details"]   = pending_details

    # ── Canonical index — compiled count split per vertical ──
    if CANONICAL_INDEX_PATH.exists():
        index    = json.loads(CANONICAL_INDEX_PATH.read_text(encoding="utf-8"))
        compiled = len(index)
        result["totals"]["compiled"] = compiled
        # Count aterms per vertical by canonical_id prefix
        for cid in index.keys():
            for v in result["verticals"]:
                if cid.startswith(v):
                    result["verticals"][v]["aterms"] = result["verticals"][v].get("aterms", 0) + 1
                    break

    # ── Per-vertical proven from lock registry ──
    if LOCK_REGISTRY_PATH.exists():
        lock_idx   = json.loads(LOCK_REGISTRY_PATH.read_text(encoding="utf-8"))
        aterms_idx = lock_idx.get("aterms", lock_idx)
        for cid, entry in aterms_idx.items():
            if entry.get("ai_locked"):
                for v in result["verticals"]:
                    if cid.startswith(v):
                        result["verticals"][v]["proven"] = result["verticals"][v].get("proven", 0) + 1
                        break

    # ── Goals — from generation_summary.json per vertical ──
    if GOALS_DIR.exists():
        for vertical_dir in GOALS_DIR.iterdir():
            if not vertical_dir.is_dir():
                continue
            summary_file = vertical_dir / "generation_summary.json"
            if summary_file.exists():
                summary = json.loads(summary_file.read_text(encoding="utf-8"))
                for wave, count in summary.get("waves", {}).items():
                    result["waves"][wave] = result["waves"].get(wave, 0) + count
                    result["totals"]["goals"] += count

    # ── Packages ──
    for v in result["verticals"]:
        result["packages"].extend(get_packages(v))

    return JSONResponse(result)


# ── Settings endpoints ────────────────────────────────────────────────────────

@router.get("/api/settings")
async def get_settings():
    """Return current non-secret settings."""
    from core.config import settings
    return JSONResponse({
        "model": settings.ANTHROPIC_MODEL,
        "has_api_key": bool(settings.ANTHROPIC_API_KEY and settings.ANTHROPIC_API_KEY.startswith("sk-")),
        "available_models": [
            {"id": "claude-sonnet-4-6",  "label": "Claude Sonnet 4.6 (recommended)"},
            {"id": "claude-opus-4-6",    "label": "Claude Opus 4.6 (most capable)"},
            {"id": "claude-haiku-4-5-20251001",   "label": "Claude Haiku 4.5 (fastest)"},
        ]
    })


@router.get("/api/settings/mockdata")
async def get_mockdata_settings():
    """Return current mock data row count settings."""
    from core.config import settings as _s
    import os
    return JSONResponse({
        "dimension": int(os.getenv("MOCK_ROWS_DIMENSION", "15")),
        "record":    int(os.getenv("MOCK_ROWS_RECORD",    "50")),
        "state":     int(os.getenv("MOCK_ROWS_STATE",     "75")),
        "snapshot":  int(os.getenv("MOCK_ROWS_SNAPSHOT",  "50")),
        "event":     int(os.getenv("MOCK_ROWS_EVENT",     "50")),
    })


@router.post("/api/settings/mockdata")
async def save_mockdata_settings(payload: dict):
    """Save mock data row count settings to .env and reload into environment."""
    import os
    from core.paths import PROJECT_ROOT
    env_path = PROJECT_ROOT / ".env"
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []

    def set_var(lines, key, value):
        for i, line in enumerate(lines):
            if line.startswith(f"{key}="):
                lines[i] = f"{key}={value}"
                return lines
        lines.append(f"{key}={value}")
        return lines

    mapping = {
        "dimension": "MOCK_ROWS_DIMENSION",
        "record":    "MOCK_ROWS_RECORD",
        "state":     "MOCK_ROWS_STATE",
        "snapshot":  "MOCK_ROWS_SNAPSHOT",
        "event":     "MOCK_ROWS_EVENT",
    }
    updated = []
    for key, env_key in mapping.items():
        if key in payload:
            val = int(payload[key])
            lines = set_var(lines, env_key, val)
            os.environ[env_key] = str(val)
            updated.append(key)

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return JSONResponse({"status": "saved", "updated": updated})


@router.post("/api/settings")
async def save_settings(payload: dict):
    """
    Save API key and/or model to .env file and reload into running settings.
    """
    from core.config import settings
    from core.paths import PROJECT_ROOT

    env_path = PROJECT_ROOT / ".env"

    # Read existing .env or start fresh
    lines = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    def set_env_var(lines, key, value):
        """Replace or append a key=value line."""
        found = False
        for i, line in enumerate(lines):
            if line.startswith(f"{key}="):
                lines[i] = f"{key}={value}"
                found = True
                break
        if not found:
            lines.append(f"{key}={value}")
        return lines

    updated = []
    if "api_key" in payload and payload["api_key"]:
        lines = set_env_var(lines, "ANTHROPIC_API_KEY", payload["api_key"])
        settings.ANTHROPIC_API_KEY = payload["api_key"]
        updated.append("api_key")

    if "model" in payload and payload["model"]:
        lines = set_env_var(lines, "ANTHROPIC_MODEL", payload["model"])
        settings.ANTHROPIC_MODEL = payload["model"]
        updated.append("model")

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return JSONResponse({"status": "saved", "updated": updated})


@router.post("/api/agents/reverify/{vertical}")
async def reverify_pending(vertical: str):
    """
    Re-run compile + verify for a vertical to attempt fixing pending aterms.
    Compile runs first (Branch A + B + C), then verify locks whatever succeeds.
    """
    from compiler.compiler_orchestrator import compile_vertical
    from compiler.verifier import verify_vertical

    results = {}
    try:
        compile_result = compile_vertical(vertical=vertical)
        results["compile"] = {
            "total_compiled": compile_result.get("total_compiled", 0),
            "status": "ok"
        }
    except Exception as e:
        results["compile"] = {"status": "error", "error": str(e)}
        return JSONResponse({"status": "error", "stage": "compile", "results": results})

    try:
        verify_result = verify_vertical(vertical=vertical)
        results["verify"] = {
            "ai_locked": verify_result.get("ai_locked", 0),
            "pending":   verify_result.get("pending", 0),
            "status": "ok"
        }
    except Exception as e:
        results["verify"] = {"status": "error", "error": str(e)}
        return JSONResponse({"status": "error", "stage": "verify", "results": results})

    return JSONResponse({"status": "ok", "results": results})

# ── Phase 9.5 — Independent Oracle Verifier ───────────────────────────────────

@router.post("/api/agents/deep-verify")
async def run_deep_verify(req: VerifyRequest):
    """
    Opt-in Phase 9.5 deep verification.
    Runs independent oracle verification on wizard + wave A-E aterms only.
    """
    from compiler.independent_oracle import run_deep_verify as _run
    from core.job_manager import create_job, run_job

    job_id = create_job(f"Independent Oracle Verify — {req.vertical}")

    def _job():
        result = _run(req.vertical)
        return result

    run_job(job_id, _job)
    return JSONResponse({"job_id": job_id, "status": "pending"})


@router.get("/api/agents/deep-verify/{vertical}")
async def get_deep_verify_results(vertical: str):
    """
    Load saved Phase 9.5 results for a vertical from disk.
    Returns empty if never run.
    """
    import core.paths as _p
    out_path = _p.DATA_DIR / f"independent_oracle_{vertical}.json"
    if not out_path.exists():
        return JSONResponse({
            "vertical":      vertical,
            "total_checked": 0,
            "tally":         {},
            "mismatches":    [],
            "warnings":      [],
            "results":       [],
            "run_at":        None,
        })
    import json as _j
    return JSONResponse(_j.loads(out_path.read_text(encoding="utf-8")))


@router.get("/api/db/oracle-review")
async def db_oracle_review(vertical: str = ""):
    """
    Oracle Review subtab — returns all aterms that have independent verdicts.
    Includes full detail: formula, goal, GPL oracle, independent result, code.
    """
    import json as _j
    import core.paths as _p

    # Load independent oracle results if available
    results_by_cid = {}
    if vertical:
        out_path = _p.DATA_DIR / f"independent_oracle_{vertical}.json"
        if out_path.exists():
            data = _j.loads(out_path.read_text(encoding="utf-8"))
            for r in data.get("results", []):
                results_by_cid[r["canonical_id"]] = r

    # Load human decisions if saved
    decisions_path = _p.DATA_DIR / f"oracle_decisions_{vertical}.json"
    decisions = {}
    if decisions_path.exists():
        decisions = _j.loads(decisions_path.read_text(encoding="utf-8"))

    # Load aterm details for each result
    items = []
    for cid, result in results_by_cid.items():
        if result["verdict"] in ("SKIP",):
            continue
        aterm_path = _p.ATERMS_DIR / f"aterm_{cid}.json"
        formula    = ""
        source_goal= result.get("canonical_id", "")
        if aterm_path.exists():
            try:
                a = _j.loads(aterm_path.read_text(encoding="utf-8"))
                formula     = a.get("formula_line", "")
                source_goal = a.get("source_goal", "")
            except Exception:
                pass

        decision = decisions.get(cid, {})
        items.append({
            "canonical_id":       cid,
            "source_goal":        source_goal,
            "formula_line":       formula,
            "verdict":            result["verdict"],
            "gpl_oracle":         result["gpl_oracle"],
            "independent_oracle": result["independent_oracle"],
            "pct_diff":           result["pct_diff"],
            "independent_code":   result.get("independent_code", ""),
            "csv_tables_used":    result.get("csv_tables_used", []),
            "reasoning":          result.get("reasoning", ""),
            "run_at":             result.get("independent_run_at", ""),
            "human_decision":     decision.get("decision", ""),
            "human_note":         decision.get("note", ""),
            "ai_analysis":        decision.get("ai_analysis", None),
        })

    # Sort: MISMATCH first, then WARNING, then others
    order = {"MISMATCH": 0, "WARNING": 1, "ERROR": 2,
             "NEAR_MATCH": 3, "MATCH": 4, "ZERO_MATCH": 5}
    items.sort(key=lambda x: order.get(x["verdict"], 9))

    return JSONResponse({
        "vertical": vertical,
        "items":    items,
        "total":    len(items),
    })


@router.post("/api/db/oracle-decision")
async def save_oracle_decision(payload: dict):
    """
    Save a human decision (False Positive / Genuine Concern) for an aterm.
    """
    import json as _j
    import core.paths as _p

    vertical = payload.get("vertical", "")
    cid      = payload.get("canonical_id", "")
    decision = payload.get("decision", "")   # "false_positive" | "genuine_concern" | ""
    note     = payload.get("note", "")

    if not vertical or not cid:
        raise HTTPException(400, "vertical and canonical_id required")

    decisions_path = _p.DATA_DIR / f"oracle_decisions_{vertical}.json"
    decisions = {}
    if decisions_path.exists():
        decisions = _j.loads(decisions_path.read_text(encoding="utf-8"))

    decisions[cid] = {"decision": decision, "note": note}
    decisions_path.write_text(_j.dumps(decisions, indent=2, ensure_ascii=False))

    return JSONResponse({"status": "ok", "canonical_id": cid, "decision": decision})


@router.post("/api/db/oracle-analysis")
async def run_oracle_analysis(payload: dict):
    """
    Run AI analysis on a single mismatch — the 'Get AI Analysis' button.
    Sends goal, formula, both results, and mock data to a fresh LLM.
    """
    import json as _j
    import core.paths as _p
    from compiler.independent_oracle import analyse_mismatch

    vertical = payload.get("vertical", "")
    cid      = payload.get("canonical_id", "")

    if not vertical or not cid:
        raise HTTPException(400, "vertical and canonical_id required")

    # Load the result from the independent oracle results file
    out_path = _p.DATA_DIR / f"independent_oracle_{vertical}.json"
    if not out_path.exists():
        raise HTTPException(404, f"No deep verify results for {vertical}")

    data    = _j.loads(out_path.read_text(encoding="utf-8"))
    results = {r["canonical_id"]: r for r in data.get("results", [])}

    if cid not in results:
        raise HTTPException(404, f"No result found for {cid}")

    result   = results[cid]
    analysis = analyse_mismatch(result, vertical)

    # Save analysis back into decisions file
    decisions_path = _p.DATA_DIR / f"oracle_decisions_{vertical}.json"
    decisions = {}
    if decisions_path.exists():
        decisions = _j.loads(decisions_path.read_text(encoding="utf-8"))
    if cid not in decisions:
        decisions[cid] = {}
    decisions[cid]["ai_analysis"] = analysis
    decisions_path.write_text(_j.dumps(decisions, indent=2, ensure_ascii=False))

    return JSONResponse({"status": "ok", "analysis": analysis})


# ══════════════════════════════════════════════════════════════════════════════
# Phase WA — Workflow Analyzer
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/api/wa/catalogue")
async def wa_catalogue():
    """Return the model catalogue with recommendations and key status."""
    from compiler.workflow_analyzer import MODEL_CATALOGUE, load_llm_keys
    keys = load_llm_keys()
    catalogue = {}
    for key, info in MODEL_CATALOGUE.items():
        catalogue[key] = {
            **info,
            "has_key": info["provider"] == "anthropic" or info["provider"] in keys,
        }
    return JSONResponse({"models": catalogue})


@router.get("/api/wa/keys")
async def wa_get_keys():
    """Return which non-Claude providers have a saved API key (not the keys themselves)."""
    from compiler.workflow_analyzer import load_llm_keys
    keys = load_llm_keys()
    return JSONResponse({"providers_with_keys": list(keys.keys())})


@router.post("/api/wa/keys")
async def wa_save_key(payload: dict):
    """Save an API key for a non-Claude model provider."""
    from compiler.workflow_analyzer import load_llm_keys, save_llm_keys
    provider = payload.get("provider", "")
    api_key  = payload.get("api_key", "").strip()
    if not provider or not api_key:
        raise HTTPException(400, "provider and api_key required")
    keys = load_llm_keys()
    keys[provider] = api_key
    save_llm_keys(keys)
    return JSONResponse({"status": "ok", "provider": provider})


@router.delete("/api/wa/keys/{provider}")
async def wa_delete_key(provider: str):
    """Delete a stored API key for a provider."""
    from compiler.workflow_analyzer import delete_llm_key
    delete_llm_key(provider)
    return JSONResponse({"status": "ok", "provider": provider})


@router.get("/api/wa/aterms/{vertical}")
async def wa_list_aterms(
    vertical: str,
    filter_by: str = "all",
    wave: str = "all",
):
    """
    List aterms for a vertical to support sample selection in the UI.
    filter_by: all | locked | unlocked
    wave: all | specific wave number/letter
    """
    from compiler.workflow_analyzer import list_aterms_for_vertical
    aterms = list_aterms_for_vertical(
        vertical=vertical,
        filter_by=filter_by,
        wave=wave if wave != "all" else None,
    )
    return JSONResponse({"vertical": vertical, "aterms": aterms, "total": len(aterms)})


@router.post("/api/wa/analyze")
async def wa_run_analysis(payload: dict):
    """
    Run workflow analysis on a sample of aterms.
    Runs as a background job — returns job_id immediately.
    Body: { vertical, canonical_ids: [...], model_keys: [...] }
    """
    from compiler.workflow_analyzer import analyze_sample
    from core.job_manager import create_job, run_job

    vertical      = payload.get("vertical", "")
    canonical_ids = payload.get("canonical_ids", [])
    model_keys    = payload.get("model_keys", ["claude"])

    if not vertical:
        raise HTTPException(400, "vertical required")
    if not canonical_ids:
        raise HTTPException(400, "canonical_ids required — select at least one aterm")

    job_id = create_job(
        f"Workflow Analyzer — {vertical} ({len(canonical_ids)} aterms, {len(model_keys)} model(s))"
    )

    def _job():
        result = analyze_sample(vertical, canonical_ids, model_keys)
        return result

    run_job(job_id, _job)
    return JSONResponse({
        "job_id": job_id,
        "status": "pending",
        "total":  len(canonical_ids),
        "models": model_keys,
    })


@router.get("/api/wa/results/{vertical}")
async def wa_get_results(vertical: str):
    """
    Load the latest workflow analysis report for a vertical.
    Merges in saved human review decisions.
    """
    from compiler.workflow_analyzer import load_latest_report, load_decisions

    report = load_latest_report(vertical)
    if not report:
        return JSONResponse({
            "vertical": vertical,
            "total":    0,
            "passed":   0,
            "flagged":  0,
            "failed":   0,
            "errors":   0,
            "results":  [],
            "run_at":   None,
        })

    decisions = load_decisions(vertical)
    for r in report.get("results", []):
        cid = r.get("canonical_id", "")
        if cid in decisions:
            d = decisions[cid]
            r["human_review_status"] = d.get("status", "pending")
            r["reviewed_by"]         = d.get("reviewed_by")
            r["reviewed_at"]         = d.get("reviewed_at")
            r["review_note"]         = d.get("note", "")

    return JSONResponse(report)


@router.post("/api/wa/decision")
async def wa_save_decision(payload: dict):
    """
    Save a human review decision for a flagged aterm.
    Body: { vertical, canonical_id, status, reviewed_by, note }
    """
    from compiler.workflow_analyzer import save_decision

    vertical    = payload.get("vertical", "")
    cid         = payload.get("canonical_id", "")
    status      = payload.get("status", "")
    reviewed_by = payload.get("reviewed_by", "factory_user")
    note        = payload.get("note", "")

    if not all([vertical, cid, status]):
        raise HTTPException(400, "vertical, canonical_id and status required")

    try:
        decision = save_decision(vertical, cid, status, reviewed_by, note)
        return JSONResponse({"status": "ok", "canonical_id": cid, "decision": decision})
    except ValueError as e:
        raise HTTPException(400, str(e))



@router.get("/api/wa/auto-select/{vertical}")
async def wa_auto_select(vertical: str):
    """
    Automatically select complex aterms for workflow analysis.
    Returns all aterms matching: kind==composed OR wave in [A,B,C,D,E]
    No human selection required — the system determines complexity.
    """
    from compiler.workflow_analyzer import auto_select_complex_aterms
    result = auto_select_complex_aterms(vertical)
    return JSONResponse(result)

@router.get("/api/wa/enabled-checks")
async def wa_get_enabled_checks():
    """Return the current checklist configuration — which checks are active."""
    from compiler.workflow_analyzer import ENABLED_CHECKS
    return JSONResponse({"checks": ENABLED_CHECKS})


# ══════════════════════════════════════════════════════════════════════════════
# Presentation Compiler — Factory Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/api/ptems")
async def factory_list_ptems():
    """
    GET /api/ptems
    Returns all report templates in the factory catalog, enriched with
    oracle coverage against the factory's canonical_index.
    """
    try:
        from compiler.ptem_compiler_service import list_ptems
        ptems = list_ptems()
        return JSONResponse({"total": len(ptems), "ptems": ptems})
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"[/api/ptems] Failed: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to load ptem catalog: {e}")




@router.get("/api/ptems/custom")
async def factory_list_custom_jobs(customer_id: Optional[str] = None):
    """
    GET /api/ptems/custom?customer_id=xxx
    List all custom ptem compile jobs, optionally filtered by customer.
    """
    from compiler.ptem_compiler_service import list_custom_jobs
    jobs = list_custom_jobs(customer_id=customer_id)
    safe = [{k: v for k, v in j.items() if k != "ptem"} for j in jobs]
    return JSONResponse({"total": len(safe), "jobs": safe})


@router.get("/api/ptems/custom/{job_id}")
async def factory_get_custom_job(job_id: str):
    """
    GET /api/ptems/custom/{job_id}
    Poll status of a custom ptem compile job.
    Returns: { job_id, status, customer_id, request, error, created_at, completed_at }
    status: pending | compiling | shipping | complete | failed
    """
    from compiler.ptem_compiler_service import get_custom_job
    job = get_custom_job(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found")
    # Don't expose full ptem in status response
    safe = {k: v for k, v in job.items() if k != "ptem"}
    return JSONResponse(safe)


@router.get("/api/ptems/{canonical_id}")
async def factory_get_ptem(canonical_id: str):
    """
    GET /api/ptems/{canonical_id}
    Returns the FULL ptem blueprint for a specific template.
    Called by the runtime when a customer clicks Install.
    Includes: sections, required_oracles, narrative config, delivery config.
    """
    try:
        from compiler.ptem_compiler_service import get_ptem
        ptem = get_ptem(canonical_id)
        if not ptem:
            raise HTTPException(404, f"Ptem '{canonical_id}' not found in catalog")
        return JSONResponse(ptem)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to load ptem: {e}")



@router.post("/api/ptems/generate")
async def factory_generate_ptems(payload: dict):
    """
    POST /api/ptems/generate
    Ask LLM to generate N complete ptem definitions for a vertical,
    then write them directly to data/ptems/ptem_catalog.json.

    Generates ONE ptem per LLM call to avoid token truncation on large counts.

    Body: { vertical: str, count: int }
    """
    import asyncio, json as _json, os, re
    from pathlib import Path
    from datetime import datetime, timezone
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

    vertical = payload.get("vertical", "supply_chain")
    count    = min(int(payload.get("count", 5)), 20)

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(400, "ANTHROPIC_API_KEY not set")

    catalog_path = Path(__file__).parent.parent / "data" / "ptems" / "ptem_catalog.json"
    ci_path      = Path(__file__).parent.parent / "data" / "canonical_index.json"

    # Load existing catalog
    if catalog_path.exists():
        cat_data = _json.loads(catalog_path.read_text(encoding="utf-8"))
        existing_ptems = cat_data.get("ptems", [])
    else:
        existing_ptems = []

    existing_ids    = {p["canonical_id"] for p in existing_ptems}
    existing_titles = {p["meta"]["title"].lower() for p in existing_ptems}

    # Load oracle IDs for the vertical
    if ci_path.exists():
        ci = _json.loads(ci_path.read_text(encoding="utf-8"))
        oracle_ids = [cid for cid in ci if cid.startswith(vertical + "_")][:80]
    else:
        oracle_ids = []

    oracle_list = "\n".join(f"  - {o}" for o in oracle_ids)

    # Load valid oracle IDs from canonical_index for validation
    ci_oracle_ids: set = set()
    if ci_path.exists():
        ci_data = _json.loads(ci_path.read_text(encoding="utf-8"))
        ci_oracle_ids = set(ci_data.keys())

    # Full example ptem for the prompt
    example = {
        "id": "ptem:report_exec_supply_chain_comprehensive_demo",
        "type": "Ptem", "version": 1,
        "canonical_id": "report_exec_supply_chain_comprehensive_demo",
        "slots": {"format": "report", "audience": "ceo", "scope": "supply_chain",
                  "frequency": "monthly", "depth": "comprehensive", "period": "all_time"},
        "meta": {"title": "Supply Chain Executive Report",
                 "description": "Full executive overview - 8 KPIs, 4 bar charts, monthly trend.",
                 "audience_label": "CEO / Executive / Board",
                 "verticals": [vertical],
                 "tags": ["executive", "comprehensive", "kpi"],
                 "install_count": 2400, "rating": 5},
        "required_oracles": [
            {"canonical_id": "supply_chain_shipments_count", "label": "Total Shipments", "format": "count", "required": True},
            {"canonical_id": "supply_chain_shipments_freight_cost_currency", "label": "Total Freight Cost", "format": "currency", "currency_symbol": "INR", "required": True},
        ],
        "sections": [
            {"id": "header", "type": "header", "title": "Supply Chain Executive Report", "oracles": [], "config": {"show_status_pill": True}},
            {"id": "kpis", "type": "kpi_grid", "title": "Key Metrics", "oracles": ["supply_chain_shipments_count", "supply_chain_shipments_freight_cost_currency"], "config": {"columns": 4, "show_rag_status": True}},
            {"id": "chart1", "type": "bar_chart", "title": "Cost by Carrier", "oracles": ["supply_chain_shipments_freight_cost_currency"], "config": {"max_bars": 8, "color": "blue"}},
            {"id": "narrative", "type": "narrative", "title": "Executive Summary", "oracles": [], "config": {}},
            {"id": "audit", "type": "audit_trail", "title": "Oracle sources", "oracles": [], "config": {}},
            {"id": "footer", "type": "footer", "title": "", "oracles": [], "config": {}}
        ],
        "narrative": {"enabled": True, "audience_tone": "strategic", "max_words": 200,
                      "structure": ["observation", "context", "driver", "implication", "action"],
                      "highlight_thresholds": True},
        "delivery": {"formats": ["html", "pdf"], "filename_template": "exec_supply_chain_{date}"}
    }

    def _build_prompt(already_titles: set, already_ids: set, index: int) -> str:
        """Build a single-ptem prompt, updated with all titles/ids generated so far."""
        already_str = "\n".join(f"  - {t}" for t in already_titles) if already_titles else "  (none yet)"
        audience_cycle = ["owner", "ceo", "cfo", "coo", "ops", "board"]
        freq_cycle     = ["daily", "weekly", "monthly", "adhoc"]
        suggested_aud  = audience_cycle[index % len(audience_cycle)]
        suggested_freq = freq_cycle[index % len(freq_cycle)]
        return (
            f"You are the GPL Presentation Template Compiler. Generate ONE complete, unique ptem "
            f"JSON object for the **{vertical}** vertical.\n\n"
            f"RULES:\n"
            f"1. Output ONLY a single valid JSON object - no array, no markdown fences, no explanation.\n"
            f"2. Only use oracle canonical_ids from the AVAILABLE ORACLES list below.\n"
            f"3. canonical_id and title must be unique - do NOT use any of these already-existing titles:\n"
            f"{already_str}\n"
            f"4. canonical_id pattern: report_{{audience}}_{vertical}_{{frequency}}_{{slug}}\n"
            f"5. id must be: \"ptem:{{canonical_id}}\"\n"
            f"6. sections.oracles must only contain canonical_ids that also appear in required_oracles.\n"
            f"7. Suggested audience: {suggested_aud} | Suggested frequency: {suggested_freq}\n"
            f"8. Cover a genuinely different business question from the titles listed above.\n"
            f"9. Include install_count (100-5000) and rating (3-5) in meta.\n"
            f"10. narrative audience_tone: \"strategic\" for ceo/board/owner, \"financial\" for cfo, \"operational\" for ops/coo.\n\n"
            f"AVAILABLE ORACLES for {vertical}:\n{oracle_list}\n\n"
            f"EXAMPLE PTEM STRUCTURE (follow this exactly):\n{_json.dumps(example, indent=2)}\n\n"
            f"Output the JSON object now - starting with {{ and ending with }}:"
        )

    def _call_llm_single(prompt: str) -> str:
        import urllib.request
        body = _json.dumps({
            "model": "claude-sonnet-4-6",
            "max_tokens": 4000,
            "messages": [{"role": "user", "content": prompt}]
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={"Content-Type": "application/json",
                     "x-api-key": api_key,
                     "anthropic-version": "2023-06-01"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = _json.loads(resp.read().decode("utf-8"))
        return "".join(b["text"] for b in result.get("content", []) if b.get("type") == "text")

    def _parse_single(raw: str) -> dict:
        clean = raw.strip()
        clean = re.sub(r"^```[a-z]*\n?", "", clean)
        clean = re.sub(r"\n?```$", "", clean)
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if m:
            clean = m.group(0)
        return _json.loads(clean)

    def _validate_and_stamp(ptem: dict, now: str) -> dict:
        if ci_oracle_ids:
            original = len(ptem.get("required_oracles", []))
            ptem["required_oracles"] = [
                o for o in ptem.get("required_oracles", [])
                if o.get("canonical_id", "") in ci_oracle_ids
            ]
            stripped = original - len(ptem["required_oracles"])
            if stripped:
                import logging as _log
                _log.getLogger(__name__).warning(
                    f"[ptems/generate] Stripped {stripped} invalid oracle IDs from '{ptem.get('canonical_id')}'"
                )
            for sec in ptem.get("sections", []):
                sec["oracles"] = [o for o in sec.get("oracles", []) if o in ci_oracle_ids]
        ptem.setdefault("meta", {})["created_at"] = now
        ptem["meta"]["created_by"] = "factory_llm_generator"
        return ptem

    # -- Generate ONE ptem per LLM call to avoid token truncation --
    added, skipped, errors = [], [], []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    loop = asyncio.get_event_loop()

    for i in range(count):
        prompt = _build_prompt(existing_titles, existing_ids, i)
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_call_llm_single, prompt)
                raw = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda f=future: f.result(timeout=60)),
                    timeout=65
                )
        except (asyncio.TimeoutError, FuturesTimeout):
            errors.append(f"Timeout on ptem #{i+1}")
            continue
        except Exception as e:
            errors.append(f"LLM error on ptem #{i+1}: {e}")
            continue

        try:
            ptem = _parse_single(raw)
        except Exception as e:
            errors.append(f"JSON parse error on ptem #{i+1}: {e}. Raw (first 200): {raw[:200]}")
            continue

        cid = ptem.get("canonical_id", "")
        if not cid or cid in existing_ids:
            skipped.append(cid or f"(no id on #{i+1})")
            continue

        ptem = _validate_and_stamp(ptem, now)
        existing_ptems.append(ptem)
        existing_ids.add(cid)
        existing_titles.add(ptem.get("meta", {}).get("title", "").lower())
        added.append(cid)

    # Write catalog once at the end
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(_json.dumps({
        "_description": "GPL Ptem Catalog - factory-managed report templates",
        "_version":     "1.0.0",
        "_total":       len(existing_ptems),
        "ptems":        existing_ptems,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    return JSONResponse({
        "success":       True,
        "added":         len(added),
        "skipped":       len(skipped),
        "errors":        len(errors),
        "added_ids":     added,
        "error_details": errors if errors else None,
    })




@router.post("/api/ptems/custom/start")
async def factory_start_custom_ptem(payload: dict):
    """
    POST /api/ptems/custom/start
    Start compiling a custom ptem from a customer's plain-English request.
    Returns job_id immediately — poll /api/ptems/custom/{job_id} for status.

    Body: {
      customer_id: str,
      request_text: str,
      callback_url: str,           — runtime URL to ship finished ptem
      available_oracle_ids: [...]  — optional: what this customer has
    }
    """
    from compiler.ptem_compiler_service import start_custom_compile
    customer_id          = payload.get("customer_id")
    request_text         = payload.get("request_text")
    callback_url         = payload.get("callback_url")
    available_oracle_ids = payload.get("available_oracle_ids", [])

    if not all([customer_id, request_text, callback_url]):
        raise HTTPException(400, "customer_id, request_text, and callback_url are required")

    job_id = start_custom_compile(
        customer_id=customer_id,
        request_text=request_text,
        callback_url=callback_url,
        available_oracle_ids=available_oracle_ids,
    )
    return JSONResponse({"job_id": job_id, "status": "pending"})
