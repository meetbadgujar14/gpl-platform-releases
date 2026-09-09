"""
compiler/ptem_compiler_service.py
==================================
GPL Presentation Template (Ptem) Compiler Service — Factory Side.

Three responsibilities:
  1. list_ptems()        — read ptem_catalog.json, enrich with oracle coverage
  2. compile_new_ptem()  — guided form input → LLM maps oracles → saves to catalog
  3. compile_custom()    — customer plain-English request → LLM builds ptem →
                           ships ptem JSON to runtime via callback_url

Uses the factory's ANTHROPIC_API_KEY and canonical_index to do the oracle mapping.
Never sees customer data — only canonical_ids and their metadata.
"""

import json
import logging
import uuid
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT         = Path(__file__).parent.parent
_CATALOG_PATH = _ROOT / "data" / "ptems" / "ptem_catalog.json"
_CI_PATH      = _ROOT / "data" / "canonical_index.json"

ANTHROPIC_MODEL = "claude-sonnet-4-6"

# ── In-memory store for custom compile jobs ────────────────────────────────────
# { job_id: { status, customer_id, request, ptem, error, created_at, completed_at } }
_custom_jobs: Dict[str, Dict] = {}
_jobs_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_catalog() -> List[Dict]:
    if not _CATALOG_PATH.exists():
        return []
    return json.loads(_CATALOG_PATH.read_text(encoding="utf-8")).get("ptems", [])


def _save_catalog(ptems: List[Dict]) -> None:
    _CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "_description": "GPL Ptem Catalog — factory-managed report templates",
        "_version": "1.0.0",
        "_total": len(ptems),
        "ptems": ptems,
    }
    _CATALOG_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_canonical_index() -> Dict:
    if not _CI_PATH.exists():
        return {}
    return json.loads(_CI_PATH.read_text(encoding="utf-8"))


def _oracle_coverage(ptem: Dict, ci: Dict) -> Dict:
    """Compute oracle coverage for a ptem against the factory's canonical_index."""
    required = ptem.get("required_oracles", [])
    available = [o for o in required if o["canonical_id"] in ci]
    missing   = [o for o in required if o["canonical_id"] not in ci]
    return {
        "total":     len(required),
        "available": len(available),
        "missing":   len(missing),
        "available_ids": [o["canonical_id"] for o in available],
        "missing_oracles": [
            {
                "canonical_id": o["canonical_id"],
                "label":        o.get("label", o["canonical_id"]),
                "required":     o.get("required", False),
                "data_requirement": _explain_oracle(o["canonical_id"]),
            }
            for o in missing
        ],
        "status": (
            "READY"           if not missing else
            "READY_PARTIAL"   if not any(o.get("required") for o in missing) else
            "PENDING_ORACLES"
        ),
    }


def _explain_oracle(canonical_id: str) -> str:
    """
    Return a human-readable explanation of what data is needed to satisfy
    a given oracle canonical_id. Hardcoded for known patterns (Option A).
    """
    cid = canonical_id.lower()

    # Pattern matching on canonical_id structure
    if "shipping_cost" in cid:
        return "Requires a 'shipping_cost' (or 'cost') numeric column in your shipments data."
    if "weight" in cid:
        return "Requires a 'weight_kg' (or 'weight') numeric column in your shipments data."
    if "carrier" in cid and "count" in cid:
        return "Requires a 'carrier' (or 'carrier_name') column in your shipments data."
    if "carrier" in cid and "cost" in cid:
        return "Requires both 'carrier' and 'shipping_cost' columns in your shipments data."
    if "destination" in cid:
        return "Requires a 'destination' (or 'dest') column in your shipments data."
    if "origin" in cid:
        return "Requires an 'origin' (or 'source') column in your shipments data."
    if "status" in cid and "shipment" in cid:
        return "Requires a 'status' (or 'shipment_status') column in your shipments data."
    if "delivered" in cid:
        return "Requires a 'status' column with value 'delivered' in your shipments data."
    if "failure" in cid or "failed" in cid:
        return "Requires a 'status' column with value 'failure' or 'failed' in your shipments data."
    if "in_transit" in cid:
        return "Requires a 'status' column with value 'in_transit' in your shipments data."
    if "revenue" in cid and "retail" in cid:
        return "Requires 'price' and 'quantity' columns in your order line items data."
    if "customer" in cid and "count" in cid:
        return "Requires a customers table (CSV with customer records)."
    if "repeat" in cid:
        return "Requires an 'orders_count' column in your customers data."
    if "city" in cid:
        return "Requires a 'default_address_city' (or 'city') column in your customers data."
    if "inventory" in cid and "safety" in cid:
        return "Requires 'quantity' and 'safety_stock' (or 'reorder_point') columns in your inventory data."
    if "inventory" in cid:
        return "Requires an inventory table (CSV with SKU quantities)."
    if "supplier" in cid:
        return "Requires a suppliers table (CSV with supplier records)."
    if "product" in cid:
        return "Requires a products table (CSV with product/SKU records)."
    if "fulfilled" in cid:
        return "Requires a 'fulfillment_status' column in your order line items data."

    # Finance / P&L oracles
    if "finance" in cid and "revenue" in cid:
        return "Requires a 'revenue' or 'total_revenue' column in a finance or orders table."
    if "cogs" in cid or "cost_of_goods" in cid:
        return "Requires a 'cogs' or 'cost_of_goods' column in a finance or orders table."
    if "gross_profit" in cid:
        return "Requires 'revenue' and 'cogs' columns to compute gross profit."
    if "net_profit" in cid:
        return "Requires a 'net_profit' column in a finance or orders table."
    if "gross_margin" in cid:
        return "Requires 'revenue' and 'cogs' columns to compute gross margin %."
    if "finance" in cid and "category" in cid:
        return "Requires a 'category' column in your finance or orders table."

    # Fallback: parse the canonical_id for a generic message
    parts = canonical_id.split("_")
    entity = parts[2] if len(parts) > 2 else "relevant"
    measure = parts[3] if len(parts) > 3 else "data"
    return f"Requires {measure.replace('_', ' ')} data in your {entity.replace('_', ' ')} table."


# ── 1. List Ptems ──────────────────────────────────────────────────────────────

def list_ptems() -> List[Dict]:
    """Return all ptems from catalog, enriched with oracle coverage."""
    catalog = _load_catalog()
    ci      = _load_canonical_index()
    result  = []
    for ptem in catalog:
        coverage = _oracle_coverage(ptem, ci)
        result.append({
            "id":            ptem["id"],
            "canonical_id":  ptem["canonical_id"],
            "type":          "Ptem",
            "version":       ptem.get("version", 1),
            "meta":          ptem["meta"],
            "slots":         ptem.get("slots", {}),
            "delivery":      ptem.get("delivery", {}),
            "coverage":      coverage,
            "sections_count": len(ptem.get("sections", [])),
            "sections":      [
                {"type": s.get("type",""), "title": s.get("title","")}
                for s in ptem.get("sections", [])
                if s.get("type") not in ("header","footer","audit_trail")
            ],
            "required_oracles": ptem.get("required_oracles", []),
            "star_rating":      ptem.get("star_rating", 3),
        })
    return result


# ── 2. Compile New Ptem (Guided Form) ─────────────────────────────────────────

def compile_new_ptem(form: Dict) -> Dict:
    """
    Take guided form input, use LLM to identify required oracle IDs from
    the canonical_index, build a complete ptem blueprint, save to catalog.

    Form fields:
      title, description, audience_label, vertical, audience, scope,
      frequency, depth, period, sections (list of section types),
      content_description (plain English of what the report should show)
    """
    import os
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set — cannot compile ptem")

    ci = _load_canonical_index()
    if not ci:
        raise ValueError("canonical_index is empty — run factory compilation first")

    # Build oracle list — full canonical_id with descriptive label
    # Group by vertical prefix so the LLM can find relevant ones easily
    vertical = form.get("vertical", "universal")
    # Prioritise oracles matching the vertical, then fill with others
    priority = [(cid, e) for cid, e in ci.items() if cid.startswith(vertical)]
    others   = [(cid, e) for cid, e in ci.items() if not cid.startswith(vertical)]
    ordered  = priority + others

    oracle_summary = "\n".join(
        f"  {cid}: {cid.replace('_', ' ')}"
        for cid, entry in ordered
    )

    content_desc = form.get('content_description', '')
    sections_req = form.get('sections', [])

    # Separate scalars from grouped for prompt clarity
    scalars_list = [(cid, e) for cid, e in ordered if "_by_" not in cid]
    grouped_list = [(cid, e) for cid, e in ordered if "_by_" in cid]

    prompt = f"""You are the GPL Presentation Template Compiler. Build a rich, multi-section business report blueprint.

REPORT SPECIFICATION:
  Title: {form.get('title')}
  Audience: {form.get('audience_label')}
  Vertical: {vertical}
  Content to show: {content_desc}

YOUR JOB:
Design a report with multiple named chart sections — one oracle per chart section.
Each chart section gets its OWN title and shows a DIFFERENT breakdown of the data.

SECTION DESIGN RULES:
1. Always start with: header → kpi_grid → [chart sections] → narrative → audit_trail → footer
2. kpi_grid: pick 4-7 scalar KPIs (counts, totals, averages, rates — no "_by_" in name)
3. Chart sections: create ONE section per breakdown dimension. Create 4-8 chart sections.
   Use DIFFERENT oracles in each. NEVER repeat the same oracle in two sections.

SECTION TYPES — choose the best fit for each oracle:
  bar_chart        → _by_category, _by_carrier, _by_supplier, _by_status breakdowns
  line_chart       → _by_month, _by_week, _by_quarter time-series oracles
  pie_chart        → _by_category distributions where you want proportions (up to 6 segments)
  donut_chart      → same as pie_chart; prefer when showing part-of-whole counts
  table            → top-N ranked breakdowns (_by_supplier_category, _by_carrier etc)
  alert_list       → 3-5 scalar KPIs that act as status indicators (reorder points, counts on hold)
  pipeline         → 3-5 scalar day/time metrics showing sequential stages (lead times, cycle times)
  donut_ring       → 2-4 scalar COUNT metrics that are sub-totals of the same thing (active/inactive/on-hold suppliers)
  funnel_chart     → 3-5 scalar counts showing a conversion or drop-off sequence
  scatter_chart    → exactly 2 scalar oracles you want to compare side-by-side

SELECTION GUIDANCE:
  - _by_month oracles → always line_chart
  - _by_category/_by_carrier → bar_chart or table
  - count scalars that are sub-groups → donut_ring
  - cycle time / lead time scalars → pipeline
  - status/reorder/safety_stock counts → alert_list

4. NEVER put multiple oracles in one bar_chart, line_chart, pie_chart, donut_chart, or table section.
5. narrative section oracles list must be EMPTY.

AVAILABLE SCALAR ORACLES (use for kpi_grid, alert_list, pipeline, donut_ring, funnel_chart, scatter_chart):
{chr(10).join(f"  {cid}" for cid, _ in scalars_list)}

AVAILABLE GROUPED ORACLES (use for bar_chart, line_chart, pie_chart, donut_chart, table — one per section):
{chr(10).join(f"  {cid}" for cid, _ in grouped_list)}

Respond ONLY with a valid JSON object:
{{
  "required_oracles": [
    {{
      "canonical_id": "exact_id_from_list_above",
      "label": "Human readable label (e.g. Revenue by Month)",
      "format": "currency|count|percent|days|number",
      "currency_symbol": "₹",
      "decimal_places": 0,
      "required": true|false
    }}
  ],
  "sections": [
    {{"type": "header",      "title": "{form.get('title')}", "oracles": []}},
    {{"type": "kpi_grid",    "title": "Key Metrics",         "oracles": ["oracle_id_1", "oracle_id_2"]}},
    {{"type": "bar_chart",   "title": "Revenue by Month",    "oracles": ["one_grouped_oracle_id"]}},
    {{"type": "bar_chart",   "title": "Shipments by Carrier","oracles": ["another_grouped_oracle_id"]}},
    {{"type": "narrative",   "title": "Business Insights",   "oracles": []}},
    {{"type": "audit_trail", "title": "Oracle Sources",      "oracles": []}},
    {{"type": "footer",      "title": "",                    "oracles": []}}
  ],
  "reasoning": "One sentence explaining what this report covers"
}}

CRITICAL: Every oracle listed in any section's oracles array must also appear in required_oracles."""

    response = _llm_call(prompt, api_key, max_tokens=1500)

    try:
        # Extract outermost JSON object — LLM sometimes appends trailing text
        _start = response.find("{")
        _end   = response.rfind("}") + 1
        if _start == -1 or _end == 0:
            raise ValueError("LLM response contained no JSON object")
        clean  = response[_start:_end]
        parsed = json.loads(clean)
    except Exception as e:
        raise ValueError(f"LLM returned invalid JSON: {e}\nResponse: {response[:300]}")

    required_oracles = parsed.get("required_oracles", [])
    if not required_oracles:
        raise ValueError("LLM selected no oracle values — refine the content description")

    # Validate all selected IDs exist in canonical_index
    valid_oracles = [o for o in required_oracles if o["canonical_id"] in ci]
    if not valid_oracles:
        raise ValueError("LLM selected oracle IDs not found in canonical_index")

    # Build sections from selected oracles + form sections
    sections = _build_sections_from_form(form, valid_oracles)

    # Generate canonical_id from slots
    audience = form.get("audience", "owner")
    scope    = form.get("scope", "universal")
    freq     = form.get("frequency", "adhoc")
    depth    = form.get("depth", "standard")
    period   = form.get("period", "all_time")
    slug     = form.get("title", "custom").lower().replace(" ", "_")[:20]
    canonical_id = f"report_{audience}_{scope}_{freq}_{depth}_{period}_{slug}"

    ptem = {
        "id":           f"ptem:{canonical_id}",
        "type":         "Ptem",
        "version":      1,
        "canonical_id": canonical_id,
        "slots": {
            "format":    "report",
            "audience":  audience,
            "scope":     scope,
            "frequency": freq,
            "depth":     depth,
            "period":    period,
        },
        "meta": {
            "title":          form.get("title"),
            "description":    form.get("description"),
            "audience_label": form.get("audience_label"),
            "verticals":      [form.get("vertical", "universal")],
            "tags":           form.get("tags", []),
            "created_at":     _now(),
            "created_by":     "factory_guided_form",
            "llm_reasoning":  parsed.get("reasoning", ""),
        },
        "required_oracles": valid_oracles,
        "sections":         sections,
        "narrative": {
            "enabled":          True,
            "audience_tone":    _tone_for_audience(audience),
            "max_words":        150,
            "structure":        ["observation", "context", "driver", "implication", "action"],
            "highlight_thresholds": True,
        },
        "delivery": {
            "formats":           ["html", "pdf"],
            "filename_template": f"{slug}_{{date}}",
        },
    }

    # Append to catalog
    catalog = _load_catalog()
    # Remove any existing ptem with same canonical_id
    catalog = [p for p in catalog if p["canonical_id"] != canonical_id]
    catalog.append(ptem)
    _save_catalog(catalog)

    log.info("New ptem compiled and saved: %s (%d oracles)", canonical_id, len(valid_oracles))

    return {
        "success":      True,
        "canonical_id": canonical_id,
        "title":        form.get("title"),
        "oracles_selected": len(valid_oracles),
        "reasoning":    parsed.get("reasoning", ""),
        "ptem":         ptem,
    }


# ── 3. Compile Custom Ptem (Customer Request) ──────────────────────────────────

def start_custom_compile(
    customer_id: str,
    request_text: str,
    callback_url: str,
    available_oracle_ids: Optional[List[str]] = None,
) -> str:
    """
    Start a background job to compile a custom ptem from a customer's
    plain-English request. Ships the finished ptem back to the runtime
    via callback_url.

    Returns job_id immediately (caller polls for status).
    """
    job_id = str(uuid.uuid4())[:8]
    with _jobs_lock:
        _custom_jobs[job_id] = {
            "job_id":       job_id,
            "status":       "pending",
            "customer_id":  customer_id,
            "request":      request_text,
            "callback_url": callback_url,
            "ptem":         None,
            "error":        None,
            "created_at":   _now(),
            "completed_at": None,
        }

    thread = threading.Thread(
        target=_run_custom_compile,
        args=(job_id, customer_id, request_text, callback_url, available_oracle_ids or []),
        daemon=True,
    )
    thread.start()
    log.info("[ptem_compiler] Custom compile started: job=%s customer=%s", job_id, customer_id)
    return job_id


def get_custom_job(job_id: str) -> Optional[Dict]:
    with _jobs_lock:
        return _custom_jobs.get(job_id)


def list_custom_jobs(customer_id: Optional[str] = None) -> List[Dict]:
    with _jobs_lock:
        jobs = list(_custom_jobs.values())
    if customer_id:
        jobs = [j for j in jobs if j["customer_id"] == customer_id]
    return sorted(jobs, key=lambda x: x["created_at"], reverse=True)


def _run_custom_compile(
    job_id: str,
    customer_id: str,
    request_text: str,
    callback_url: str,
    available_oracle_ids: List[str],
) -> None:
    """Background thread: compile custom ptem → ship to runtime."""
    import os

    def _update(status: str, **kwargs):
        with _jobs_lock:
            _custom_jobs[job_id]["status"] = status
            _custom_jobs[job_id].update(kwargs)

    try:
        _update("compiling")
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not configured on factory")

        ci = _load_canonical_index()

        # If customer sent their available oracle IDs, filter to those only
        # (Option B: factory builds generically from available oracle pool)
        if available_oracle_ids:
            working_ci = {k: v for k, v in ci.items() if k in available_oracle_ids}
        else:
            working_ci = ci

        if not working_ci:
            raise ValueError("No oracle values available to build this report")

        # Build oracle summary with clear readable names
        # Scalars first (kpi_grid candidates), then grouped/by_X (chart candidates)
        scalars  = [(cid, e) for cid, e in working_ci.items() if "_by_" not in cid]
        grouped  = [(cid, e) for cid, e in working_ci.items() if "_by_" in cid]
        ordered  = scalars + grouped


        prompt = f"""You are the GPL Presentation Template Compiler. Build a rich, multi-section custom report for a customer.

CUSTOMER REQUEST: "{request_text}"

YOUR JOB:
Design a report that fully answers the customer request using multiple named chart sections.
Each chart section gets its OWN title and shows a DIFFERENT breakdown of the data.

SECTION DESIGN RULES:
1. Structure: header → kpi_grid → [chart sections] → narrative → audit_trail → footer
2. kpi_grid: pick 4-7 scalar KPIs that directly answer the customer request
3. Chart sections: create ONE section per breakdown dimension. Examples:
   - "Procurement by Month" → use a _by_month oracle
   - "Spend by Region" → use a _by_region oracle
   - "Shipments by Carrier" → use a _by_carrier oracle
   - "Inventory by Category" → use a _by_item_category oracle
   Create 4-8 chart sections. More relevant breakdowns = richer report.
4. NEVER put the same oracle in two sections.
5. NEVER put multiple oracles in one chart section — exactly ONE oracle per chart section.
6. narrative section oracles list must be EMPTY.
7. Only use IDs from the lists below — exactly as written.

AVAILABLE SCALAR ORACLES (for kpi_grid):
{chr(10).join(f"  {cid}" for cid, _ in scalars)}

AVAILABLE GROUPED ORACLES (one per chart section):
{chr(10).join(f"  {cid}" for cid, _ in grouped)}

Respond ONLY with a valid JSON object:
{{
  "title": "Short descriptive report title",
  "description": "One sentence describing what this report shows",
  "audience": "owner|ops|cfo|coo",
  "scope": "supply_chain|logistics|retail|universal",
  "required_oracles": [
    {{
      "canonical_id": "exact_id_from_list_above",
      "label": "Human readable label",
      "format": "currency|count|percent|days|number",
      "currency_symbol": "₹",
      "decimal_places": 0,
      "required": true|false
    }}
  ],
  "sections": [
    {{"type": "header",      "title": "Your Report Title",   "oracles": []}},
    {{"type": "kpi_grid",    "title": "Key Metrics",         "oracles": ["scalar_oracle_1", "scalar_oracle_2"]}},
    {{"type": "bar_chart",   "title": "Spend by Month",      "oracles": ["one_grouped_oracle"]}},
    {{"type": "bar_chart",   "title": "Volume by Carrier",   "oracles": ["another_grouped_oracle"]}},
    {{"type": "narrative",   "title": "Business Insights",   "oracles": []}},
    {{"type": "audit_trail", "title": "Oracle Sources",      "oracles": []}},
    {{"type": "footer",      "title": "",                    "oracles": []}}
  ]
}}

CRITICAL: Every oracle in sections.oracles must also appear in required_oracles."""

        response = _llm_call(prompt, api_key, max_tokens=3000)
        # Extract outermost JSON object — LLM sometimes appends trailing text
        _start = response.find("{")
        _end   = response.rfind("}") + 1
        if _start == -1 or _end == 0:
            raise ValueError("LLM response contained no JSON object")
        clean  = response[_start:_end]
        parsed = json.loads(clean)

        valid_oracles = [
            o for o in parsed.get("required_oracles", [])
            if o["canonical_id"] in working_ci
        ]
        if not valid_oracles:
            raise ValueError("Could not identify matching oracle values for your request")

        # Build ptem
        slug         = parsed.get("title", "custom").lower().replace(" ", "_")[:20]
        canonical_id = f"custom_{customer_id}_{job_id}_{slug}"
        sections     = _build_sections_from_form(parsed, valid_oracles)

        ptem = {
            "id":           f"ptem:{canonical_id}",
            "type":         "Ptem",
            "version":      1,
            "canonical_id": canonical_id,
            "slots": {
                "format":    "report",
                "audience":  parsed.get("audience", "owner"),
                "scope":     parsed.get("scope", "universal"),
                "frequency": "adhoc",
                "depth":     "standard",
                "period":    "all_time",
            },
            "meta": {
                "title":          parsed.get("title", "Custom Report"),
                "description":    parsed.get("description", request_text[:100]),
                "audience_label": "Custom Report",
                "verticals":      [parsed.get("scope", "universal")],
                "tags":           ["custom", customer_id],
                "created_at":     _now(),
                "created_by":     f"customer_request:{customer_id}",
                "original_request": request_text,
            },
            "required_oracles": valid_oracles,
            "sections":         sections,
            "narrative": {
                "enabled":             True,
                "audience_tone":       _tone_for_audience(parsed.get("audience", "owner")),
                "max_words":           150,
                "structure":           ["observation", "driver", "implication", "action"],
                "highlight_thresholds": True,
            },
            "delivery": {
                "formats":           ["html", "pdf"],
                "filename_template": f"custom_{slug}_{{date}}",
            },
        }

        _update("shipping", ptem=ptem)

        # Ship ptem JSON to runtime
        _ship_ptem_to_runtime(ptem, customer_id, job_id, callback_url)

        _update("complete", completed_at=_now())
        log.info("[ptem_compiler] Custom compile complete: job=%s → %s", job_id, canonical_id)

    except Exception as e:
        log.error("[ptem_compiler] Custom compile failed: job=%s error=%s", job_id, e)
        _update("failed", error=str(e), completed_at=_now())


def _ship_ptem_to_runtime(ptem: Dict, customer_id: str, job_id: str, callback_url: str) -> None:
    """POST the ptem JSON to the runtime's receive endpoint."""
    import httpx
    ptem_bytes = json.dumps(ptem, indent=2, ensure_ascii=False).encode("utf-8")
    resp = httpx.post(
        callback_url,
        content=ptem_bytes,
        headers={
            "Content-Type":    "application/json",
            "X-Ptem-Job-Id":   job_id,
            "X-Customer-Id":   customer_id,
            "X-Package-Type":  "custom_ptem",
            "X-Canonical-Id":  ptem["canonical_id"],
        },
        timeout=30,
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Runtime rejected ptem: HTTP {resp.status_code} — {resp.text[:200]}")
    log.info("[ptem_compiler] Ptem shipped to runtime: %s", callback_url)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _llm_call(prompt: str, api_key: str, max_tokens: int = 1000) -> str:
    """Call Anthropic API with a hard 45-second wall-clock timeout."""
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

    def _do_call() -> str:
        body = json.dumps({
            "model":      ANTHROPIC_MODEL,
            "max_tokens": max_tokens,
            "messages":   [{"role": "user", "content": prompt}],
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "Content-Type":      "application/json",
                "x-api-key":         api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=40) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        return " ".join(b["text"] for b in result.get("content", []) if b.get("type") == "text")

    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_do_call)
        try:
            return future.result(timeout=45)
        except FuturesTimeout:
            future.cancel()
            raise ValueError("LLM call timed out after 45 seconds")


def _slots_to_label(slots: Dict) -> str:
    parts = [slots.get("entity", ""), slots.get("measure", ""), slots.get("unit", ""), slots.get("time", "")]
    return " ".join(p for p in parts if p and p not in ("all", "all_time", "total", "scalar")).strip() or "metric"


def _tone_for_audience(audience: str) -> str:
    return {
        "cfo": "financial", "board": "financial", "investor": "financial",
        "ops": "operational", "coo": "operational",
        "ceo": "strategic", "owner": "strategic",
    }.get(audience, "strategic")


def _build_sections_from_form(form: Dict, valid_oracles: List[Dict]) -> List[Dict]:
    """Build sections list from form data and selected oracles."""
    requested = form.get("sections", ["header", "kpi_grid", "bar_chart", "narrative", "audit_trail", "footer"])

    # Group oracles by section
    by_section: Dict[str, List[str]] = {}
    for o in valid_oracles:
        sec = o.get("section", "kpi_grid")
        by_section.setdefault(sec, []).append(o["canonical_id"])

    sections = []
    for sec_type in requested:
        sec: Dict = {"id": sec_type, "type": sec_type, "title": _section_title(sec_type), "oracles": [], "config": {}}

        if sec_type == "header":
            sec["config"] = {"show_status_pill": True}
        elif sec_type == "kpi_grid":
            sec["oracles"] = by_section.get("kpi_grid", [cid for o in valid_oracles[:4] for cid in [o["canonical_id"]]])
            sec["config"]  = {"columns": min(len(sec["oracles"]), 4)}
        elif sec_type == "table":
            sec["oracles"] = by_section.get("table", [])
            sec["config"]  = {"max_rows": 10, "columns": ["Name", "Value"]}
        elif sec_type == "bar_chart":
            sec["oracles"] = by_section.get("bar_chart", [cid for o in valid_oracles if "grouped" in o["canonical_id"] for cid in [o["canonical_id"]]][:1])
            sec["config"]  = {"max_bars": 8, "color": "blue"}
        elif sec_type == "narrative":
            sec["config"]  = {}
        elif sec_type == "audit_trail":
            sec["title"]   = "Oracle sources"
            sec["config"]  = {}
        elif sec_type == "footer":
            sec["title"]   = ""
            sec["config"]  = {}

        sections.append(sec)

    return sections


def _section_title(sec_type: str) -> str:
    return {
        "header":      "",
        "kpi_grid":    "Key metrics",
        "table":       "Breakdown",
        "bar_chart":   "Analysis",
        "city_breakdown": "Geographic breakdown",
        "narrative":   "Business narrative",
        "audit_trail": "Oracle sources",
        "footer":      "",
    }.get(sec_type, sec_type.replace("_", " ").title())


# ── 4. Get Single Ptem (Full Blueprint) ───────────────────────────────────────

def get_ptem(canonical_id: str) -> Optional[Dict]:
    """
    Return the full ptem blueprint for a single canonical_id.
    Includes sections, required_oracles, narrative config — everything
    the runtime needs to execute the report.
    Also enriches with oracle coverage from the factory canonical_index.
    Returns None if not found.
    """
    catalog = _load_catalog()
    ci      = _load_canonical_index()

    ptem = next((p for p in catalog if p["canonical_id"] == canonical_id), None)
    if not ptem:
        return None

    coverage = _oracle_coverage(ptem, ci)

    return {
        "id":               ptem["id"],
        "canonical_id":     ptem["canonical_id"],
        "type":             "Ptem",
        "version":          ptem.get("version", 1),
        "meta":             ptem["meta"],
        "slots":            ptem.get("slots", {}),
        "required_oracles": ptem.get("required_oracles", []),
        "sections":         ptem.get("sections", []),
        "narrative":        ptem.get("narrative", {}),
        "delivery":         ptem.get("delivery", {}),
        "coverage":         coverage,
    }
