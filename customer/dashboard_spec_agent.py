"""
customer/dashboard_spec_agent.py
=================================
AI agent that generates a dashboard.json specification for a given business context.

Called automatically after a table is ingested and its context is detected.

Output: customer_runtime/customers/{cid}/contexts/{context_id}/dashboard.json

The spec is generic — no domain-specific logic in the runtime.
The Factory (this agent) decides everything; the runtime just executes it.
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ── Column role taxonomy ──────────────────────────────────────────────────────

ROLE_IDENTIFIER = "identifier"
ROLE_DIMENSION  = "dimension"
ROLE_MEASURE    = "measure"
ROLE_DATE       = "date"
ROLE_STATUS     = "status"

# Deterministic role hints (keyword matching — used before AI)
_DATE_KEYWORDS     = {"_date", "_time", "_at", "_on", "created_", "updated_", "submitted_", "timestamp", "booked_", "planned_", "actual_", "record_", "event_", "trip_", "order_", "snapshot_", "hire_", "opened_", "commissioned_", "declaration_", "incident_"}
_STATUS_KEYWORDS   = {"status", "state", "stage", "flag", "is_", "type", "category", "tier", "mode", "level"}
_ID_KEYWORDS       = {"id", "code", "ref", "number", "num", "_no", "key", "uuid", "sku"}
_MEASURE_TYPES     = {"integer", "float", "numeric", "number"}
_DIMENSION_TYPES   = {"string", "text"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json(text: str) -> Optional[Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        # Try array
        match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except Exception:
        return None


# ── Deterministic column classifier ──────────────────────────────────────────

def _classify_column(name: str, col_type: str) -> str:
    """Fast rule-based column role classification."""
    nl = name.lower()

    # Date signals -- match as suffix/prefix to avoid false positives like "exception_category"
    if col_type in ("date", "datetime") or any(kw in nl for kw in _DATE_KEYWORDS):
        return ROLE_DATE

    # ID signals
    if any(nl.endswith(kw) or nl.startswith(kw) for kw in _ID_KEYWORDS):
        return ROLE_IDENTIFIER

    # Status / categorical signals
    if nl.startswith("is_") or any(kw in nl for kw in _STATUS_KEYWORDS):
        return ROLE_STATUS

    # Numeric → measure
    if col_type in _MEASURE_TYPES:
        return ROLE_MEASURE

    # String → dimension
    return ROLE_DIMENSION


# ── KPI candidate generation ──────────────────────────────────────────────────

def _generate_kpi_candidates(
    tables: Dict[str, Dict],
) -> List[Dict]:
    """
    Generate candidate KPIs deterministically from classified columns.
    Returns a list of KPI spec dicts (un-scored).
    """
    candidates = []

    for table_name, table_info in tables.items():
        columns   = table_info.get("columns", [])
        col_roles = table_info.get("col_roles", {})

        id_cols     = [c["name"] for c in columns if col_roles.get(c["name"]) == ROLE_IDENTIFIER]
        measure_cols = [c["name"] for c in columns if col_roles.get(c["name"]) == ROLE_MEASURE]
        status_cols  = [c["name"] for c in columns if col_roles.get(c["name"]) == ROLE_STATUS]

        # COUNT(identifier) — total records
        if id_cols:
            pk = id_cols[0]
            candidates.append({
                "id":          f"total_{table_name}",
                "title":       f"Total {_humanize(table_name)}",
                "type":        "simple",
                "table":       table_name,
                "column":      pk,
                "aggregation": "count",
                "format":      "number",
                "score":       10,
            })

        # SUM / AVG for each measure column
        for col in measure_cols[:4]:  # cap at 4 measures per table
            candidates.append({
                "id":          f"total_{col}_{table_name}",
                "title":       f"Total {_humanize(col)}",
                "type":        "simple",
                "table":       table_name,
                "column":      col,
                "aggregation": "sum",
                "format":      _infer_format(col),
                "score":       7,
            })
            candidates.append({
                "id":          f"avg_{col}_{table_name}",
                "title":       f"Avg {_humanize(col)}",
                "type":        "simple",
                "table":       table_name,
                "column":      col,
                "aggregation": "avg",
                "format":      _infer_format(col),
                "score":       5,
            })

        # COUNT where status = <value> → ratio KPIs
        for scol in status_cols[:2]:
            enums = table_info.get("enums", {}).get(scol, [])
            if id_cols and enums:
                for val in enums[:3]:
                    slug = re.sub(r"[^a-z0-9]", "_", val.lower())
                    candidates.append({
                        "id":    f"{slug}_{table_name}",
                        "title": f"{_humanize(val)} {_humanize(table_name)}",
                        "type":  "simple",
                        "table": table_name,
                        "column":      id_cols[0],
                        "aggregation": "count_where",
                        "filter_column": scol,
                        "filter_value":  val,
                        "format": "number",
                        "score":  6,
                    })

                # Delivery-rate style ratio
                if id_cols and len(enums) >= 2:
                    main_val = enums[0]
                    slug = re.sub(r"[^a-z0-9]", "_", main_val.lower())
                    candidates.append({
                        "id":    f"{slug}_rate_{table_name}",
                        "title": f"{_humanize(main_val)} Rate",
                        "type":  "ratio",
                        "table": table_name,
                        "numerator": {
                            "column":      id_cols[0],
                            "aggregation": "count_where",
                            "filter_column": scol,
                            "filter_value":  main_val,
                        },
                        "denominator": {
                            "column":      id_cols[0],
                            "aggregation": "count",
                        },
                        "format": "percent",
                        "score":  9,
                    })

    return candidates


# ── Chart candidate generation ────────────────────────────────────────────────

def _generate_chart_candidates(
    tables: Dict[str, Dict],
) -> List[Dict]:
    """Generate chart candidates using deterministic rules."""
    candidates = []

    for table_name, table_info in tables.items():
        columns   = table_info.get("columns", [])
        col_roles = table_info.get("col_roles", {})

        date_cols    = [c["name"] for c in columns if col_roles.get(c["name"]) == ROLE_DATE]
        dim_cols     = [c["name"] for c in columns if col_roles.get(c["name"]) == ROLE_DIMENSION]
        measure_cols = [c["name"] for c in columns if col_roles.get(c["name"]) == ROLE_MEASURE]
        status_cols  = [c["name"] for c in columns if col_roles.get(c["name"]) == ROLE_STATUS]
        id_cols      = [c["name"] for c in columns if col_roles.get(c["name"]) == ROLE_IDENTIFIER]

        count_col = id_cols[0] if id_cols else (measure_cols[0] if measure_cols else None)

        # DATE + IDENTIFIER → line chart (volume over time)
        if date_cols and count_col:
            dcol = date_cols[0]
            candidates.append({
                "id":             f"{table_name}_over_time",
                "title":          f"{_humanize(table_name)} Over Time",
                "type":           "line",
                "table":          table_name,
                "dimension":      dcol,
                "dimension_type": "date",
                "measure":        count_col,
                "aggregation":    "count",
                "score":          10,
            })

        # DATE + MEASURE → line chart
        for mcol in measure_cols[:2]:
            if date_cols:
                dcol = date_cols[0]
                candidates.append({
                    "id":             f"{mcol}_{table_name}_over_time",
                    "title":          f"{_humanize(mcol)} Over Time",
                    "type":           "line",
                    "table":          table_name,
                    "dimension":      dcol,
                    "dimension_type": "date",
                    "measure":        mcol,
                    "aggregation":    "sum",
                    "score":          8,
                })

        # DIMENSION + MEASURE → bar chart
        for dcol in dim_cols[:2]:
            enums = table_info.get("enums", {}).get(dcol, [])
            cardinality = len(enums) if enums else 999
            if 2 <= cardinality <= 30:
                for mcol in measure_cols[:2]:
                    candidates.append({
                        "id":             f"{mcol}_by_{dcol}_{table_name}",
                        "title":          f"{_humanize(mcol)} by {_humanize(dcol)}",
                        "type":           "bar",
                        "table":          table_name,
                        "dimension":      dcol,
                        "dimension_type": "category",
                        "measure":        mcol,
                        "aggregation":    "sum",
                        "score":          7,
                    })

                # Volume by dimension
                if count_col:
                    candidates.append({
                        "id":             f"count_by_{dcol}_{table_name}",
                        "title":          f"{_humanize(table_name)} by {_humanize(dcol)}",
                        "type":           "bar",
                        "table":          table_name,
                        "dimension":      dcol,
                        "dimension_type": "category",
                        "measure":        count_col,
                        "aggregation":    "count",
                        "score":          7,
                    })

        # STATUS (low cardinality) → donut chart
        for scol in status_cols[:2]:
            enums = table_info.get("enums", {}).get(scol, [])
            if enums and 2 <= len(enums) <= 8 and count_col:
                candidates.append({
                    "id":             f"{scol}_distribution_{table_name}",
                    "title":          f"{_humanize(scol)} Distribution",
                    "type":           "donut",
                    "table":          table_name,
                    "dimension":      scol,
                    "dimension_type": "status",
                    "measure":        count_col,
                    "aggregation":    "count",
                    "score":          8,
                })

    return candidates


# ── Dashboard spec generator ──────────────────────────────────────────────────

def generate_dashboard_spec(
    data_dir: Path,
    context_id: str,
    context_name: str,
    table_names: List[str],
    data_store_path: Path,
) -> Optional[Dict]:
    """
    Generate the full dashboard.json for a context.

    Reads table schemas from data_store.json, classifies columns,
    generates KPI and chart candidates, uses AI to curate the final selection,
    and writes contexts/{context_id}/dashboard.json.

    Returns the spec dict, or None on failure.
    """
    # ── Load table data ───────────────────────────────────────────────────────
    if not data_store_path.exists():
        log.warning(f"[dashboard_spec_agent] data_store.json not found at {data_store_path}")
        return None

    try:
        raw = json.loads(data_store_path.read_text(encoding="utf-8"))
    except Exception as e:
        log.error(f"[dashboard_spec_agent] Failed to read data_store: {e}")
        return None

    # Build a working dict of tables in this context
    tables: Dict[str, Dict] = {}
    for tname in table_names:
        # Fuzzy match table name against data_store keys
        matched_key = _match_table_key(raw, tname)
        if not matched_key:
            continue
        rows    = raw[matched_key].get("rows", [])
        columns = _extract_columns(rows)
        enums   = _extract_enums(rows, columns)
        col_roles = {c["name"]: _classify_column(c["name"], c["type"]) for c in columns}

        tables[tname] = {
            "key":      matched_key,
            "columns":  columns,
            "col_roles": col_roles,
            "enums":    enums,
            "row_count": len(rows),
        }

    if not tables:
        log.warning(f"[dashboard_spec_agent] No tables found in data_store for context '{context_id}'")
        return None

    # ── Identify primary date field ───────────────────────────────────────────
    primary_date = _find_primary_date(tables)

    # ── Generate candidates ───────────────────────────────────────────────────
    kpi_candidates   = _generate_kpi_candidates(tables)
    chart_candidates = _generate_chart_candidates(tables)

    # ── AI full-spec generation ───────────────────────────────────────────────
    ai_result = _ai_generate_full_spec(
        tables, context_id, context_name, primary_date,
        kpi_candidates, chart_candidates,
    )
    if ai_result:
        final_kpis   = ai_result.get("kpis", [])
        final_charts = ai_result.get("charts", [])
    else:
        fallback     = _deterministic_spec(kpi_candidates, chart_candidates)
        final_kpis   = fallback["kpis"]
        final_charts = fallback["charts"]

    # ── Assemble spec ─────────────────────────────────────────────────────────
    spec = {
        "context_id":         context_id,
        "context_name":       context_name,
        "generated_at":       _now(),
        "primary_date_field": primary_date,
        "filters": [
            {
                "id":     "period",
                "type":   "date_range",
                "table":  primary_date["table"] if primary_date else None,
                "column": primary_date["column"] if primary_date else None,
            }
        ] if primary_date else [],
        "kpis":   final_kpis,
        "charts": final_charts,
    }

    # ── Write dashboard.json ──────────────────────────────────────────────────
    ctx_dir = data_dir.parent / "context" / context_id
    ctx_dir.mkdir(parents=True, exist_ok=True)
    out_path = ctx_dir / "dashboard.json"
    out_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    log.info(f"[dashboard_spec_agent] dashboard.json written to {out_path}")

    return spec


# ── AI full-spec generation ───────────────────────────────────────────────────

def _ai_generate_full_spec(tables, context_id, context_name, primary_date, kpi_candidates, chart_candidates):
    """Single AI call: professional dashboard designer sees full schemas and returns complete spec."""
    try:
        from anthropic import Anthropic
        from core.config import settings
        client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    except Exception:
        return None

    # Build rich schema description
    schema_lines = []
    all_date_cols = {}

    for tname, info in tables.items():
        date_cols    = [c["name"] for c in info["columns"] if info["col_roles"].get(c["name"]) == ROLE_DATE]
        measure_cols = [c["name"] for c in info["columns"] if info["col_roles"].get(c["name"]) == ROLE_MEASURE]
        dim_cols     = [c["name"] for c in info["columns"] if info["col_roles"].get(c["name"]) == ROLE_DIMENSION]
        status_cols  = [c["name"] for c in info["columns"] if info["col_roles"].get(c["name"]) == ROLE_STATUS]
        id_cols      = [c["name"] for c in info["columns"] if info["col_roles"].get(c["name"]) == ROLE_IDENTIFIER]
        enum_sample  = {k: v[:6] for k, v in list(info.get("enums", {}).items())[:6]}
        all_date_cols[tname] = date_cols

        rc = info["row_count"]
        line_parts = [
            "TABLE: " + tname + "  (" + str(rc) + " rows)",
            "  identifiers:  " + str(id_cols),
            "  dates:        " + str(date_cols),
            "  measures:     " + str(measure_cols),
            "  dimensions:   " + str(dim_cols),
            "  status_cols:  " + str(status_cols),
            "  enum_values:  " + str(enum_sample),
        ]
        schema_lines.append("\n".join(line_parts))

    kpi_s = json.dumps([{
        "id": k["id"], "title": k["title"], "type": k.get("type", "simple"),
        "table": k.get("table"), "column": k.get("column"),
        "aggregation": k.get("aggregation"), "format": k.get("format", "number"),
        "numerator": k.get("numerator"), "denominator": k.get("denominator"),
    } for k in kpi_candidates], indent=2)

    chart_s = json.dumps([{
        "id": c["id"], "title": c["title"], "type": c["type"],
        "table": c.get("table"), "dimension": c.get("dimension"),
        "dimension_type": c.get("dimension_type"), "measure": c.get("measure"),
        "aggregation": c.get("aggregation", "count"), "format": c.get("format", "number"),
    } for c in chart_candidates], indent=2)

    date_col_map = "\n".join(
        "  " + tname + ": " + str(dcols)
        for tname, dcols in all_date_cols.items() if dcols
    ) or "  (none)"

    prompt = (
        'You are a professional BI dashboard architect with 15 years of experience building'
        ' enterprise dashboards for logistics, supply chain, retail, and finance teams.\n\n'
        'You are designing a dashboard for the "' + context_name + '" business context.\n\n'
        '## SCREEN & LAYOUT CONSTRAINTS\n'
        'The dashboard renders at 1440×900px (standard laptop). The fixed chrome takes:\n'
        '- Top navigation bar: 52px\n'
        '- Dashboard topbar (context + period pills): 52px\n'
        '- KPI strip: 100px\n'
        '- Section gaps and padding: 40px\n'
        'AVAILABLE HEIGHT for charts: ~656px total.\n\n'
        'Chart area layout:\n'
        '- Row 1 (snapshot charts): 3-column grid, card height 340px\n'
        '- Row 2 (trend cards): 2-column grid, card height 188px\n'
        '- Total used: 340 + 16 (gap) + 20 (trend label) + 188 = 564px → fits in 656px ✓\n\n'
        'THEREFORE you MUST produce EXACTLY:\n'
        '- **3 snapshot charts** (bar/donut/pie) → fills Row 1 perfectly, no scroll\n'
        '- **2 line charts** (trend) → fills Row 2 perfectly, no scroll\n'
        'Do NOT produce more than 3 snapshot charts or more than 2 line charts.\n'
        'Choose the 3 MOST VALUABLE snapshot charts from the available data.\n\n'
        '## DASHBOARD STRUCTURE\n'
        'The dashboard has TWO views:\n\n'
        '1. **OVERVIEW** — KPI strip + exactly 3 snapshot charts (bar/donut/pie) + 2 trend cards at the bottom.\n'
        '   All content visible without scrolling on a 1440×900 screen.\n'
        '   Answers: "What is the current state of my business?"\n\n'
        '2. **TREND ANALYSIS TAB** — same 2 line charts shown full-width for focused analysis.\n'
        '   Answers: "How are my metrics moving over time?"\n\n'
        '## AVAILABLE DATA\n' + "\n\n".join(schema_lines) + "\n\n"
        '## CHART TYPE SELECTION — pick the most insightful chart for each slot:\n'
        'Categorical (2-8 unique values) + count/sum  → donut (use for status distributions)\n'
        'Categorical (8-30 values) + measure ranking  → bar   (use for names, categories)\n'
        'Date column + any measure                    → line  (TREND ANALYSIS only)\n\n'
        'For your 3 snapshot charts, pick the ones that give the most business insight:\n'
        '- Distribution of a key status field (donut) → e.g. order status, payment status\n'
        '- Ranking of a key dimension (bar) → e.g. revenue by channel, volume by supplier\n'
        '- Another high-value breakdown (donut or bar)\n\n'
        '## KPI CANDIDATES:\n'
        'Pick only the KPIs that are genuinely meaningful for this data.\n'
        'Use as many as the data warrants — minimum 3, maximum 8.\n'
        'Do NOT pad with weak or redundant KPIs just to reach a number.\n'
        'If the data only supports 4 strong KPIs, return 4.\n' + kpi_s + "\n\n"
        '## CHART CANDIDATES (pick EXACTLY 3 snapshot + 2 line from these):\n' + chart_s + "\n\n"
        '## HARD CONSTRAINTS — violating these breaks the dashboard:\n'
        '1. LINE CHARTS: "dimension" MUST be one of these exact date columns:\n'
        + date_col_map + "\n"
        '   NEVER use a name/category/status column for a line chart.\n'
        '2. SNAPSHOT CHARTS: MUST NOT use a date column as dimension.\n'
        '3. EXACTLY 3 snapshot charts and EXACTLY 2 line charts — no more, no less.\n'
        '4. All table/column/measure/dimension values MUST exactly match column names in the schemas above.\n'
        '5. ORDER in charts array: all 3 snapshot charts FIRST, then 2 line charts LAST.\n'
        '6. "id" fields must be unique snake_case slugs.\n'
        '7. KPI format: currency=money, percent=rates/ratios, decimal=decimal averages, number=counts.\n'
        '8. ratio KPI: type="ratio", fill numerator/denominator, set column=null, aggregation=null.\n'
        '9. simple KPI: type="simple", fill column+aggregation, set numerator=null, denominator=null.\n\n'
        'Return ONLY valid JSON, no markdown, no explanation:\n'
        '{"kpis":[{"id":"...","title":"...","type":"simple|ratio","table":"exact","column":"col_or_null",'
        '"aggregation":"count|sum|avg","format":"number|currency|percent|decimal","numerator":null,"denominator":null}],'
        '"charts":[{"id":"...","title":"...","type":"bar|donut|pie|line","table":"exact",'
        '"dimension":"exact_col","dimension_type":"category|status|date","measure":"exact_col",'
        '"aggregation":"count|sum|avg","format":"number|currency|percent|decimal"}]}'
    )

    try:
        resp = client.messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text if resp.content else ""
        parsed = _safe_json(text)
        if isinstance(parsed, dict) and "kpis" in parsed and "charts" in parsed:
            non_lines = [c for c in parsed["charts"] if c.get("type") != "line"]
            lines     = [c for c in parsed["charts"] if c.get("type") == "line"]
            # Enforce exactly 3 snapshot + 2 line for above-fold layout
            non_lines = non_lines[:3]
            lines     = lines[:2]
            parsed["charts"] = non_lines + lines
            log.info(
                "[dashboard_spec_agent] AI generated %d KPIs, %d charts (%d snapshot, %d line)",
                len(parsed["kpis"]), len(parsed["charts"]), len(non_lines), len(lines),
            )
            return parsed
        log.warning("[dashboard_spec_agent] AI returned unexpected shape: %s", text[:400])
    except Exception as e:
        log.warning("[dashboard_spec_agent] Full-spec AI call failed: %s", e)
    return None


def _deterministic_spec(kpi_candidates, chart_candidates):
    """Fallback: score-based selection, non-lines before lines."""
    kpis      = [_strip_score(k) for k in sorted(kpi_candidates,   key=lambda x: -x.get("score", 0))[:6]]
    non_lines = sorted([c for c in chart_candidates if c["type"] != "line"], key=lambda x: -x.get("score", 0))[:3]
    lines     = sorted([c for c in chart_candidates if c["type"] == "line"],  key=lambda x: -x.get("score", 0))[:2]
    charts    = [_strip_score(c) for c in non_lines + lines]
    return {"kpis": kpis, "charts": charts}


# ── Utilities ─────────────────────────────────────────────────────────────────

def _match_table_key(raw: Dict, table_name: str) -> Optional[str]:
    """Find the data_store key that best matches table_name."""
    tl = table_name.lower()
    # Exact match
    if table_name in raw:
        return table_name
    # Substring match
    for key in raw:
        if tl in key.lower() or key.lower() in tl:
            return key
    # Keyword overlap
    best_key, best_score = None, 0
    t_words = set(re.split(r"[_\s]+", tl))
    for key in raw:
        k_words = set(re.split(r"[_\s]+", key.lower()))
        score = len(t_words & k_words)
        if score > best_score:
            best_key, best_score = key, score
    return best_key if best_score > 0 else None


def _extract_columns(rows: List[Dict]) -> List[Dict[str, str]]:
    """Infer column names and types from row data."""
    if not rows:
        return []
    sample = rows[:50]
    columns = []
    for col in rows[0].keys():
        vals = [str(r.get(col, "")) for r in sample if r.get(col)]
        col_type = _infer_type(vals)
        columns.append({"name": col, "type": col_type})
    return columns


def _infer_type(vals: List[str]) -> str:
    if not vals:
        return "string"
    numeric = date = 0
    for v in vals[:20]:
        try:
            float(v.replace(",", ""))
            numeric += 1
            continue
        except ValueError:
            pass
        if re.match(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}", v):
            date += 1
    total = len(vals[:20])
    if date / total > 0.5:
        return "date"
    if numeric / total > 0.5:
        return "float"
    return "string"


def _extract_enums(rows: List[Dict], columns: List[Dict]) -> Dict[str, List[str]]:
    """Extract unique values for string columns (potential enums)."""
    enums = {}
    for col in columns:
        if col["type"] == "string":
            vals = list({str(r.get(col["name"], "")) for r in rows if r.get(col["name"])})
            if 2 <= len(vals) <= 30:
                enums[col["name"]] = sorted(vals)
    return enums


def _find_primary_date(tables: Dict) -> Optional[Dict]:
    """Find the most suitable date field to use for period filtering.

    Prefers event/record tables over dimension tables, and prefers columns
    whose name ends with '_date' or '_at' over broader matches.
    """
    # Priority 1: event/record tables with a column ending in _date or _at
    for table_name, info in tables.items():
        is_event = any(token in table_name for token in ("_event", "_record"))
        if not is_event:
            continue
        for col in info.get("columns", []):
            n = col["name"].lower()
            if info["col_roles"].get(col["name"]) == ROLE_DATE and (n.endswith("_date") or n.endswith("_at")):
                return {"table": table_name, "column": col["name"]}

    # Priority 2: any table with a column ending in _date or _at
    for table_name, info in tables.items():
        for col in info.get("columns", []):
            n = col["name"].lower()
            if info["col_roles"].get(col["name"]) == ROLE_DATE and (n.endswith("_date") or n.endswith("_at")):
                return {"table": table_name, "column": col["name"]}

    # Priority 3: any ROLE_DATE column (fallback)
    for table_name, info in tables.items():
        for col in info.get("columns", []):
            if info["col_roles"].get(col["name"]) == ROLE_DATE:
                return {"table": table_name, "column": col["name"]}

    return None


def _humanize(text: str) -> str:
    """
    Convert a snake_case identifier to a readable label.

    For table names following the GPL naming convention:
        {vertical}_{entity}_{source}_{type}
        e.g. logistics_carriers_MANUAL_dimension
             supply_chain_shipments_WMS_record

    Strips the vertical prefix and the trailing source+type suffix so the
    customer sees "Carriers" not "Logistics Carriers Manual Dimension".

    For plain column names (no vertical prefix) just title-cases as before.
    """
    _VERTICAL_PREFIXES = (
        "supply_chain_",
        "logistics_",
        "retail_shopify_",
        "retail_",
        "hr_",
        "human_resources_",
        "finance_",
        "sales_",
    )
    # Source system and data type tokens that appear at the end of table names
    _SUFFIX_TOKENS = {
        "manual", "erp", "wms", "crm", "api",
        "dimension", "record", "event", "state", "snapshot",
    }

    s = text.strip().lower()

    # Strip known vertical prefix
    for prefix in _VERTICAL_PREFIXES:
        if s.startswith(prefix):
            s = s[len(prefix):]
            break

    # Split into tokens and drop trailing source/type tokens
    parts = s.split("_")
    while parts and parts[-1] in _SUFFIX_TOKENS:
        parts.pop()

    if not parts:
        return re.sub(r"[_\-]+", " ", text).title()

    return " ".join(p.title() for p in parts)


def _infer_format(col_name: str) -> str:
    cl = col_name.lower()
    if any(kw in cl for kw in ("cost", "amount", "price", "revenue", "fee", "charge", "pay")):
        return "currency"
    if any(kw in cl for kw in ("rate", "ratio", "percent", "pct", "%")):
        return "percent"
    if any(kw in cl for kw in ("weight", "kg", "ton", "lb", "distance", "km", "mile")):
        return "decimal"
    return "number"


def _strip_score(item: Dict) -> Dict:
    return {k: v for k, v in item.items() if k != "score"}
