"""
customer/numbers_graph.py
==========================
Builds the "Your Numbers" walk graph from a customer's canonical_index.
NO LLM. Pure dict construction. ~30-60ms on 500 metrics.

depends_on is derived from formula_line:
  composed aterms look like:  "cid = {dep1} / ({dep2} if ...)"
  operator aterms look like:  "MEASURE('table', ...)"

RETURNED GRAPH SHAPE:
{
  "ok": true,
  "nodes": {
    "<cid>": {
      "name", "label", "value", "unit", "gap",
      "up":             [{"name": cid}, ...],
      "down":           [{"name": cid, "op": "/"}, ...],
      "time_siblings":  [{"name", "time", "value"}, ...],
      "state_siblings": [{"name", "state", "value"}, ...],
      "series_variants":[{"name", "series"}, ...],
      "source_table", "source_goal", "formula", "kind"
    }
  },
  "order":        [cid, ...],
  "page":         [cid, ...],
  "domain_label": "Logistics"
}
"""
from __future__ import annotations
import json, re
from collections import defaultdict
from pathlib import Path
from typing import Optional


# ── unit correction ───────────────────────────────────────────────────────────
# The factory compiler sometimes assigns unit='currency' to non-currency fields.
# We correct this at graph-build time by inspecting the canonical_id.
# Rules are applied in order; first match wins.

_UNIT_RULES: list[tuple[str, str]] = [
    # percent / growth / rate — must come before _kg / _km checks
    # because growth_freight_cost_percent has _percent in it
    ("_percent",        "percent"),
    ("_growth_",        "percent"),   # growth aterms are always % change
    ("_utilisation_rate", "percent"),
    ("_rate_percent",   "percent"),
    ("_delivery_rate",  "percent"),
    ("_compliance_rate","percent"),
    ("_fulfillment_rate","percent"),
    ("_delay_rate",     "percent"),
    ("_resolution_rate","percent"),
    ("_success_rate",   "percent"),
    # rates that are ratios (currency/count or currency/currency)
    ("_claim_rate",     "ratio"),
    ("_cost_per_",      "ratio"),
    ("_per_trip",       "ratio"),
    ("_per_unit",       "ratio"),
    # weight
    ("_weight_kg",      "kg"),
    ("_total_weight",   "kg"),
    ("_max_weight",     "kg"),
    ("_min_weight",     "kg"),
    # distance / efficiency
    ("_km_per_litre",   "ratio"),
    ("_per_kg",         "ratio"),     # cost/kg → ratio, not currency
    ("_distance_km",    "km"),
    ("_km",             "km"),
    # time
    ("_days",           "days"),
    ("_resolution_days","days"),
    ("_lead_time_days", "days"),
    # count
    ("_count",          "count"),
    ("_total_trips_completed", "count"),
    ("_number_of",      "count"),
]

def _correct_unit(cid: str, declared_unit: str) -> str:
    """
    Override a wrongly declared unit using canonical_id keyword rules.
    If no rule matches, return the declared unit unchanged.
    """
    cid_lower = cid.lower()
    for keyword, correct_unit in _UNIT_RULES:
        if keyword in cid_lower:
            return correct_unit
    return declared_unit


# ── helpers ───────────────────────────────────────────────────────────────────

def _label(cid: str) -> str:
    """logistics_incidents_average_claim_amount_currency → Incidents average claim amount"""
    parts = cid.split("_")
    DOMAINS = {"logistics", "supply", "chain", "retail", "shopify"}
    while parts and parts[0] in DOMAINS:
        parts = parts[1:]
    # also strip trailing unit word if it matches the unit
    UNIT_WORDS = {"currency", "count", "percent", "days", "kg", "km", "ratio"}
    if parts and parts[-1] in UNIT_WORDS:
        parts = parts[:-1]
    return " ".join(parts)


def _fmt(v) -> Optional[float]:
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _source_table(formula: str) -> str:
    m = re.search(r"'([^']+)'", formula or "")
    return m.group(1) if m else ""


_AGG_OPS = {"SUM", "COUNT", "COUNT_DISTINCT", "AVG", "MEAN", "MAX", "MIN",
            "MEDIAN", "LAST", "FIRST"}

def parse_formula(formula: str) -> Optional[dict]:
    """
    Parse any MEASURE* formula_line into:
      { table, operation, column, filters, variant, is_primitive }

    Handles all MEASURE variants produced by the compiler:
      MEASURE('table', 'AGG', 'col', [filters])
      MEASURE_MONTH_FIXED('table', 'date_col', 'AGG', 'col', 'window', [filters])
      MEASURE_QUARTER_FIXED(...)  — same layout as MONTH_FIXED
      MEASURE_YTD('table', 'date_col', 'AGG', 'col', [filters])
      MEASURE_GROUPED('table', 'AGG', 'col', ...)
      MEASURE_GROUPED_YTD('table', 'date_col', 'AGG', 'col', ...)
      MEASURE_SNAPSHOT_DEDUPED('table', 'snap_col', [keys], 'AGG', 'col', [f])
      MEASURE_SNAPSHOT_DEDUPED_MONTH/QUARTER/YTD — similar layout
      MEASURE_SET_CHURNED('table', ...) — count-only, no agg col

    Returns None for compound formulas ({dep} references) or unparseable input.
    """
    if not formula:
        return None
    # compound: contains {dep_cid} references
    if re.search(r'\{[a-z][a-z0-9_]+\}', formula):
        return None
    # must start with MEASURE
    m = re.match(r'(MEASURE\w*)\s*\(', formula)
    if not m:
        return None

    variant = m.group(1)
    # extract all string literals in order
    strings = re.findall(r"'([^']*)'", formula)
    # extract filter list (list of dicts)
    try:
        import ast
        rhs = formula[formula.index('('):]
        args = ast.literal_eval(rhs) if rhs.startswith('(') else ()
        filters = next((a for a in args if isinstance(a, list)), []) or []
    except Exception:
        filters = []

    if not strings:
        return None

    table = strings[0]
    # Find the aggregation op among the string args
    operation = next((s for s in strings if s.upper() in _AGG_OPS), "COUNT")
    # Column is the string immediately after the operation
    column = None
    for i, s in enumerate(strings):
        if s.upper() == operation and i + 1 < len(strings):
            column = strings[i + 1]
            break

    # For MEASURE_SET_CHURNED there's no meaningful column
    if variant == "MEASURE_SET_CHURNED":
        operation = "COUNT"
        column = None

    return {
        "variant":      variant,
        "table":        table,
        "operation":    operation,
        "column":       column,
        "filters":      filters,
        "is_primitive": True,
    }


def parse_compound_deps(formula: str, index: dict) -> list[dict]:
    """
    For a compound formula ({dep1} / {dep2} ...), return each dep with its
    parse_formula result so the trace endpoint can recurse.
    """
    deps = re.findall(r'\{([a-z][a-z0-9_]+)\}', formula)
    seen, out = set(), []
    for dep in deps:
        if dep in seen:
            continue
        seen.add(dep)
        entry = index.get(dep, {})
        dep_formula = entry.get("formula_line", "")
        parsed = parse_formula(dep_formula)
        out.append({
            "cid":       dep,
            "label":     _label(dep),
            "value":     _fmt(entry.get("oracle_value")),
            "unit":      (entry.get("slots") or {}).get("unit", ""),
            "primitive": parsed,
            "compound":  dep_formula if not parsed and dep_formula else None,
        })
    return out


def _extract_deps(cid: str, formula: str, all_cids: set) -> list[str]:
    if not formula or not formula.startswith(cid):
        return []
    refs = re.findall(r'\{([a-z][a-z0-9_]*)\}', formula)
    seen, out = set(), []
    for r in refs:
        if r in all_cids and r != cid and r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _op(formula: str, dep: str) -> str:
    if not formula:
        return "+"
    if "/" in formula:
        after_div = formula.split("/", 1)[1]
        if dep in after_div:
            return "/"
    if " - " in formula:
        after_sub = formula.split(" - ", 1)[1]
        if dep in after_sub:
            return "-"
    return "+"


# ── main builder ──────────────────────────────────────────────────────────────

def build_numbers_graph(canonical_index: dict,
                        context_filter: Optional[str] = None) -> dict:

    # Collect all available domains from the full index (before filtering)
    DOMAIN_LABELS_ALL = {
        "logistics":      "Logistics",
        "supply_chain":   "Supply chain",
        "retail_shopify": "Retail",
    }
    available_contexts = sorted(set(
        (e.get("slots") or {}).get("domain", "")
        for e in canonical_index.values()
        if (e.get("slots") or {}).get("domain", "")
    ))
    available_contexts_labeled = [
        {"value": d, "label": DOMAIN_LABELS_ALL.get(d, d.replace("_", " ").title())}
        for d in available_contexts
    ]

    # 1. Filter by context
    raw = {
        cid: e for cid, e in canonical_index.items()
        if not context_filter
           or (e.get("slots") or {}).get("domain") == context_filter
    }
    if not raw:
        return {"ok": True, "nodes": {}, "order": [], "page": [],
                "domain_label": context_filter or "",
                "available_contexts": available_contexts_labeled}

    all_cids = set(raw)

    # 2. Derive depends_on from formula_line
    depends_on: dict[str, list] = {}
    for cid, e in raw.items():
        depends_on[cid] = _extract_deps(cid, e.get("formula_line", ""), all_cids)

    # 3. Reverse index: child → composed parents
    used_by: dict[str, list] = defaultdict(list)
    for cid, deps in depends_on.items():
        for dep in deps:
            used_by[dep].append(cid)

    # 4. Slot indexes for sibling lookups
    by_ent_meas:      dict[tuple, list] = defaultdict(list)
    by_ent_meas_time: dict[tuple, list] = defaultdict(list)
    by_series:        dict[tuple, list] = defaultdict(list)

    for cid, e in raw.items():
        s     = e.get("slots") or {}
        ent   = s.get("entity", "")
        meas  = s.get("measure", "")
        time  = s.get("time", "")
        ser   = s.get("series", "scalar")
        state = s.get("state", "all")
        if ser == "scalar":
            if state == "all":
                by_ent_meas[(ent, meas)].append(cid)
            by_ent_meas_time[(ent, meas, time)].append(cid)
        else:
            by_series[(ent, meas, time)].append(cid)

    # 5. Build nodes
    nodes: dict[str, dict] = {}
    for cid, e in raw.items():
        s     = e.get("slots") or {}
        ent   = s.get("entity", "")
        meas  = s.get("measure", "")
        time  = s.get("time", "")
        ser   = s.get("series", "scalar")
        state = s.get("state", "all")
        declared_unit = s.get("unit", "")
        form  = e.get("formula_line", "")
        oval  = _fmt(e.get("oracle_value"))
        deps  = depends_on[cid]
        is_composed = bool(deps)

        # ── unit correction ──────────────────────────────────────────────────
        unit = _correct_unit(cid, declared_unit)

        down = [{"name": d, "op": _op(form, d)} for d in deps]
        up   = [{"name": u} for u in used_by.get(cid, [])]

        # time siblings
        time_sib = []
        if ser == "scalar" and state == "all" and time == "all_time":
            for sib in by_ent_meas.get((ent, meas), []):
                if sib == cid:
                    continue
                se = raw[sib]
                ss = se.get("slots") or {}
                time_sib.append({
                    "name":  sib,
                    "time":  ss.get("time", ""),
                    "value": _fmt(se.get("oracle_value")),
                    "unit":  _correct_unit(sib, ss.get("unit", "")),
                })

        # state siblings
        state_sib = []
        if ser == "scalar" and state == "all":
            for sib in by_ent_meas_time.get((ent, meas, time), []):
                if sib == cid:
                    continue
                se = raw[sib]
                ss = se.get("slots") or {}
                sib_state = ss.get("state", "all")
                if sib_state == "all":
                    continue
                state_sib.append({
                    "name":  sib,
                    "state": sib_state,
                    "value": _fmt(se.get("oracle_value")),
                    "unit":  _correct_unit(sib, ss.get("unit", "")),
                })

        # series variants
        ser_var = []
        if ser == "scalar":
            for sib in by_series.get((ent, meas, time), []):
                se = raw[sib]
                ss = se.get("slots") or {}
                oval_raw = se.get("oracle_value")
                if isinstance(oval_raw, dict) and oval_raw:
                    ser_total = sum(v for v in oval_raw.values() if isinstance(v, (int, float)))
                    ser_bkd   = [{"k": k, "v": _fmt(v)} for k, v in oval_raw.items()]
                elif isinstance(oval_raw, (int, float)):
                    ser_total = oval_raw
                    ser_bkd   = []
                else:
                    ser_total = None
                    ser_bkd   = []
                ser_var.append({
                    "name":      sib,
                    "series":    ss.get("series", ""),
                    "unit":      _correct_unit(sib, ss.get("unit", "")),
                    "value":     _fmt(ser_total),
                    "breakdown": ser_bkd,
                    "gap":       ser_total is None,
                })

        nodes[cid] = {
            "name":            cid,
            "label":           _label(cid),
            "value":           oval,
            "unit":            unit,
            "declared_unit":   declared_unit,   # keep original for debugging
            "gap":             oval is None,
            "kind":            "composed" if is_composed else "operator",
            "up":              up,
            "down":            down,
            "time_siblings":   time_sib,
            "state_siblings":  state_sib,
            "series_variants": ser_var,
            "source_table":    _source_table(form),
            "source_goal":     e.get("source_goal", ""),
            "formula":         form,
        }

    # 6. Walk order
    walk_cids = [
        cid for cid, e in raw.items()
        if (e.get("slots") or {}).get("series", "scalar") == "scalar"
        and (e.get("slots") or {}).get("state", "all") == "all"
        and (e.get("slots") or {}).get("time", "") == "all_time"
    ]
    def _sort_key(cid):
        n = nodes[cid]
        s = raw[cid].get("slots") or {}
        return (0 if n["kind"] == "composed" else 1,
                s.get("entity", ""), s.get("measure", ""))
    walk_cids.sort(key=_sort_key)

    # 7. Starting page — universal priority scorer (vertical-agnostic)
    #
    # Priority is based on what the measure MEANS, not which vertical it
    # came from. Works for any future vertical automatically.
    #
    # Score breakdown (higher = more important):
    #   +40  Tier 1 — "how much money" (revenue, spend, profit, total cost)
    #   +30  Tier 2 — "how well performing" (rates, efficiency, fulfillment)
    #   +20  Tier 3 — "how much volume" (counts of key business objects)
    #   +10  Tier 4 — base: any real metric not in tiers above
    #    +5  Bonus  — composed aterm (formula, not raw operator)
    #    +3  Bonus  — no UP parents (true top-level)
    #   -10  Penalty — average_*, min_*, max_* (derived/drill-down)
    #   -20  Penalty — per_* / ratio patterns (very granular)

    _T1 = {  # Money totals — always hero KPIs
        "total_revenue", "revenue", "gross_revenue", "net_revenue",
        "total_spend", "spend_amount", "procurement_spend",
        "gross_amount", "net_amount", "total_amount",
        "profit", "net_profit", "gross_profit", "profit_margin",
        "total_sales", "sales_amount",
        "total_cost", "total_cost_impact", "cost_impact",
        "total_inventory_value", "inventory_value",
        "total_freight_cost", "freight_cost",
        "contracted_value", "total_line_items",
        "claim_amount", "return_value", "credit_amount",
        "gross_merchandise_value", "gmv", "total_spent",
    }
    _T2 = {  # Performance rates — always second priority
        "on_time_delivery_rate", "fulfillment_rate", "fill_rate",
        "purchase_order_fulfillment_rate", "order_fulfillment_rate",
        "defect_rate", "quality_acceptance_rate",
        "supplier_on_time_delivery_rate", "delivery_rate",
        "fleet_active_utilisation_rate", "utilisation_rate",
        "forecast_accuracy_rate", "inventory_fill_rate",
        "incident_resolution_rate", "resolution_rate",
        "fuel_efficiency_km_per_litre", "fuel_efficiency",
        "conversion_rate", "churn_rate", "retention_rate",
        "on_time_rate", "accuracy_rate", "compliance_rate",
        "trip_delivery_success_rate", "capacity_utilisation",
    }
    _T3 = {  # Volume counts — third priority
        "total_orders", "order_count",
        "total_shipments", "shipment_count",
        "total_trips_completed", "trip_count",
        "customer_count", "active_customers",
        "supplier_count", "total_products", "product_count",
        "employee_count", "headcount", "total_transactions",
    }

    def _priority_score(cid: str) -> int:
        m  = (raw[cid].get("slots") or {}).get("measure", "")
        nd = nodes[cid]
        score = 0

        # Tier matching — exact first, then substring
        if m in _T1 or any(t in m for t in _T1):
            score += 40
        elif m in _T2 or any(t in m for t in _T2):
            score += 30
        elif m in _T3 or any(t in m for t in _T3):
            score += 20
        else:
            score += 10  # real metric, not in priority list

        # Bonuses
        if nd["kind"] == "composed":
            score += 5
        if not nd["up"]:
            score += 3

        # Penalties for derived/drill-down measures
        if m.startswith(("average_", "min_", "max_")):
            score -= 10
        if any(p in m for p in ("_per_", "per_kg", "per_unit",
                                  "per_trip", "per_shipment", "per_order",
                                  "cost_per_", "avg_freight_cost")):
            score -= 20

        return score

    scored = sorted(walk_cids, key=_priority_score, reverse=True)
    page   = scored[:4]

    DOMAIN_LABELS = {
        "logistics":      "Logistics",
        "supply_chain":   "Supply chain",
        "retail_shopify": "Retail",
    }

    return {
        "ok":                True,
        "nodes":             nodes,
        "order":             walk_cids,
        "page":              page,
        "domain_label":      DOMAIN_LABELS.get(context_filter or "", context_filter or "All"),
        "available_contexts": available_contexts_labeled,
    }


def load_and_build(canonical_index_path: Path,
                   context: Optional[str] = None) -> dict:
    if not canonical_index_path.exists():
        return {"ok": False, "error": "No metrics compiled yet."}
    try:
        idx = json.loads(canonical_index_path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"ok": False, "error": f"Could not read index: {e}"}
    return build_numbers_graph(idx, context_filter=context)
