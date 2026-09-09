"""
agents/vertical_schema_agent.py
================================
VerticalSchemaAgent — Claude managed agent that creates atom definitions
for a business vertical.

TWO MODES:
  GENERIC  — Creates atoms from industry knowledge (no real data needed).
  CUSTOMER — Reconciles a customer's real schema against existing template atoms.

BATCH SUPPORT:
  Large verticals (50+ atoms) can be split into logical batches using batch_focus.
  Each batch run checks existing atoms first and only creates what is missing.
  Options: "all" | "core" | "operational" | "reference" | "financial"

SOURCE FILES USED FROM PROJECT:
  compiler/decision_tree.py  — operator routing rules
  compiler/slot_constants.py — canonical ID structure and field inference
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from anthropic import Anthropic

from core.config import settings
import core.paths as _paths
from services import atom_registry

from compiler.decision_tree import (
    OPERATOR_MEASURE,
    OPERATOR_MEASURE_SNAPSHOT_DEDUPED,
    OPERATOR_MEASURE_MONTH_FIXED,
    OPERATOR_MEASURE_QUARTER_FIXED,
    OPERATOR_MEASURE_YTD,
    OPERATOR_MEASURE_GROUPED,
    ROUTE_COMPOSITION,
    ROUTE_LLM_WIZARD,
)
from compiler.slot_constants import (
    clean_token,
    CURRENCY_HINTS, COUNT_MEASURES, PERCENT_MEASURES, DAY_MEASURES,
)

def _get_client():
    """Build a fresh client per call so a runtime API-key update takes effect immediately."""
    return Anthropic(api_key=settings.ANTHROPIC_API_KEY)


def _get_model():
    return settings.ANTHROPIC_MODEL

_RECORD_TYPES = ["record", "event", "state", "snapshot", "dimension"]
_FIELD_ROLES  = ["primary_key", "dimension", "measure", "time", "flag"]
_FIELD_TYPES  = ["string", "number", "date", "boolean"]
_ADDITIVITY   = ["additive", "non_additive", "semi_additive"]
_SYSTEMS      = ["ERP", "WMS", "CRM", "BANK", "MANUAL", "SYSTEM", "API", "FILE"]

_BATCH_DESCRIPTIONS = {
    "all":         "Create ALL atoms for this vertical — complete coverage.",
    "core":        "Focus only on CORE TRANSACTION atoms — primary fact tables recording main business transactions (orders, invoices, payments, shipments). Most queried atoms.",
    "operational": "Focus only on OPERATIONAL atoms — day-to-day operations and status tracking (inventory, schedules, maintenance logs, quality checks, production runs).",
    "reference":   "Focus only on REFERENCE / DIMENSION atoms — lookup and master data (customers, products, suppliers, employees, locations, chart of accounts).",
    "financial":   "Focus only on FINANCIAL atoms — money, budgets, forecasts, accounting (budgets, forecasts, ledger entries, bank transactions, cost allocations).",
}

_VERTICAL_HINTS = {
    "supply_chain": {
        "core":        ["orders", "order_lines", "purchase_orders", "po_lines", "invoices", "invoice_lines", "shipments", "goods_receipts", "returns", "payments"],
        "operational": ["inventory", "stock_movements", "quality_inspections", "delivery_events", "carrier_events", "demand_forecasts", "replenishment_orders"],
        "reference":   ["suppliers", "customers", "products", "items", "warehouses", "locations", "carrier_accounts", "employees"],
        "financial":   ["payables", "receivables", "ledger_entries", "budgets", "cost_allocations", "landed_costs"],
    },
    "finance": {
        "core":        ["journal_entries", "invoices", "payments", "receipts", "bank_transactions", "expense_claims"],
        "operational": ["approvals", "reconciliations", "period_closings", "audit_trail"],
        "reference":   ["chart_of_accounts", "cost_centers", "legal_entities", "currencies", "tax_codes"],
        "financial":   ["account_balances", "budgets", "forecasts", "fixed_assets", "depreciation_schedules", "tax_records"],
    },
    "hr": {
        "core":        ["employees", "contracts", "payroll_runs", "payslips", "timesheets", "attendance_records"],
        "operational": ["leave_requests", "performance_reviews", "disciplinary_actions", "training_completions", "recruitment_stages"],
        "reference":   ["departments", "positions", "job_grades", "locations", "competencies", "org_chart"],
        "financial":   ["salary_bands", "bonus_allocations", "benefits_costs", "headcount_budgets"],
    },
    "sales": {
        "core":        ["opportunities", "quotes", "orders", "order_lines", "invoices", "contracts"],
        "operational": ["activities", "tasks", "calls", "meetings", "pipeline_stages", "deal_changes"],
        "reference":   ["accounts", "contacts", "products", "price_lists", "territories", "sales_reps"],
        "financial":   ["commissions", "revenue_recognition", "forecasts", "targets", "discount_approvals"],
    },
    "logistics": {
        "core":        ["shipments", "deliveries", "collections", "freight_bookings", "customs_declarations"],
        "operational": ["tracking_events", "proof_of_deliveries", "exceptions", "route_changes", "driver_logs"],
        "reference":   ["carriers", "routes", "vehicles", "drivers", "depots", "customers", "locations"],
        "financial":   ["freight_costs", "surcharges", "carrier_invoices", "fuel_costs", "budgets"],
    },
    "warehouse": {
        "core":        ["receipts", "put_aways", "pick_lists", "pack_lists", "dispatch_orders", "transfers"],
        "operational": ["inventory", "stock_adjustments", "cycle_counts", "location_moves", "replenishment_tasks"],
        "reference":   ["locations", "bins", "zones", "products", "units_of_measure", "suppliers"],
        "financial":   ["inventory_valuation", "handling_costs", "storage_costs", "shrinkage_records"],
    },
    "procurement": {
        "core":        ["requisitions", "rfqs", "purchase_orders", "po_amendments", "receipts", "invoices"],
        "operational": ["supplier_responses", "bid_evaluations", "approvals", "expedites", "delivery_schedules"],
        "reference":   ["suppliers", "supplier_contacts", "categories", "items", "contracts", "approved_lists"],
        "financial":   ["spend_records", "budgets", "price_agreements", "savings_tracking", "payment_terms"],
    },
    "marketing": {
        "core":        ["campaigns", "campaign_activities", "leads", "conversions", "content_pieces"],
        "operational": ["email_sends", "ad_impressions", "clicks", "form_submissions", "event_attendances"],
        "reference":   ["channels", "segments", "personas", "products", "regions", "marketing_team"],
        "financial":   ["campaign_spend", "budgets", "roi_records", "cost_per_lead", "attribution_records"],
    },
    "customer": {
        "core":        ["customers", "accounts", "interactions", "support_tickets", "orders"],
        "operational": ["ticket_updates", "escalations", "churn_events", "renewal_events", "onboarding_steps"],
        "reference":   ["segments", "tiers", "products", "channels", "agents", "regions"],
        "financial":   ["subscriptions", "invoices", "payments", "mrr_records", "nps_surveys", "ltv_records"],
    },
    "operations": {
        "core":        ["work_orders", "production_runs", "jobs", "tasks", "incidents"],
        "operational": ["downtime_events", "shift_logs", "quality_checks", "maintenance_events", "safety_reports"],
        "reference":   ["assets", "machines", "locations", "workers", "products", "materials"],
        "financial":   ["labor_costs", "material_costs", "overhead_allocations", "budgets", "variance_records"],
    },
    "maintenance": {
        "core":        ["work_orders", "preventive_maintenance_plans", "corrective_actions", "inspections"],
        "operational": ["failure_events", "parts_usage", "technician_logs", "downtime_records", "meter_readings"],
        "reference":   ["assets", "asset_hierarchy", "locations", "technicians", "parts", "vendors"],
        "financial":   ["maintenance_costs", "parts_costs", "labor_costs", "budgets", "warranty_records"],
    },
    "project_management": {
        "core":        ["projects", "tasks", "milestones", "deliverables", "issues"],
        "operational": ["time_entries", "status_updates", "risks", "dependencies", "change_requests"],
        "reference":   ["resources", "teams", "clients", "skills", "project_types"],
        "financial":   ["budgets", "actuals", "forecasts", "invoices", "expense_claims"],
    },
    "clinical": {
        "core":        ["patients", "encounters", "admissions", "procedures", "prescriptions", "lab_orders"],
        "operational": ["lab_results", "vitals", "nursing_notes", "diagnoses", "discharge_events", "referrals"],
        "reference":   ["providers", "departments", "facilities", "icd_codes", "cpt_codes", "medications"],
        "financial":   ["claims", "billing_records", "payments", "insurance_authorizations", "cost_records"],
    },
    "legal": {
        "core":        ["matters", "cases", "contracts", "filings", "hearings", "transactions"],
        "operational": ["tasks", "deadlines", "document_events", "communications", "approval_events"],
        "reference":   ["clients", "counterparties", "attorneys", "courts", "jurisdictions", "practice_areas"],
        "financial":   ["billing_entries", "invoices", "payments", "budgets", "trust_accounts"],
    },
    "compliance": {
        "core":        ["policies", "controls", "audits", "findings", "incidents", "regulatory_filings"],
        "operational": ["control_tests", "remediation_actions", "training_completions", "risk_assessments"],
        "reference":   ["regulations", "frameworks", "business_units", "control_owners", "risk_categories"],
        "financial":   ["fines_penalties", "audit_costs", "remediation_costs", "compliance_budgets"],
    },
    "risk": {
        "core":        ["risk_events", "risk_assessments", "incidents", "near_misses", "insurance_claims"],
        "operational": ["control_activities", "mitigation_actions", "risk_reviews", "escalations"],
        "reference":   ["risk_categories", "business_units", "risk_owners", "frameworks", "rating_scales"],
        "financial":   ["exposures", "reserves", "insurance_premiums", "loss_records", "risk_budgets"],
    },
    "treasury": {
        "core":        ["cash_transactions", "bank_statements", "payments", "receipts", "fx_trades"],
        "operational": ["cash_positions", "bank_reconciliations", "payment_approvals", "fx_settlements"],
        "reference":   ["bank_accounts", "currencies", "counterparties", "investment_instruments", "entities"],
        "financial":   ["cash_forecasts", "investment_positions", "debt_facilities", "interest_records", "hedges"],
    },
}


def _get_batch_hint(vertical: str, batch_focus: str) -> str:
    hints = _VERTICAL_HINTS.get(vertical, {})
    if not hints or batch_focus == "all":
        return f"Think carefully about ALL main entities a {vertical} business needs. Be thorough."
    entities = hints.get(batch_focus, [])
    if not entities:
        return f"Focus on {batch_focus} entities for the {vertical} vertical."
    return f"Cover these entity types: {', '.join(entities)}. Create an atom for each one that is relevant."


def _build_decision_tree_rules() -> str:
    return f"""
DECISION TREE ROUTING (compiler/decision_tree.py):

  record_type = "state"
    time = "all_time"  → {OPERATOR_MEASURE_SNAPSHOT_DEDUPED}
    any other time     → {ROUTE_LLM_WIZARD}
    ⚠ dedup_sort_col MUST be set

  record_type = "record" | "event" | "snapshot"
    time = "this_month"/"last_month"                             → {OPERATOR_MEASURE_MONTH_FIXED}
    time = "ytd"/"year_to_date"                                  → {OPERATOR_MEASURE_YTD}
    time = "this_quarter"/"last_quarter"/"this_year"/"last_year" → {OPERATOR_MEASURE_QUARTER_FIXED}
    time = "all_time"                                            → {OPERATOR_MEASURE}
    any other time                                               → {ROUTE_LLM_WIZARD}

  series != "scalar" or scope starts with "by_"  → {OPERATOR_MEASURE_GROUPED}
  complexity = COMPOSED_1/COMPOSED_2/KPI         → {ROUTE_COMPOSITION}

UNIT INFERENCE (compiler/slot_constants.py):
  currency: {sorted(CURRENCY_HINTS)}
  count: {sorted(COUNT_MEASURES)}
  percent: {sorted(PERCENT_MEASURES)}
  days: {sorted(DAY_MEASURES)}
"""


# ── Atom discovery thinking framework ────────────────────────────────────────
# Forces Claude to think through 6 systematic lenses before creating atoms.
# Works for ANY vertical — known or unknown. Scales naturally with domain complexity.
_DISCOVERY_FRAMEWORK = """
Before creating any atoms, work through this discovery framework for the {vertical} vertical.
For each lens below, list every applicable entity you can think of. Be exhaustive.
Then create an atom for every entity you listed.

LENS 1 — MASTER DATA (dimension atoms)
  What are the core reference objects everything else links to?
  Think: people, places, products, categories, classifications, configurations.
  Examples: customers, employees, products, locations, departments, suppliers, accounts.

LENS 2 — TRANSACTIONS (record atoms)
  What discrete business events get recorded with a timestamp?
  Think: orders, invoices, payments, claims, requests, submissions, applications.
  Each transaction type that has its own fields and lifecycle = one atom.

LENS 3 — PROCESSES & WORKFLOWS (record atoms)
  What multi-step business processes generate their own data?
  Think: approvals, assessments, audits, reviews, inspections, evaluations, onboarding.
  If it has a start date, end date, assignee, and status — it's a process atom.

LENS 4 — STATE TRACKING (state atoms)
  What entities change their status/condition over time and need history?
  Think: inventory levels, case status, account balance, employee status, contract stage.
  If you need to know "what was the status ON A SPECIFIC DATE" — it's a state atom.

LENS 5 — ACTIVITY LOGS (event atoms)
  What timestamped individual actions or occurrences get logged?
  Think: login events, status changes, document accesses, system alerts, communications.
  Events are immutable facts — something happened at a point in time.

LENS 6 — PERIODIC SNAPSHOTS (snapshot atoms)
  What metrics or summaries get captured at regular intervals?
  Think: month-end balances, weekly KPI summaries, daily sales totals, quarterly reviews.
  Snapshots freeze a measurement at a point in time for trend analysis.

After completing all 6 lenses:
- You should have a list of 20-50 entities depending on the vertical's complexity
- Create an atom for each entity you identified
- A simple vertical (e.g. gift shop) may have 10-15 atoms — that is correct
- A complex vertical (e.g. healthcare, finance) may have 40-60 atoms — that is also correct
- The count should be driven by the domain, not by any arbitrary target
- Do NOT stop early just because you have "enough" — complete all 6 lenses first
"""


_SYSTEM = f"""
You are the VerticalSchemaAgent for the BizzBrain GPL system.

Your job is to create atom definitions for business verticals.
An atom defines one business data table — its fields, roles, grain keys, and FK relationships.

ATOM STRUCTURE:
  canonical_id : {{domain}}_{{entity}}_{{system}}_{{record_type}}
                 all lowercase, underscores only
                 e.g. supply_chain_purchase_orders_ERP_record

  record_type  : {_RECORD_TYPES}
  system       : {_SYSTEMS}
  grain_keys   : field names that uniquely identify one row
  grain_description : "one row per ..."
  dedup_sort_col    : date field for dedup — ONLY for state, null for all others

FIELD RULES:
  role       : {_FIELD_ROLES}
  type       : {_FIELD_TYPES}
  additivity : {_ADDITIVITY}

  primary_key → grain key, non_additive
  measure     → numeric to SUM/AVG/MAX/MIN — MUST be additive or semi_additive
  time        → date/datetime — at most ONE per atom
  dimension   → categorical for filtering — non_additive
  flag        → boolean/binary — non_additive

  ⚠ Status columns → role="dimension" non_additive (NOT "flag" unless truly boolean)
  ⚠ Measure fields MUST be additive or semi_additive — NEVER non_additive
  ⚠ Unit prices, unit costs, rates, percentages, ratios → role="measure" additivity="semi_additive"
     (you can AVG them meaningfully but not SUM them — e.g. AVG(unit_price) makes sense, SUM(unit_price) does not)
  ⚠ Quantities, amounts, totals, counts → role="measure" additivity="additive"
     (you can SUM them — e.g. SUM(order_amount), SUM(quantity) both make sense)

DECISION TREE:
{_build_decision_tree_rules()}
QUALITY RULES:
  Every atom MUST have at least one grain_key, one measure, one time, one dimension.
  Fields per atom: 5-15.
  State atoms MUST have dedup_sort_col set.

MOCK DATA:
  MockDataAgent generates data/mock_data/{{canonical_id}}.csv using DuckDB.
  CSV columns match atom field names exactly.
  Types: string=VARCHAR, number=DOUBLE, date=DATE, boolean=BOOLEAN.

IMPORTANT: Always call get_existing_atoms first. Write every atom. Do not stop early.
"""


TOOLS: List[Dict[str, Any]] = [
    {
        "name": "get_existing_atoms",
        "description": "Read existing atoms. Call this first to avoid duplicates.",
        "input_schema": {
            "type": "object",
            "properties": {"domain": {"type": "string", "description": "Filter by domain. Empty = all."}},
            "required": ["domain"]
        }
    },
    {
        "name": "write_atom",
        "description": "Write one atom to atoms.json. Validates before writing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "atom": {"type": "object", "description": "Complete atom with canonical_id, domain, name, description, record_type, system, grain_keys, grain_description, dedup_sort_col, fields."}
            },
            "required": ["atom"]
        }
    },
    {
        "name": "validate_atom",
        "description": "Validate atom before writing. Returns errors — empty means valid.",
        "input_schema": {
            "type": "object",
            "properties": {"atom": {"type": "object"}},
            "required": ["atom"]
        }
    },
    {
        "name": "get_field_values",
        "description": "Read enum values from field_values.json. Only useful in CUSTOMER mode.",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    }
]


def _tool_get_existing_atoms(domain: str) -> Dict:
    result = []
    for cid in atom_registry.list_atoms():
        a = atom_registry.get_atom(cid)
        if a and (not domain or a.get("domain") == domain):
            result.append({
                "canonical_id":   a.get("canonical_id"),
                "domain":         a.get("domain"),
                "record_type":    a.get("record_type"),
                "grain_keys":     a.get("grain_keys", []),
                "dedup_sort_col": a.get("dedup_sort_col"),
                "fields":         [f["name"] for f in a.get("fields", [])],
            })
    return {"atoms": result, "count": len(result)}


def _tool_validate_atom(atom: Dict) -> Dict:
    errors, warnings = [], []
    for k in ["canonical_id", "domain", "name", "description",
              "record_type", "system", "grain_keys", "grain_description", "fields"]:
        if k not in atom:
            errors.append(f"Missing required key: '{k}'")
    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}

    if atom.get("record_type") not in _RECORD_TYPES:
        errors.append(f"record_type must be one of {_RECORD_TYPES}")
    if atom.get("system") not in _SYSTEMS:
        errors.append(f"system must be one of {_SYSTEMS}")
    if not isinstance(atom.get("grain_keys"), list) or not atom["grain_keys"]:
        errors.append("grain_keys must be a non-empty list")
    if atom.get("record_type") == "state" and not atom.get("dedup_sort_col"):
        errors.append(f"dedup_sort_col required for state atoms ({OPERATOR_MEASURE_SNAPSHOT_DEDUPED})")
    if atom.get("record_type") != "state" and atom.get("dedup_sort_col"):
        errors.append(f"dedup_sort_col must be null for record_type='{atom.get('record_type')}'")

    names = set()
    has_measure = has_time = False
    for i, f in enumerate(atom.get("fields", [])):
        n = f.get("name", f"[{i}]")
        if n in names: errors.append(f"Duplicate field: '{n}'")
        names.add(n)
        if f.get("role") not in _FIELD_ROLES:       errors.append(f"Field '{n}': invalid role '{f.get('role')}'")
        if f.get("type") not in _FIELD_TYPES:       errors.append(f"Field '{n}': invalid type '{f.get('type')}'")
        if f.get("additivity") not in _ADDITIVITY:  errors.append(f"Field '{n}': invalid additivity '{f.get('additivity')}'")
        if f.get("role") == "measure": has_measure = True
        if f.get("role") == "time":    has_time    = True

        # measure must never be non_additive — catches unit_price, rates, ratios
        if f.get("role") == "measure" and f.get("additivity") == "non_additive":
            errors.append(
                f"Field '{n}': measure cannot be non_additive. "
                f"Use additive (quantities/amounts) or semi_additive (prices/rates/percentages)"
            )

    for gk in atom.get("grain_keys", []):
        if gk not in names: errors.append(f"grain_key '{gk}' not in fields")
    dsc = atom.get("dedup_sort_col")
    if dsc and dsc not in names: errors.append(f"dedup_sort_col '{dsc}' not in fields")

    if not has_measure: warnings.append("No measure fields")
    if not has_time:    warnings.append(f"No time field — {OPERATOR_MEASURE_MONTH_FIXED}/{OPERATOR_MEASURE_YTD} won't apply")

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


def _tool_write_atom(atom: Dict) -> Dict:
    v = _tool_validate_atom(atom)
    if not v["valid"]:
        return {"status": "error", "errors": v["errors"]}
    try:
        result = atom_registry.upsert_atom(atom)
        return {"status": result["status"], "canonical_id": result["canonical_id"],
                "fields_total": result["fields_total"], "warnings": v.get("warnings", [])}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _tool_get_field_values() -> Dict:
    if not _paths.FIELD_VALUES_PATH.exists():
        return {"field_values": {}, "note": "field_values.json not found"}
    try:
        return {"field_values": json.loads(_paths.FIELD_VALUES_PATH.read_text(encoding="utf-8"))}
    except Exception as e:
        return {"field_values": {}, "error": str(e)}


def _dispatch(name: str, inp: Dict) -> str:
    try:
        if   name == "get_existing_atoms":  r = _tool_get_existing_atoms(inp.get("domain", ""))
        elif name == "write_atom":          r = _tool_write_atom(inp["atom"])
        elif name == "validate_atom":       r = _tool_validate_atom(inp["atom"])
        elif name == "get_field_values":    r = _tool_get_field_values()
        else:                               r = {"error": f"Unknown tool: {name}"}
    except Exception as e:
        r = {"error": f"Tool error: {e}"}
    return json.dumps(r)


def run(
    vertical:        str,
    mode:            str = "GENERIC",
    customer_schema: Optional[Dict] = None,
    entity_checklist: Optional[List[str]] = None,
    batch_focus:     str = "all",
    max_iterations:  int = 400,
) -> Dict:
    """
    Run the VerticalSchemaAgent.

    Args:
        vertical:        e.g. "supply_chain", "hr", "finance"
        mode:            "GENERIC" or "CUSTOMER"
        customer_schema: Required for CUSTOMER mode
        batch_focus:     "all" | "core" | "operational" | "reference" | "financial"
                         Split large verticals (50+ atoms) into batches.
                         Each batch checks existing atoms first — no duplicates.
        max_iterations:  400 covers even the largest vertical (100 atoms x 3.5 + 2 = 352)

    Returns:
        {status, mode, vertical, batch_focus, atoms_created, atoms_updated,
         iterations, summary}
    """
    if mode not in ("GENERIC", "CUSTOMER"):
        return {"status": "error", "error": "mode must be GENERIC or CUSTOMER"}
    if mode == "CUSTOMER" and not customer_schema:
        return {"status": "error", "error": "customer_schema required for CUSTOMER mode"}
    if batch_focus not in _BATCH_DESCRIPTIONS:
        return {"status": "error", "error": f"batch_focus must be one of {list(_BATCH_DESCRIPTIONS.keys())}"}

    batch_desc = _BATCH_DESCRIPTIONS[batch_focus]
    batch_hint = _get_batch_hint(vertical, batch_focus)

    if mode == "GENERIC":
        # Build checklist section — explicit entity list if provided
        if entity_checklist:
            checklist_str = "\n".join(f"  - {e}" for e in entity_checklist)
            checklist_section = f"""
ENTITY CHECKLIST (you must create an atom for each of these):
{checklist_str}

Go through this list item by item. Do not stop until every entity on the list
has been created OR you have confirmed it already exists in get_existing_atoms.
"""
        else:
            checklist_section = _DISCOVERY_FRAMEWORK.format(vertical=vertical)

        task = f"""
Create atom definitions for the '{vertical}' business vertical.

MODE: GENERIC — use industry knowledge. No real customer data exists yet.

BATCH FOCUS: {batch_focus.upper()}
{batch_desc}

{checklist_section}

STEPS:
1. Call get_existing_atoms(domain='{vertical}') first.
   For each entity on the checklist — if it already exists, skip it.
   If it does not exist, create it.
2. Work through the checklist top to bottom. Create every missing atom.
3. Use validate_atom before write_atom if unsure about structure.

Note: do NOT declare FK relationships — that is handled by a separate
agent after this one finishes. Focus only on atom definitions.

Do not declare done until every item on the checklist is either
already present or just created. Be thorough.
"""
    else:
        task = f"""
Reconcile the customer's schema for '{vertical}'.

MODE: CUSTOMER — use ONLY columns in the customer's data.

BATCH FOCUS: {batch_focus.upper()}
{batch_desc}

CUSTOMER SCHEMA:
{json.dumps(customer_schema, indent=2)}

STEPS:
1. Call get_existing_atoms(domain='{vertical}') to see templates.
2. For each relevant customer table:
   - Find closest matching template atom.
   - Matched columns: INHERIT role from template.
   - New columns: infer role from name and type.
   - Missing columns: EXCLUDE them.

Note: do NOT declare FK relationships — that is handled by a separate
agent after this one finishes. Focus only on atom definitions.
"""

    messages       = [{"role": "user", "content": task}]
    atoms_created  = []
    atoms_updated  = []
    iterations     = 0
    summary        = ""

    while iterations < max_iterations:
        iterations += 1

        response = _get_client().messages.create(
            model=_get_model(),
            max_tokens=4096,
            system=_SYSTEM,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    summary = block.text
            break

        if response.stop_reason == "tool_use":
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result_str = _dispatch(block.name, block.input)
                result_obj = json.loads(result_str)
                if block.name == "write_atom":
                    if result_obj.get("status") == "created":
                        atoms_created.append(result_obj.get("canonical_id"))
                    elif result_obj.get("status") == "updated":
                        atoms_updated.append(result_obj.get("canonical_id"))
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": result_str})
            messages.append({"role": "user", "content": results})
        else:
            break

    status = "success" if (atoms_created or atoms_updated) else "no_changes"
    return {
        "status":                status,
        "mode":                  mode,
        "vertical":              vertical,
        "batch_focus":           batch_focus,
        "atoms_created":         atoms_created,
        "atoms_updated":         atoms_updated,
        "iterations":            iterations,
        "summary":               summary,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run VerticalSchemaAgent")
    parser.add_argument("vertical")
    parser.add_argument("--mode",  default="GENERIC", choices=["GENERIC", "CUSTOMER"])
    parser.add_argument("--batch", default="all",     choices=list(_BATCH_DESCRIPTIONS.keys()))
    parser.add_argument("--schema", help="Customer schema JSON path (CUSTOMER mode)")
    args = parser.parse_args()

    schema = None
    if args.mode == "CUSTOMER":
        if not args.schema:
            print("ERROR: --schema required for CUSTOMER mode")
            sys.exit(1)
        schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))

    result = run(vertical=args.vertical, mode=args.mode,
                 customer_schema=schema, batch_focus=args.batch)
    print(json.dumps(result, indent=2))
