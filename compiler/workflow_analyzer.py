"""
compiler/workflow_analyzer.py
==============================
Workflow Analyzer — semantic validation of compiled aterms.

Alistair's concept: take a compiled aterm (business question + formula +
execution path) and ask AI: "does this formula look correct for the stated
business question?"

This is NOT mathematical verification (that's verifier.py).
This is NOT a second-opinion number check (that's independent_oracle.py).
This is a semantic/business-logic gut-check using AI reasoning.

Design decisions:
  - Advisory only: results never automatically update ai_locked
  - Human review required before any status change
  - Multi-model: Claude default + optional GPT-4o / Gemini
  - Confidence = checks_passed / total_checks (earned, not estimated)
  - Runs on a sample, not the full vertical
  - Configurable checklist: disable checks that prove unhelpful

Entry point:
  analyze_sample(vertical, canonical_ids, models) → results dict
  analyze_single(aterm, models)                   → single result dict

Storage:
  data/workflow_analysis_reports/{vertical}/report_{timestamp}.json
  data/llm_keys.json  ← API keys for non-Claude models

Human review decisions:
  data/wa_decisions_{vertical}.json  ← {cid: {status, reviewed_by, reviewed_at, note}}
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import core.paths as _paths
from core.config import settings

log = logging.getLogger(__name__)


# ── Configurable checklist ─────────────────────────────────────────────────────
# Set a check to False to disable it globally. After early runs, disable
# checks that consistently add no signal.

ENABLED_CHECKS = {
    # Composed aterm checks (kind == "composed")
    "C1_operation_matches_goal":        True,
    "C2_dependencies_correct":          True,
    "C3_time_scope_aligned":            True,
    "C4_state_filter_aligned":          True,
    "C5_division_by_zero_handled":      True,
    "C6_result_unit_consistent":        True,
    # Operator aterm checks (kind == "operator" or wave-based)
    "O1_aggregation_matches_goal":      True,
    "O2_correct_column_aggregated":     True,
    "O3_correct_source_table":          True,
    "O4_time_filter_correct":           True,
    "O5_groupby_filter_correct":        True,
    "O6_result_unit_consistent":        True,
}

# ── Model catalogue ────────────────────────────────────────────────────────────
MODEL_CATALOGUE = {
    "claude": {
        "label":       "Claude (Anthropic)",
        "provider":    "anthropic",
        "model_id":    "claude-sonnet-4-6",
        "recommended": True,
        "reason":      "Strong structured reasoning and formula semantics. Default — no API key required.",
        "default":     True,
    },
    "gpt4o": {
        "label":       "GPT-4o (OpenAI)",
        "provider":    "openai",
        "model_id":    "gpt-4o",
        "recommended": True,
        "reason":      "Excellent at code-level logical analysis and formula correctness.",
        "default":     False,
    },
    "gemini": {
        "label":       "Gemini 1.5 Pro (Google)",
        "provider":    "google",
        "model_id":    "gemini-1.5-pro",
        "recommended": True,
        "reason":      "Strong at structured data reasoning and multi-step analysis.",
        "default":     False,
    },
}

# ── Verdict thresholds ─────────────────────────────────────────────────────────
PASS_THRESHOLD = 0.90   # avg confidence >= 0.90 → PASSED
FAIL_THRESHOLD = 0.60   # avg confidence < 0.60 and models agree wrong → FAILED
                        # between 0.60-0.90 → FLAGGED


# ══════════════════════════════════════════════════════════════════════════════
# LLM KEY MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def _keys_path() -> Path:
    return _paths.DATA_DIR / "llm_keys.json"


def load_llm_keys() -> Dict[str, str]:
    """Load saved API keys for non-Claude models."""
    p = _keys_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_llm_keys(keys: Dict[str, str]) -> None:
    """Save API keys. Keys are stored as-is — never logged."""
    _keys_path().write_text(json.dumps(keys, indent=2), encoding="utf-8")


def delete_llm_key(provider: str) -> None:
    """Remove a stored API key for a provider."""
    keys = load_llm_keys()
    keys.pop(provider, None)
    save_llm_keys(keys)


def get_api_key(provider: str) -> Optional[str]:
    """Get API key: Claude uses settings, others use llm_keys.json."""
    if provider == "anthropic":
        return settings.ANTHROPIC_API_KEY
    return load_llm_keys().get(provider)


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDING
# ══════════════════════════════════════════════════════════════════════════════

def _load_relationships(vertical: str) -> List[Dict]:
    """
    Load atom relationships from data/atom_relationships.json.
    Filters to only relationships relevant to the given vertical.
    Returns a flat list of {from_atom, from_field, to_atom, to_field} dicts.
    """
    rel_path = _paths.DATA_DIR / "atom_relationships.json"
    if not rel_path.exists():
        return []
    try:
        rels = json.loads(rel_path.read_text(encoding="utf-8"))
        # Filter to this vertical only
        return [
            r for r in rels
            if vertical in r.get("from_atom", "") or vertical in r.get("to_atom", "")
        ]
    except Exception as e:
        log.warning(f"[WA] Could not load relationships: {e}")
        return []


def _get_atom_relationships(atom_canonical_id: str, all_rels: List[Dict]) -> List[Dict]:
    """
    Return all relationships where the given atom is either source or target.
    """
    return [
        r for r in all_rels
        if r.get("from_atom") == atom_canonical_id or r.get("to_atom") == atom_canonical_id
    ]


def _format_relationships(atom_canonical_id: str, atom_rels: List[Dict]) -> str:
    """
    Format relationships for a given atom as a readable block for the prompt.
    Shows what this atom connects to and via which fields.
    """
    if not atom_rels:
        return "No relationships defined for this atom."

    lines = []
    for r in atom_rels:
        if r["from_atom"] == atom_canonical_id:
            lines.append(
                f"  {atom_canonical_id}.{r['from_field']}"
                f"  →  {r['to_atom']}.{r['to_field']}"
                f"  (this atom joins TO {r['to_atom']})"
            )
        else:
            lines.append(
                f"  {r['from_atom']}.{r['from_field']}"
                f"  →  {atom_canonical_id}.{r['to_field']}"
                f"  ({r['from_atom']} joins INTO this atom)"
            )
    return "\n".join(lines)


def _is_grouped_formula(formula: str) -> bool:
    """Check if a formula uses MEASURE_GROUPED (cross-atom join implied)."""
    return "MEASURE_GROUPED" in formula.upper()


def _deps_span_different_atoms(aterm: Dict) -> bool:
    """
    Check if a composed aterm's dependencies come from different source atoms.
    If yes, relationships are relevant to verify the composition makes sense.
    """
    dep_cids  = aterm.get("depends_on", [])
    dep_atoms = set()
    for dep_cid in dep_cids:
        dep_aterm = _load_aterm(dep_cid)
        if dep_aterm:
            atom_name = _extract_atom_name_from_formula(dep_aterm.get("formula_line", ""))
            if atom_name:
                dep_atoms.add(atom_name)
    return len(dep_atoms) > 1


def _extract_atom_name_from_formula(formula: str) -> Optional[str]:
    """
    Extract the source atom canonical_id from a formula string.
    Handles patterns like:
      MEASURE(SUM(col) FROM atom_name WHERE ...)
      MEASURE_MONTH_FIXED(COUNT_DISTINCT(col) FROM atom_name WHERE ...)
    """
    import re
    match = re.search(r'\bFROM\s+([\w]+)', formula, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def _build_aterm_context(aterm: Dict) -> str:
    """
    Format aterm as human-readable context for the prompt.
    Includes:
      - Metric metadata (question, formula, slots)
      - Atom schema (columns, types, grain)
      - Full mock CSV data (all rows, all columns)
      - For composed aterms: each dependency with its own schema + mock data
    """
    slots       = aterm.get("slots", {})
    kind        = aterm.get("kind", "operator")
    source_goal = aterm.get("source_goal", "")
    cid         = aterm.get("canonical_id", "")
    formula     = aterm.get("formula_line", "")
    unit        = slots.get("unit", "")
    time_scope  = slots.get("time", "all_time")
    state       = slots.get("state", "all")
    entity      = slots.get("entity", "")
    domain      = slots.get("domain", "")

    ctx = f"""BUSINESS QUESTION: {source_goal}
METRIC ID:         {cid}
KIND:              {kind}
DOMAIN:            {domain}
ENTITY:            {entity}
UNIT:              {unit}
TIME SCOPE:        {time_scope}
STATE FILTER:      {state if state not in ('all', 'base', '') else 'none'}

FORMULA:
  {formula}"""

    if kind == "composed" and aterm.get("depends_on"):
        # ── Composed aterm: expand each dependency with schema + mock data ──
        dep_values   = aterm.get("dep_values", {})
        seen_atoms   = {}   # atom_canonical_id → already shown (avoid duplicate mock data)

        # Check if relationships are needed (deps span different atoms)
        all_rels      = _load_relationships(domain) if _deps_span_different_atoms(aterm) else []
        include_rels  = len(all_rels) > 0

        ctx += "\n\n" + "─" * 60
        ctx += "\nDEPENDENCY BREAKDOWN"
        ctx += "\n" + "─" * 60

        if include_rels:
            ctx += (
                "\nNOTE: This composed metric draws from multiple different source atoms. "
                "Atom relationships are provided below each dependency so you can verify "
                "whether the composition is semantically valid — i.e. the atoms being "
                "combined are genuinely related entities, not arbitrary cross-domain divisions."
            )

        for i, dep_cid in enumerate(aterm.get("depends_on", []), 1):
            dep_val   = dep_values.get(dep_cid, "unknown")
            dep_aterm = _load_aterm(dep_cid)

            ctx += f"\n\nDEPENDENCY {i}: {{{dep_cid}}}"
            ctx += f"\n  Oracle value: {dep_val}"

            if dep_aterm:
                dep_formula = dep_aterm.get("formula_line", "")
                dep_goal    = dep_aterm.get("source_goal", "")
                dep_slots   = dep_aterm.get("slots", {})
                ctx += f"\n  Goal:         {dep_goal}"
                ctx += f"\n  Formula:      {dep_formula}"
                ctx += f"\n  Time scope:   {dep_slots.get('time', '')}   State: {dep_slots.get('state', '')}"

                # Atom schema for this dependency
                dep_atom_name = _extract_atom_name_from_formula(dep_formula)
                if dep_atom_name:
                    if dep_atom_name not in seen_atoms:
                        atom_schema = _load_atom_schema(dep_atom_name)
                        if atom_schema:
                            ctx += f"\n\n  SOURCE ATOM SCHEMA — {dep_atom_name}:"
                            for line in _format_atom_schema(atom_schema).split("\n"):
                                ctx += f"\n    {line}"

                        # Full mock data
                        mock_csv = _load_mock_data(dep_atom_name, domain)
                        if mock_csv:
                            ctx += f"\n\n  FULL MOCK DATA — {dep_atom_name} (all rows):"
                            ctx += "\n  " + "·" * 55
                            for row in mock_csv.split("\n"):
                                ctx += f"\n  {row}"
                            ctx += "\n  " + "·" * 55

                        # Relationships — only when deps span different atoms
                        if include_rels:
                            dep_rels = _get_atom_relationships(dep_atom_name, all_rels)
                            if dep_rels:
                                ctx += f"\n\n  ATOM RELATIONSHIPS — {dep_atom_name}:"
                                ctx += "\n  " + "·" * 55
                                for line in _format_relationships(dep_atom_name, dep_rels).split("\n"):
                                    ctx += f"\n  {line}"
                                ctx += "\n  " + "·" * 55

                        seen_atoms[dep_atom_name] = True
                    else:
                        ctx += f"\n  [Mock data and schema for {dep_atom_name} already shown above — same source atom]"

    else:
        # ── Operator aterm: single source atom schema + mock data ──
        # Relationships included only if formula uses MEASURE_GROUPED
        atom_name = _extract_atom_name_from_formula(formula)
        if atom_name:
            atom_schema = _load_atom_schema(atom_name)
            if atom_schema:
                ctx += f"\n\n" + "─" * 60
                ctx += f"\nSOURCE ATOM SCHEMA — {atom_name}:"
                ctx += "\n" + "─" * 60
                for line in _format_atom_schema(atom_schema).split("\n"):
                    ctx += f"\n{line}"

            # Relationships — only for MEASURE_GROUPED (cross-atom join implied)
            if _is_grouped_formula(formula):
                all_rels  = _load_relationships(domain)
                atom_rels = _get_atom_relationships(atom_name, all_rels)
                ctx += f"\n\n" + "─" * 60
                ctx += f"\nATOM RELATIONSHIPS — {atom_name}:"
                ctx += "\n" + "─" * 60
                ctx += f"\n{_format_relationships(atom_name, atom_rels)}"
                ctx += (
                    "\n\nNOTE: This formula uses MEASURE_GROUPED (grouping implies a potential "
                    "cross-atom join). Use the relationships above to verify that any GROUP BY "
                    "field either exists on the source atom directly, or is reachable via a "
                    "defined relationship. Flag if a GROUP BY field requires a join that is "
                    "not reflected in the formula."
                )

            mock_csv = _load_mock_data(atom_name, domain)
            if mock_csv:
                ctx += f"\n\n" + "─" * 60
                ctx += f"\nFULL MOCK DATA — {atom_name} (all rows):"
                ctx += "\n" + "─" * 60
                ctx += "\n" + mock_csv

    oracle = aterm.get("oracle_value")
    if oracle is not None:
        val_str = f"{oracle:,.4f}" if isinstance(oracle, float) else str(oracle)
        ctx += f"\n\n" + "─" * 60
        ctx += f"\nCOMPUTED RESULT: {val_str}"
        ctx += "\n" + "─" * 60

    return ctx


def _build_checklist_for_kind(kind: str) -> tuple[list, str]:
    """
    Return (check_ids, checklist_text) for the given aterm kind.
    Only includes enabled checks.
    """
    if kind == "composed":
        all_checks = [
            ("C1", "C1_operation_matches_goal",
             "Does the mathematical operation in the formula match the business question?\n"
             "    (division for 'average', subtraction+division for 'growth rate',\n"
             "     ratio/proportion for 'rate' or '% of', SUM for totals)"),
            ("C2", "C2_dependencies_correct",
             "Are the dependency metrics the correct building blocks for this formula?\n"
             "    (e.g. average = total_value / total_count — are those the right totals?\n"
             "     ratio = numerator_metric / denominator_metric — correct metrics selected?)"),
            ("C3", "C3_time_scope_aligned",
             "Do the dependency time scopes align with this metric's time scope?\n"
             "    (all deps should share the same time window as the composed metric;\n"
             "     mixing all_time deps into a this_month metric would be wrong)"),
            ("C4", "C4_state_filter_aligned",
             "Do the dependency state filters align with this metric's state filter?\n"
             "    (if the composed metric is unfiltered, deps should also be unfiltered;\n"
             "     if filtered to 'active', deps should be consistently filtered)"),
            ("C5", "C5_division_by_zero_handled",
             "Is division-by-zero handled if the denominator dependency could be zero?\n"
             "    (look for a guard like 'if denominator != 0 else 0' or similar;\n"
             "     a missing guard is a real risk in production)"),
            ("C6", "C6_result_unit_consistent",
             "Does the result unit make sense for the operation performed?\n"
             "    (currency / count = currency ✓, count / count = ratio ✓,\n"
             "     (this_month - last_month) / last_month * 100 = percent ✓)"),
        ]
    else:
        all_checks = [
            ("O1", "O1_aggregation_matches_goal",
             "Does the aggregation function match the business question?\n"
             "    (SUM for totals/values, COUNT_DISTINCT for 'how many unique X',\n"
             "     COUNT for row counts, AVG for averages, MAX/MIN for extremes,\n"
             "     MEASURE_GROUPED for 'by X' breakdowns)"),
            ("O2", "O2_correct_column_aggregated",
             "Is the correct column being aggregated?\n"
             "    (e.g. SUM(declared_value) for a declared value total — not SUM(cargo_item_id);\n"
             "     COUNT_DISTINCT(order_id) for order count — not COUNT_DISTINCT(customer_id))"),
            ("O3", "O3_correct_source_table",
             "Is the correct source atom being used for this entity and domain?\n"
             "    Atom names follow the pattern: domain_entity_SYSTEM_type.\n"
             "    If the domain and entity in the atom name match the business question,\n"
             "    the table is correct — do not fail this check because the atom name\n"
             "    looks unfamiliar or includes a system suffix like ERP_record or WMS_state."),
            ("O4", "O4_time_filter_correct",
             "For time-scoped goals: is the time filter operator correct?\n"
             "    (MEASURE_MONTH_FIXED for this/last month, MEASURE_YTD for year-to-date,\n"
             "     MEASURE_QUARTER_FIXED for quarterly; if all_time, no time filter expected)"),
            ("O5", "O5_groupby_filter_correct",
             "For filtered or grouped goals: does the GROUP BY or WHERE field match\n"
             "    what the question is grouping or filtering by?\n"
             "    GPL formulas may have multiple WHERE clauses (e.g. WHERE date IN 2026-07\n"
             "    WHERE status=Cancelled) — both apply simultaneously, this is correct syntax.\n"
             "    Only fail this if the wrong field or wrong value is being filtered."),
            ("O6", "O6_result_unit_consistent",
             "Does the result unit match what the aggregation function produces based on column type?\n"
             "    (COUNT_DISTINCT → count, SUM(value_col) → currency, AVG(ratio) → ratio)\n"
             "    IMPORTANT: A computed result of 0.0 is still a valid value — never fail this\n"
             "    check because the oracle value is zero. Only fail if the aggregation function\n"
             "    itself produces the wrong unit type regardless of the data."),
        ]

    enabled = [(cid, key, text) for cid, key, text in all_checks if ENABLED_CHECKS.get(key, True)]

    check_lines = "\n\n".join(
        f"  {cid}. {text}" for cid, key, text in enabled
    )
    check_ids = [cid for cid, key, text in enabled]

    return check_ids, check_lines


def _is_composite_formula(formula: str) -> bool:
    """True if the formula uses MEASURE_MULTI or MEASURE_MULTI_GROUPED."""
    return "MEASURE_MULTI(" in formula or "MEASURE_MULTI_GROUPED(" in formula


def _build_composite_checklist(base_check_ids: list, base_check_lines: str) -> tuple:
    """
    For composite aterms:
      - Remove O6 (unit consistency) — always a false positive for composites
      - Add C7 (composite column consistency) — validates each (agg, col) pair
    """
    # Remove O6
    filtered_ids   = [c for c in base_check_ids if c != "O6"]
    filtered_lines = "\n\n".join(
        line for line in base_check_lines.split("\n\n")
        if not line.strip().startswith("O6")
    )

    # Add C7
    c7_text = (
        "C7. Composite column consistency: For each (aggregation, column) pair "
        "in the MEASURE_MULTI formula, is the aggregation semantically correct "
        "for that column type?\n"
        "    Rules:\n"
        "      SUM on a currency/amount/value/cost column → PASS\n"
        "      SUM on an ID or primary-key column → FAIL\n"
        "      COUNT_DISTINCT on a primary-key column → PASS\n"
        "      COUNT_DISTINCT on a currency/measure column → FAIL\n"
        "      AVG on a duration/days column → PASS\n"
        "      AVG on an ID column → FAIL\n"
        "    Only fail C7 if a specific pair is clearly wrong for its column type."
    )
    filtered_ids.append("C7")
    filtered_lines = filtered_lines + "\n\n  " + c7_text

    return filtered_ids, filtered_lines


def build_prompt(aterm: Dict) -> str:
    """Build the full prompt for workflow analysis of a single aterm."""
    kind    = aterm.get("kind", "operator")
    formula = aterm.get("formula_line", "")
    aterm_ctx = _build_aterm_context(aterm)
    check_ids, checklist = _build_checklist_for_kind(kind)

    # Part 2a+2c: Composite aterms get O6 replaced by C7
    if _is_composite_formula(formula):
        check_ids, checklist = _build_composite_checklist(check_ids, checklist)

    # Part 3: Inject composite formula type hint into aterm context
    if _is_composite_formula(formula):
        aterm_ctx = (
            aterm_ctx
            + "\nFORMULA TYPE: composite — this formula uses MEASURE_MULTI or "
            "MEASURE_MULTI_GROUPED. It returns MULTIPLE named values (a dict), "
            "NOT a single scalar. Do NOT apply single-value unit reasoning. "
            "Use C7 (composite column consistency) instead of O6 for unit checks."
        )
    n_checks = len(check_ids)
    checks_json = ", ".join(f'"{c}": true | false' for c in check_ids)
    reasoning_json = ", ".join(f'"{c}": "<one sentence>"' for c in check_ids)

    prompt = f"""You are a business metric auditor. Your job is to determine whether a compiled formula correctly answers its stated business question.

You will assess the formula using a fixed checklist. Your confidence score is computed directly from your checklist answers — it is NOT a free estimate. You must answer every check honestly.

═══════════════════════════════════════
GPL FORMULA SYNTAX REFERENCE — READ BEFORE ASSESSING
═══════════════════════════════════════
These formulas are written in GPL (Goal Programming Language), NOT SQL.
Apply the following rules strictly when evaluating every check:

ATOM NAMING CONVENTION
  Atom names follow the pattern: {{domain}}_{{entity}}_{{SYSTEM}}_{{type}}
  Examples:
    logistics_incidents_ERP_record   → domain=logistics, entity=incidents, system=ERP
    supply_chain_orders_WMS_state    → domain=supply_chain, entity=orders, system=WMS
  The atom name IS the correct source table for that entity. Never fail O3
  (source table check) solely because the atom name looks unfamiliar —
  if the domain and entity match the business question, the table is correct.

WHERE CLAUSE SYNTAX
  GPL uses multiple chained WHERE clauses, each as a separate filter:
    WHERE booked_date IN 2026-07 WHERE shipment_status=Cancelled
  This means: booked_date=July 2026 AND shipment_status=Cancelled.
  Both filters apply simultaneously. Do NOT treat the second WHERE as
  a syntax error or as overriding the first. Never fail O5 (group-by/filter
  check) because a formula has two WHERE clauses — that is correct GPL.

COUNT_DISTINCT vs COUNT
  ERP and WMS record tables often store multiple rows per entity (e.g. one
  row per status change per shipment). COUNT_DISTINCT(entity_id) is always
  the correct choice when counting unique entities. COUNT would overcount.
  Never fail O2 (correct column) or O1 (aggregation match) by arguing that
  COUNT should replace COUNT_DISTINCT on a record table.

ZERO ORACLE VALUES
  A computed result of 0.0000 means the formula returned zero for the
  current dataset — it is NOT evidence of a formula error or unit mismatch.
  Never fail O6 (result unit) because the oracle value is zero or null.
  Only fail O6 if the aggregation function itself produces the wrong type
  (e.g. COUNT_DISTINCT when the declared unit is currency).

TIME FILTER OPERATORS
  MEASURE_MONTH_FIXED  → filters to a fixed calendar month (e.g. 2026-07)
  MEASURE_QUARTER_FIXED → filters to a fixed calendar quarter (e.g. 2026-Q3)
  MEASURE_YTD          → year-to-date from Jan 1 to current date
  These are correct GPL time operators. Do not flag them as non-standard.

ATOM RELATIONSHIPS (when provided)
  Relationships show how atoms connect via shared fields (foreign key style).
  Format: from_atom.from_field -> to_atom.to_field
  Use relationships to verify:
    - For MEASURE_GROUPED: does the GROUP BY field exist on the source atom
      directly, or must it come from a related atom via a join?
      Flag if a required join is not reflected in the formula.
    - For composed metrics spanning multiple atoms: are the atoms being
      combined genuinely related, or from completely unrelated domains?
      Flag if two unrelated entities are being divided or combined.
  Do NOT flag a formula solely because relationships exist. Only flag if
  the formula requires a relationship that is missing or uses an unrelated atom.

═══════════════════════════════════════
METRIC TO ASSESS
═══════════════════════════════════════
{aterm_ctx}

═══════════════════════════════════════
CHECKLIST ({n_checks} checks)
═══════════════════════════════════════
Answer each with true (check passes) or false (check fails).

{checklist}

═══════════════════════════════════════
CONFIDENCE RULE (mandatory)
═══════════════════════════════════════
confidence = number of true answers / {n_checks}
Do NOT adjust confidence based on intuition. Compute it exactly.

verdict:
  CORRECT      if confidence >= 0.90
  QUESTIONABLE if 0.60 <= confidence < 0.90
  INCORRECT    if confidence < 0.60

═══════════════════════════════════════
RESPONSE FORMAT
═══════════════════════════════════════
Respond with ONLY a valid JSON object. No text before or after. No markdown.

{{
  "verdict": "CORRECT" | "QUESTIONABLE" | "INCORRECT",
  "confidence": <float computed from checks>,
  "checks": {{ {checks_json} }},
  "check_reasoning": {{ {reasoning_json} }},
  "flags": ["<any concern worth human attention — empty array if none>"],
  "reasoning": "<one sentence overall summary>"
}}"""

    return prompt


# ══════════════════════════════════════════════════════════════════════════════
# MODEL CALLERS
# ══════════════════════════════════════════════════════════════════════════════

def _call_claude(prompt: str) -> Dict:
    """Call Claude (Anthropic) and return parsed response dict."""
    from anthropic import Anthropic
    client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=settings.ANTHROPIC_MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    return _parse_model_response(text)


def _ensure_package(package: str, import_name: str) -> bool:
    """
    Try to import a package. If missing, install it automatically using pip.
    Returns True if the package is available after the attempt.
    """
    import importlib
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        pass
    try:
        import subprocess, sys
        log.info(f"[WA] Installing missing package: {package}")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package, "--quiet", "--break-system-packages"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            # Try without --break-system-packages (older pip)
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package, "--quiet"],
                capture_output=True, text=True, timeout=120
            )
        if result.returncode == 0:
            log.info(f"[WA] Successfully installed {package}")
            importlib.invalidate_caches()
            return True
        else:
            log.error(f"[WA] Failed to install {package}: {result.stderr}")
            return False
    except Exception as e:
        log.error(f"[WA] Auto-install failed for {package}: {e}")
        return False


def _call_openai(prompt: str, api_key: str) -> Dict:
    """Call GPT-4o (OpenAI) and return parsed response dict."""
    if not _ensure_package("openai", "openai"):
        return _error_result(
            "openai package could not be installed automatically. "
            "Please run: pip install openai  in your project environment and restart the server."
        )
    try:
        import importlib
        openai = importlib.import_module("openai")
        client = openai.OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content.strip()
        return _parse_model_response(text)
    except Exception as e:
        return _error_result(f"OpenAI call failed: {e}")


def _call_gemini(prompt: str, api_key: str) -> Dict:
    """Call Gemini 1.5 Pro (Google) and return parsed response dict."""
    if not _ensure_package("google-generativeai", "google.generativeai"):
        return _error_result(
            "google-generativeai package could not be installed automatically. "
            "Please run: pip install google-generativeai  in your project environment and restart the server."
        )
    try:
        import importlib
        genai = importlib.import_module("google.generativeai")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-pro")
        resp  = model.generate_content(prompt)
        text  = resp.text.strip()
        return _parse_model_response(text)
    except Exception as e:
        return _error_result(f"Gemini call failed: {e}")


def _parse_model_response(text: str) -> Dict:
    """Parse the model's JSON response. Robust to markdown fences."""
    cleaned = text
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1]
    if "```" in cleaned:
        cleaned = cleaned.split("```")[0]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        # Validate required fields
        verdict    = data.get("verdict", "QUESTIONABLE")
        confidence = float(data.get("confidence", 0.5))
        checks     = data.get("checks", {})
        # Recompute confidence from checks as a safety measure
        if checks:
            true_count = sum(1 for v in checks.values() if v is True)
            computed   = round(true_count / len(checks), 4)
            confidence = computed  # always use computed, not model's self-report
        return {
            "verdict":          verdict,
            "confidence":       confidence,
            "checks":           checks,
            "check_reasoning":  data.get("check_reasoning", {}),
            "flags":            data.get("flags", []),
            "reasoning":        data.get("reasoning", ""),
            "error":            None,
        }
    except Exception as e:
        return _error_result(f"Failed to parse model response: {e}. Raw: {text[:200]}")


def _error_result(msg: str) -> Dict:
    return {
        "verdict":         "ERROR",
        "confidence":      0.0,
        "checks":          {},
        "check_reasoning": {},
        "flags":           [msg],
        "reasoning":       "Analysis failed — see flags.",
        "error":           msg,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CORE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def _load_aterm(canonical_id: str) -> Optional[Dict]:
    """Load a single aterm from disk."""
    path = _paths.ATERMS_DIR / f"aterm_{canonical_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_atom_schema(atom_canonical_id: str) -> Optional[Dict]:
    """
    Load atom schema from data/atoms.json by canonical_id.
    Returns a dict with name, description, grain_description, fields list.
    """
    atoms_path = _paths.DATA_DIR / "atoms.json"
    if not atoms_path.exists():
        return None
    try:
        atoms_data = json.loads(atoms_path.read_text(encoding="utf-8"))
        atom_store = atoms_data.get("_default", atoms_data)
        for _, atom in atom_store.items():
            if atom.get("canonical_id") == atom_canonical_id:
                return atom
        return None
    except Exception as e:
        log.warning(f"[WA] Could not load atom schema for {atom_canonical_id}: {e}")
        return None


def _load_mock_data(atom_canonical_id: str, vertical: str) -> Optional[str]:
    """
    Load full mock CSV content for an atom.
    Returns the raw CSV string (all rows, all columns), or None if not found.
    Mock CSVs live at: data/mock_data/{vertical}/{atom_canonical_id}.csv
    """
    mock_path = _paths.MOCK_DATA_DIR / vertical / f"{atom_canonical_id}.csv"
    if mock_path.exists():
        try:
            return mock_path.read_text(encoding="utf-8").strip()
        except Exception as e:
            log.warning(f"[WA] Could not read mock data {mock_path}: {e}")

    # Fallback: search all vertical subdirs
    for subdir in _paths.MOCK_DATA_DIR.iterdir():
        if subdir.is_dir():
            candidate = subdir / f"{atom_canonical_id}.csv"
            if candidate.exists():
                try:
                    return candidate.read_text(encoding="utf-8").strip()
                except Exception:
                    return None
    return None


def _format_atom_schema(atom: Dict) -> str:
    """Format atom schema as a readable block for the prompt."""
    lines = []
    lines.append(f"Name:        {atom.get('name', '')}")
    lines.append(f"Description: {atom.get('description', '')}")
    lines.append(f"Grain:       {atom.get('grain_description', '')}")
    lines.append(f"Record type: {atom.get('record_type', '')}")
    fields = atom.get("fields", [])
    if fields:
        lines.append(f"Columns ({len(fields)}):")
        for f in fields:
            ftype = f.get("type", "")
            role  = f.get("role", "")
            desc  = f.get("description", "")
            role_str = f" [{role}]" if role else ""
            lines.append(f"  {f['name']:35} {ftype:12}{role_str}  {desc}")
    return "\n".join(lines)


# ── Complex aterm auto-selection ───────────────────────────────────────────────

COMPLEX_WAVES = {"A", "B", "C", "D", "E"}


def auto_select_complex_aterms(vertical: str) -> Dict:
    """
    Automatically select complex aterms for workflow analysis.
    Selects aterms matching:
      - kind == "composed"  (ratio, average, growth — always complex)
      - wave in [A,B,C,D,E] (advanced wave goals — complex rankings/comparisons)

    Returns a dict with:
      selected: list of aterm summaries
      composed_count: int
      wave_count: int
      total: int
    """
    aterms_dir = _paths.ATERMS_DIR
    composed   = []
    wave_adv   = []
    seen       = set()

    for path in sorted(aterms_dir.glob(f"aterm_{vertical}_*.json")):
        try:
            a = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        cid   = a.get("canonical_id", "")
        kind  = a.get("kind", "operator")
        wave  = str(a.get("wave", "")).strip().upper()

        if kind == "composed" and cid not in seen:
            composed.append(_aterm_summary(a))
            seen.add(cid)
        elif wave in COMPLEX_WAVES and cid not in seen:
            wave_adv.append(_aterm_summary(a))
            seen.add(cid)

    all_selected = composed + wave_adv
    return {
        "vertical":      vertical,
        "selected":      all_selected,
        "composed_count": len(composed),
        "wave_count":    len(wave_adv),
        "total":         len(all_selected),
        "canonical_ids": [a["canonical_id"] for a in all_selected],
    }


def _aterm_summary(a: Dict) -> Dict:
    """Return a lightweight summary dict for display in the UI."""
    return {
        "canonical_id": a.get("canonical_id", ""),
        "source_goal":  a.get("source_goal", ""),
        "formula_line": a.get("formula_line", ""),
        "kind":         a.get("kind", "operator"),
        "wave":         str(a.get("wave", "")),
        "ai_locked":    a.get("ai_locked", False),
        "reason":       "composed" if a.get("kind") == "composed" else f"wave_{a.get('wave','')}",
    }


def _call_model(model_key: str, prompt: str) -> Dict:
    """Dispatch to the right model caller."""
    info = MODEL_CATALOGUE.get(model_key, {})
    provider = info.get("provider", "anthropic")

    if provider == "anthropic":
        return _call_claude(prompt)

    api_key = get_api_key(provider)
    if not api_key:
        return _error_result(f"No API key found for {model_key}. Add it in Workflow Analyzer settings.")

    if provider == "openai":
        return _call_openai(prompt, api_key)
    elif provider == "google":
        return _call_gemini(prompt, api_key)

    return _error_result(f"Unknown provider: {provider}")


def analyze_single(aterm: Dict, model_keys: List[str]) -> Dict:
    """
    Run workflow analysis on a single aterm across all selected models.
    Returns a result dict with per-model verdicts and ensemble summary.
    """
    cid         = aterm.get("canonical_id", "unknown")
    source_goal = aterm.get("source_goal", "")
    formula     = aterm.get("formula_line", "")
    kind        = aterm.get("kind", "operator")

    log.info(f"[WA] Analyzing {cid} ({kind}) with models: {model_keys}")

    prompt = build_prompt(aterm)

    model_results = {}
    for model_key in model_keys:
        label = MODEL_CATALOGUE.get(model_key, {}).get("label", model_key)
        log.info(f"[WA]   → {label}")
        t0     = time.time()
        result = _call_model(model_key, prompt)
        result["model_key"]    = model_key
        result["model_label"]  = label
        result["elapsed_ms"]   = round((time.time() - t0) * 1000)
        model_results[model_key] = result

    # ── Ensemble reconciliation ────────────────────────────────────────────────
    valid_results = [r for r in model_results.values() if r.get("error") is None]

    if not valid_results:
        ensemble_confidence = 0.0
        ensemble_verdict    = "ERROR"
    else:
        confidences         = [r["confidence"] for r in valid_results]
        ensemble_confidence = round(sum(confidences) / len(confidences), 4)
        verdicts            = [r["verdict"] for r in valid_results]

        if ensemble_confidence >= PASS_THRESHOLD:
            ensemble_verdict = "PASSED"
        elif ensemble_confidence < FAIL_THRESHOLD and len(set(v for v in verdicts if v != "ERROR")) == 1:
            ensemble_verdict = "FAILED"
        else:
            ensemble_verdict = "FLAGGED"

    # Collect all flags from all models
    all_flags = []
    for r in model_results.values():
        for flag in r.get("flags", []):
            if flag and flag not in all_flags:
                all_flags.append(flag)

    return {
        "canonical_id":       cid,
        "source_goal":        source_goal,
        "formula_line":       formula,
        "kind":               kind,
        "ensemble_verdict":   ensemble_verdict,
        "ensemble_confidence": ensemble_confidence,
        "model_results":      model_results,
        "all_flags":          all_flags,
        "analyzed_at":        _now(),
        # Human review fields (default state)
        "human_review_status": "pending",
        "reviewed_by":         None,
        "reviewed_at":         None,
        "review_note":         None,
    }


def analyze_sample(vertical: str, canonical_ids: List[str], model_keys: List[str]) -> Dict:
    """
    Run workflow analysis on a sample of aterms for a vertical.
    Returns a summary dict and saves a report to disk.

    canonical_ids: the specific aterms to analyze (user-selected sample)
    model_keys: list of model keys from MODEL_CATALOGUE
    """
    if not model_keys:
        model_keys = ["claude"]

    log.info(f"[WA] Starting analysis — vertical={vertical} sample={len(canonical_ids)} models={model_keys}")

    results     = []
    errors      = []
    passed      = 0
    flagged     = 0
    failed      = 0
    error_count = 0

    for i, cid in enumerate(canonical_ids):
        log.info(f"[WA] [{i+1}/{len(canonical_ids)}] {cid}")
        aterm = _load_aterm(cid)
        if not aterm:
            errors.append({"canonical_id": cid, "error": "aterm file not found"})
            error_count += 1
            continue

        try:
            result = analyze_single(aterm, model_keys)
            results.append(result)

            verdict = result["ensemble_verdict"]
            if verdict == "PASSED":
                passed += 1
            elif verdict == "FLAGGED":
                flagged += 1
            elif verdict == "FAILED":
                failed += 1
            else:
                error_count += 1

        except Exception as e:
            log.error(f"[WA] Error analyzing {cid}: {e}")
            errors.append({"canonical_id": cid, "error": str(e)})
            error_count += 1

    summary = {
        "vertical":     vertical,
        "models_used":  model_keys,
        "total":        len(canonical_ids),
        "passed":       passed,
        "flagged":      flagged,
        "failed":       failed,
        "errors":       error_count,
        "results":      results,
        "error_details": errors,
        "run_at":       _now(),
    }

    # Save report to disk
    _save_report(vertical, summary)

    log.info(f"[WA] Done — passed={passed} flagged={flagged} failed={failed} errors={error_count}")
    return summary


def _save_report(vertical: str, summary: Dict) -> None:
    """Save analysis report to data/workflow_analysis_reports/{vertical}/"""
    reports_dir = _paths.DATA_DIR / "workflow_analysis_reports" / vertical
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"report_{timestamp}.json"
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"[WA] Report saved: {report_path}")


def load_latest_report(vertical: str) -> Optional[Dict]:
    """Load the most recent analysis report for a vertical."""
    reports_dir = _paths.DATA_DIR / "workflow_analysis_reports" / vertical
    if not reports_dir.exists():
        return None
    reports = sorted(reports_dir.glob("report_*.json"))
    if not reports:
        return None
    try:
        return json.loads(reports[-1].read_text(encoding="utf-8"))
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# HUMAN REVIEW
# ══════════════════════════════════════════════════════════════════════════════

def _decisions_path(vertical: str) -> Path:
    return _paths.DATA_DIR / f"wa_decisions_{vertical}.json"


def load_decisions(vertical: str) -> Dict:
    p = _decisions_path(vertical)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_decision(vertical: str, canonical_id: str, status: str,
                  reviewed_by: str, note: str = "") -> Dict:
    """
    Save a human review decision for a flagged aterm.
    status: "approved" | "rejected"
    Returns the saved decision dict.
    """
    if status not in ("approved", "rejected"):
        raise ValueError(f"Invalid status '{status}' — must be 'approved' or 'rejected'")

    decisions = load_decisions(vertical)
    decision  = {
        "status":      status,
        "reviewed_by": reviewed_by,
        "reviewed_at": _now(),
        "note":        note,
    }
    decisions[canonical_id] = decision

    _decisions_path(vertical).write_text(
        json.dumps(decisions, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info(f"[WA] Human decision saved — {canonical_id}: {status} by {reviewed_by}")
    return decision


# ══════════════════════════════════════════════════════════════════════════════
# ATERM LISTING (for UI subset selection)
# ══════════════════════════════════════════════════════════════════════════════

def list_aterms_for_vertical(
    vertical: str,
    filter_by: str = "all",       # "all" | "locked" | "unlocked"
    wave: Optional[str] = None,   # None = all waves
) -> List[Dict]:
    """
    Return a list of aterm summaries for the UI to display.
    Filtered by vertical, optional lock status, optional wave.
    """
    aterms_dir = _paths.ATERMS_DIR
    results    = []

    for path in sorted(aterms_dir.glob(f"aterm_{vertical}_*.json")):
        try:
            a = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        # Lock status filter
        ai_locked = a.get("ai_locked", False)
        if filter_by == "locked"   and not ai_locked:
            continue
        if filter_by == "unlocked" and ai_locked:
            continue

        # Wave filter — wave is stored as int (e.g. 5) or string (e.g. "A")
        # Compare both as strings to handle both cases
        aterm_wave = str(a.get("wave", ""))
        if wave and wave != "all" and aterm_wave != str(wave):
            continue

        results.append({
            "canonical_id":  a.get("canonical_id", ""),
            "source_goal":   a.get("source_goal", ""),
            "formula_line":  a.get("formula_line", ""),
            "kind":          a.get("kind", "operator"),
            "wave":          aterm_wave,
            "ai_locked":     ai_locked,
            "lock_method":   a.get("lock_method", ""),
            "verified":      a.get("verified", False),
        })

    return results


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
