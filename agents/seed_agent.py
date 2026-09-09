"""
agents/seed_agent.py
=====================
SeedAgent — generates the vertical seed file that drives goal generation.

The seed file is the blueprint for the entire goal space. It tells the
goal generator exactly what questions can be asked about this vertical:
  - What entities exist (orders, customers, invoices...)
  - What states/statuses each entity has (pending, shipped, delivered...)
  - What measures each entity has (order_amount, quantity...)
  - What time scopes apply (this_month, ytd, all_time...)
  - What grouping axes exist (by_region, by_segment, by_month...)
  - What KPIs are needed (conversion_rate, average_order_value...)
  - What cross-table goals are possible (from FK relationships)

INPUTS (all must exist before SeedAgent runs):
  data/atoms.json              — entity definitions + field roles
  data/field_values.json       — real enum values from discover_enums
  data/atom_relationships.json — FK relationships for cross-table goals

OUTPUT:
  data/seeds/{vertical}_seed.json

HOW IT WORKS:
  Phase 1 — Deterministic extraction (pure Python, $0):
    Read atoms + field_values + relationships
    Extract entities, measures, states, dimensions from field roles
    Build cross_domain_refs from FK relationships

  Phase 2 — AI enrichment (one Claude call per vertical):
    Claude receives the entity definitions + available KPI components
    Claude suggests: KPI formulas, time_scopes, series_axes
    KPI formulas validated against real canonical IDs (no hallucination)

  Phase 3 — Write + validate:
    Write seed file to data/seeds/{vertical}_seed.json
    Merge with existing seed if present (atoms may have changed)
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from anthropic import Anthropic

from core.config import settings
import core.paths as _paths
from services import atom_registry
from services.relationship_registry import list_relationships
from compiler.slot_constants import (
    build_canonical_id,
    infer_unit,
    clean_token,
    PLACEHOLDER_RE,
)

log    = logging.getLogger(__name__)
def _get_client():
    """Build a fresh client per call so a runtime API-key update takes effect immediately."""
    return Anthropic(api_key=settings.ANTHROPIC_API_KEY)


def _get_model():
    return settings.ANTHROPIC_MODEL

# ── Roles that represent state/status columns ──────────────────────────────────
_STATE_ROLES = {"dimension", "flag"}
_STATE_HINTS = {"status", "state", "stage", "phase"}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Loaders ────────────────────────────────────────────────────────────────────

def _load_field_values() -> Dict[str, List]:
    if not _paths.FIELD_VALUES_PATH.exists():
        return {}
    try:
        text = _paths.FIELD_VALUES_PATH.read_text(encoding="utf-8").strip()
        return json.loads(text) if text else {}
    except Exception as e:
        log.warning(f"seed_agent: could not load field_values: {e}")
        return {}


# ── Entity extraction ──────────────────────────────────────────────────────────

def _get_states(cid: str, fields: List[Dict], field_values: Dict) -> List[str]:
    """
    Extract state/status enum values for this atom.
    Only reads from fields whose name contains status/state/stage/phase.
    Values come from field_values.json (grounded in real mock data).
    """
    states: List[str] = []
    seen   = set()
    for f in fields:
        if f.get("role") not in _STATE_ROLES:
            continue
        col = f.get("name", "")
        if not any(hint in col.lower() for hint in _STATE_HINTS):
            continue
        key  = f"{cid}.{col}"
        vals = field_values.get(key, [])
        for v in vals:
            if v not in seen:
                seen.add(v)
                states.append(v)
    return states


def _build_entity(atom: Dict, field_values: Dict) -> Tuple[str, Dict]:
    """Build one entity definition from an atom. Returns (entity_name, entity_dict)."""
    cid    = atom.get("canonical_id", "")
    fields = atom.get("fields", [])
    gkeys  = atom.get("grain_keys", [])

    grain_key = gkeys[0] if gkeys else ""
    time_col  = next((f["name"] for f in fields if f.get("role") == "time"), None)
    measures  = [f["name"] for f in fields if f.get("role") == "measure"]
    states    = _get_states(cid, fields, field_values)

    # State columns are filter targets — exclude them from dimensions
    state_cols = {
        f["name"] for f in fields
        if f.get("role") in _STATE_ROLES
        and any(hint in f["name"].lower() for hint in _STATE_HINTS)
        and field_values.get(f"{cid}.{f['name']}")
    }
    dimensions = [
        f["name"] for f in fields
        if f.get("role") == "dimension" and f["name"] not in state_cols
    ]

    # Derive entity name from canonical_id
    # Format: {domain}_{entity}_{system}_{record_type}
    # Entity name = part(s) between domain prefix and system
    parts = cid.split("_")
    domain_parts = len(atom.get("domain", "").split("_"))
    # entity is parts after domain, before second-to-last (system) and last (record_type)
    entity_parts = parts[domain_parts:-2]
    entity_name  = "_".join(entity_parts) if entity_parts else parts[1] if len(parts) > 1 else cid

    return entity_name, {
        "atom":        cid,
        "record_type": atom.get("record_type", "record"),
        "grain_key":   grain_key,
        "time_col":    time_col,
        "measures":    measures,
        "states":      states,
        "dimensions":  dimensions,
    }


def _build_entities(atoms: List[Dict], field_values: Dict) -> Dict[str, Dict]:
    entities = {}
    for atom in atoms:
        name, defn = _build_entity(atom, field_values)
        entities[name] = defn
    return entities


# ── Cross-domain refs ──────────────────────────────────────────────────────────

def _build_cross_domain_refs(
    domain:          str,
    domain_atom_ids: set,
    all_atoms:       Dict[str, Dict],
    relationships:   List[Dict],
    field_values:    Dict,
) -> List[Dict]:
    """
    For relationships where one side is in this domain and the other is not,
    create a cross_domain_ref entry.
    This enables cross-table goals between different domain atoms.
    """
    refs = []
    seen = set()

    for r in relationships:
        fa = r.get("from_atom", "")
        ta = r.get("to_atom", "")
        from_in = fa in domain_atom_ids
        to_in   = ta in domain_atom_ids

        # Only care about cross-domain relationships
        if from_in == to_in:
            continue

        foreign_cid    = ta if from_in else fa
        foreign_atom   = all_atoms.get(foreign_cid, {})
        foreign_domain = foreign_atom.get("domain", "")

        if not foreign_domain or foreign_domain == domain:
            continue
        if foreign_cid in seen:
            continue
        seen.add(foreign_cid)

        # Derive entity name from foreign atom
        _, entity_defn = _build_entity(foreign_atom, field_values)
        parts          = foreign_cid.split("_")
        domain_len     = len(foreign_domain.split("_"))
        entity_parts   = parts[domain_len:-2]
        entity_name    = "_".join(entity_parts) if entity_parts else parts[1] if len(parts) > 1 else foreign_cid

        refs.append({
            "ref_domain":       foreign_domain,
            "ref_entity":       entity_name,
            "ref_atom":         foreign_cid,
            "via_relationship": {
                "from_atom":  r["from_atom"],
                "from_field": r["from_field"],
                "to_atom":    r["to_atom"],
                "to_field":   r["to_field"],
            }
        })

    return refs


# ── KPI component vocabulary ───────────────────────────────────────────────────

def _build_available_components(domain: str, entities: Dict) -> Dict[str, str]:
    """
    Build the closed vocabulary of KPI components Claude can reference.
    These are real canonical IDs that will definitely compile.
    Anti-hallucination: Claude can ONLY use names from this list in KPI formulas.
    """
    components: Dict[str, str] = {}
    for entity_name, e in entities.items():
        # Count of all entities
        components[f"{entity_name}_count"] = build_canonical_id(
            domain, entity_name, "count", unit="count"
        )
        # Count per state
        for state in e.get("states", []):
            label = f"{entity_name}_{clean_token(state)}_count"
            components[label] = build_canonical_id(
                domain, entity_name, "count", state=state, unit="count"
            )
        # Total per measure
        for measure in e.get("measures", []):
            unit  = infer_unit(measure)
            label = f"{entity_name}_{measure}"
            components[label] = build_canonical_id(
                domain, entity_name, measure, unit=unit
            )
    return components


def _resolve_kpi_components(
    kpi_definitions:     List[Dict],
    available_components: Dict[str, str],
) -> List[Dict]:
    """
    Validate and resolve KPI formula {placeholder} tokens against the
    closed component vocabulary. Drops KPIs with unresolvable components
    rather than keeping broken formulas.
    """
    def _resolve_one(token: str) -> Optional[str]:
        if token in available_components:
            return available_components[token]
        # Fuzzy fallback: token set must be subset of label set
        token_set  = set(token.split("_")) - {""}
        candidates = [
            cid for label, cid in available_components.items()
            if token_set and token_set.issubset(set(label.split("_")))
        ]
        return candidates[0] if len(candidates) == 1 else None

    resolved = []
    for kpi in kpi_definitions:
        name    = kpi.get("name", "")
        formula = kpi.get("formula", "")
        if not name or not formula:
            log.warning(f"seed_agent: dropping KPI '{name}' — missing name or formula")
            continue

        tokens = PLACEHOLDER_RE.findall(formula)
        if not tokens:
            log.warning(f"seed_agent: dropping KPI '{name}' — no {{placeholder}} tokens")
            continue

        token_map = {t: _resolve_one(t) for t in tokens}
        missing   = [t for t, cid in token_map.items() if cid is None]
        if missing:
            log.warning(
                f"seed_agent: dropping KPI '{name}' — "
                f"unresolvable components {missing}"
            )
            continue

        resolved_formula = formula
        depends_on: List[str] = []
        for token, cid in token_map.items():
            resolved_formula = resolved_formula.replace("{" + token + "}", "{" + cid + "}")
            if cid not in depends_on:
                depends_on.append(cid)

        kpi_out              = dict(kpi)
        kpi_out["formula"]   = resolved_formula
        kpi_out["depends_on"]= depends_on
        resolved.append(kpi_out)

    return resolved


# ── AI enrichment ──────────────────────────────────────────────────────────────

def _ai_enrichment(
    domain:               str,
    entities:             Dict,
    cross_domain_refs:    List[Dict],
    available_components: Dict[str, str],
) -> Dict:
    """
    Ask Claude to suggest KPIs, time_scopes, and series_axes.
    Returns dict with kpi_definitions, time_scopes, series_axes.
    Falls back to safe defaults on any error.
    """
    entity_lines = [
        f"  {name}: measures={e['measures']}, states={e['states']}, "
        f"dims={e['dimensions']}, has_time={'yes' if e['time_col'] else 'no'}"
        for name, e in entities.items()
    ]
    ref_lines = [
        f"  {r['ref_entity']} (from {r['ref_domain']}) via "
        f"{r['via_relationship']['from_field']} → {r['via_relationship']['to_field']}"
        for r in cross_domain_refs
    ]
    component_lines = [f"  {label}" for label in sorted(available_components.keys())]

    system = (
        "You are a business metrics expert. "
        "Generate seed configuration for a GPL goal programming system. "
        "Return ONLY valid JSON — no markdown, no explanation."
    )

    user = f"""Domain: {domain}

Own entities:
{chr(10).join(entity_lines) or "  none"}

Cross-domain entities available via FK relationships:
{chr(10).join(ref_lines) if ref_lines else "  none"}

Available KPI formula components (ONLY use these exact names as {{placeholder}} tokens):
{chr(10).join(component_lines) or "  none"}

Return JSON with exactly these keys:
{{
  "kpi_definitions": [
    {{
      "name": "snake_case_name",
      "description": "one sentence",
      "formula": "{{component_a}} / {{component_b}}",
      "unit": "currency|ratio|percent|days|count"
    }}
  ],
  "time_scopes": ["all_time", "this_month", "last_month", "this_quarter", "ytd"],
  "series_axes": ["by_month", "by_region", "by_segment"]
}}

Rules:
- KPI formula {{placeholders}} MUST be exact names from the component list — nothing else
- Only include time_scopes if at least one entity has a time column
- series_axes should reference real dimension columns from the entities
- Max 5 KPIs, max 6 time_scopes, max 8 series_axes"""

    try:
        response = _get_client().messages.create(
            model=_get_model(),
            max_tokens=1500,
            messages=[{"role": "user", "content": user}],
            system=system,
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        data = json.loads(text)
        log.info(
            f"seed_agent: AI enrichment — "
            f"{len(data.get('kpi_definitions',[]))} KPIs, "
            f"{len(data.get('time_scopes',[]))} time_scopes, "
            f"{len(data.get('series_axes',[]))} series_axes"
        )
        return data
    except Exception as e:
        log.warning(f"seed_agent: AI enrichment failed: {e} — using defaults")
        has_time = any(e.get("time_col") for e in entities.values())
        return {
            "kpi_definitions": [],
            "time_scopes":     ["all_time", "this_month", "last_month", "ytd"] if has_time else ["all_time"],
            "series_axes":     [],
        }


# ── Main runner ────────────────────────────────────────────────────────────────

def run(vertical: str) -> Dict[str, Any]:
    """
    Run SeedAgent for a vertical.

    Args:
        vertical: e.g. "supply_chain", "hr", "finance"

    Returns:
        {
          "status":   "success" | "error",
          "vertical": str,
          "seed_path": str,
          "entities":  int,
          "kpis":      int,
          "time_scopes": int,
          "series_axes": int,
          "seed":      dict,   ← full seed for inspection
        }
    """
    log.info(f"[SeedAgent] Starting — vertical={vertical}")

    # Load inputs
    all_atoms_list = atom_registry.get_all_atoms()
    all_atoms      = {a["canonical_id"]: a for a in all_atoms_list
                      if isinstance(a, dict) and "canonical_id" in a}
    domain_atoms   = [a for a in all_atoms_list if a.get("domain") == vertical]

    if not domain_atoms:
        return {
            "status": "error",
            "error":  f"No atoms found for vertical '{vertical}'. Run VerticalSchemaAgent first.",
        }

    field_values     = _load_field_values()
    relationships    = list_relationships()
    domain_atom_ids  = {a["canonical_id"] for a in domain_atoms}

    # Phase 1 — deterministic extraction
    entities          = _build_entities(domain_atoms, field_values)
    cross_domain_refs = _build_cross_domain_refs(
        vertical, domain_atom_ids, all_atoms, relationships, field_values
    )

    log.info(
        f"[SeedAgent] Phase 1 complete — "
        f"{len(entities)} entities, {len(cross_domain_refs)} cross-domain refs"
    )

    # Phase 2 — AI enrichment
    available_components = _build_available_components(vertical, entities)
    ai_data              = _ai_enrichment(
        vertical, entities, cross_domain_refs, available_components
    )

    # Phase 3 — resolve KPI formulas against closed vocabulary
    raw_kpis  = ai_data.get("kpi_definitions", [])
    kpis      = _resolve_kpi_components(raw_kpis, available_components)

    log.info(
        f"[SeedAgent] KPI resolution — "
        f"{len(raw_kpis)} raw → {len(kpis)} resolved"
    )

    # Merge with existing seed (atoms may have changed)
    seed_path = _paths.SEEDS_DIR / f"{vertical}_seed.json"
    existing  = {}
    if seed_path.exists():
        try:
            existing = json.loads(seed_path.read_text(encoding="utf-8"))
            log.info(f"[SeedAgent] Merging with existing {vertical}_seed.json")
        except Exception:
            pass

    merged_entities = dict(existing.get("entities", {}))
    merged_entities.update(entities)

    seed = {
        "domain":                vertical,
        "version":               "1.0",
        "generated_at":          _now(),
        "generated_from_atoms":  list(domain_atom_ids),
        "entities":              merged_entities,
        "cross_domain_refs":     cross_domain_refs,
        "kpi_definitions":       kpis,
        "time_scopes":           ai_data.get(
                                     "time_scopes",
                                     existing.get("time_scopes", ["all_time", "this_month", "ytd"])
                                 ),
        "series_axes":           ai_data.get(
                                     "series_axes",
                                     existing.get("series_axes", [])
                                 ),
    }

    seed_path.write_text(json.dumps(seed, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info(
        f"[SeedAgent] Done — "
        f"entities={len(merged_entities)}, kpis={len(kpis)}, "
        f"time_scopes={len(seed['time_scopes'])}, "
        f"series_axes={len(seed['series_axes'])} → {seed_path}"
    )

    return {
        "status":      "success",
        "vertical":    vertical,
        "seed_path":   str(seed_path),
        "entities":    len(merged_entities),
        "kpis":        len(kpis),
        "time_scopes": len(seed["time_scopes"]),
        "series_axes": len(seed["series_axes"]),
        "seed":        seed,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run SeedAgent")
    parser.add_argument("vertical", help="e.g. supply_chain")
    args = parser.parse_args()
    result = run(vertical=args.vertical)
    # Print without full seed for readability
    display = {k: v for k, v in result.items() if k != "seed"}
    print(json.dumps(display, indent=2))
    if result.get("status") == "success":
        print(f"\nSeed written to: {result['seed_path']}")
