"""
agents/vocabulary_agent.py
===========================
VocabularyAgent — generates the GPL vocabulary (dialect) file for a vertical.

The dialect file is what allows the GPL runtime to understand natural language
queries. When a user types "how many pending orders this month?", the dialect
maps "pending" → canonical state, "orders" → entity, "this month" → time scope.

Without the dialect, goal generation cannot map user language to canonical IDs.

INPUTS (all must exist):
  data/seeds/{vertical}_seed.json    — entity names, states, measures, time scopes
  data/atoms.json                    — field definitions
  data/field_values.json             — real enum values (grounded in actual data)

OUTPUT:
  data/dialects/{vertical}_dialect.json

FIVE PHASES:
  Phase A — Deterministic structural extraction ($0, instant):
    Entity names → canonical + plural forms
    State values → from field_values.json (GROUNDED in real data)
    Measure columns → column hint terms
    Time scopes → from seed + natural language aliases
    Series axes → from seed

  Phase B — Gap analysis ($0):
    Find what's in the seed but not yet in vocabulary
    Rank gaps by impact (how many goals they block)

  Phase C — AI synonym expansion (Claude, one call per entity):
    Entity aliases: "order" → ["purchase", "sale", "transaction"]
    State synonyms: "Completed" → ["done", "finished", "fulfilled"]
    Measure synonyms: "order_amount" → ["revenue", "sales", "value"]
    GROUNDING: synonyms must map to canonical values that exist in data

  Phase D — Predictive vocabulary ($0):
    KPI names from seed → natural language question forms
    Cross-domain entity references

  Phase E — Validation + write:
    Deduplicate entries (highest weight wins per term)
    Grounding check (state synonyms must map to real data values)
    Measure coverage score
    Write to data/dialects/{vertical}_dialect.json
"""

import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from anthropic import Anthropic

from core.config import settings
import core.paths as _paths

log    = logging.getLogger(__name__)
def _get_client():
    """Build a fresh client per call so a runtime API-key update takes effect immediately."""
    return Anthropic(api_key=settings.ANTHROPIC_API_KEY)


def _get_model():
    return settings.ANTHROPIC_MODEL

# ── Weight conventions ─────────────────────────────────────────────────────────
W_STRUCTURAL   = 1.00
W_FIELD_VALUES = 1.00
W_AI_ENTITY    = 0.95
W_AI_STATE     = 0.90
W_AI_MEASURE   = 0.85
W_PREDICTED    = 0.70


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Loaders ────────────────────────────────────────────────────────────────────

def _load_seed(vertical: str) -> Dict:
    path = _paths.SEEDS_DIR / f"{vertical}_seed.json"
    if not path.exists():
        raise ValueError(f"No seed file for '{vertical}'. Run SeedAgent first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_field_values() -> Dict[str, List]:
    if not _paths.FIELD_VALUES_PATH.exists():
        return {}
    try:
        text = _paths.FIELD_VALUES_PATH.read_text(encoding="utf-8").strip()
        return json.loads(text) if text else {}
    except Exception:
        return {}


def _load_atoms() -> Dict[str, Dict]:
    if not _paths.ATOMS_PATH.exists():
        return {}
    try:
        raw     = json.loads(_paths.ATOMS_PATH.read_text(encoding="utf-8"))
        default = raw.get("_default", {})
        return {v["canonical_id"]: v for v in default.values()
                if isinstance(v, dict) and "canonical_id" in v}
    except Exception:
        return {}


# ── Simple English pluraliser ──────────────────────────────────────────────────

def _pluralize(word: str) -> str:
    irregulars = {
        "inventory": "inventories", "category": "categories",
        "entity": "entities", "entry": "entries",
        "company": "companies", "currency": "currencies",
        "policy": "policies", "delivery": "deliveries",
        "activity": "activities", "opportunity": "opportunities",
    }
    if word in irregulars:
        return irregulars[word]
    # Already plural — skip
    if word.endswith(("ers","ors","ies","ses","xes","ches","shes")):
        return word
    if word.endswith("s") and not word.endswith(("ss","us")):
        return word
    if word.endswith(("s", "sh", "ch", "x", "z")):
        return word + "es"
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


def _strip_unit_suffix(col: str) -> str:
    for suffix in ["_currency","_count","_score","_hours","_days","_pct",
                   "_percent","_quantity","_amount","_value","_price",
                   "_total","_subtotal","_rate","_ratio"]:
        if col.endswith(suffix):
            return col[:-len(suffix)]
    return col


# ── Vocab entry ────────────────────────────────────────────────────────────────

class VEntry:
    def __init__(self, term: str, canonical: str, weight: float, source: str):
        self.term      = term.lower().strip()
        self.canonical = canonical
        self.weight    = weight
        self.source    = source

    def to_dict(self) -> Dict:
        return {"term": self.term, "canonical": self.canonical,
                "weight": self.weight, "source": self.source}


# ── Phase A: Deterministic extraction ─────────────────────────────────────────

def _phase_a(vertical: str, seed: Dict, field_values: Dict) -> Dict[str, List[VEntry]]:
    """Extract vocabulary from seed + field_values. Pure Python, $0."""
    vocab: Dict[str, List[VEntry]] = {
        "entity": [], "state": [], "measure": [],
        "column_hints": [], "time": [], "scope": [], "kpi": [],
    }

    # Global measure terms (always present)
    for term, canonical in [
        ("count","count"),("how many","count"),("number of","count"),
        ("total","sum"),("sum","sum"),("amount","sum"),
        ("average","avg"),("avg","avg"),("mean","avg"),
        ("maximum","max"),("max","max"),("highest","max"),
        ("minimum","min"),("min","min"),("lowest","min"),
        ("rate","rate"),("ratio","ratio"),("percent","percent"),
        ("growth","growth"),("change","change"),
    ]:
        vocab["measure"].append(VEntry(term, canonical, W_STRUCTURAL, "global"))

    entities = seed.get("entities", {})

    # A1 — entity names
    for entity_name, entity_def in entities.items():
        rt = entity_def.get("record_type", "record")
        vocab["entity"].append(VEntry(entity_name, entity_name, W_STRUCTURAL, "structural"))
        plural = _pluralize(entity_name)
        if plural != entity_name:
            vocab["entity"].append(VEntry(plural, entity_name, W_STRUCTURAL, "structural"))
        # record_type-specific aliases
        if rt == "state":
            vocab["entity"].append(VEntry(f"current {entity_name}", entity_name, 0.95, "structural"))
            vocab["entity"].append(VEntry(f"active {entity_name}", entity_name, 0.90, "structural"))
        elif rt == "dimension":
            vocab["entity"].append(VEntry(f"{entity_name} list", entity_name, 0.85, "structural"))
            vocab["entity"].append(VEntry(f"{entity_name} master", entity_name, 0.85, "structural"))

    # A2 — state values from field_values (GROUNDED in real data)
    for entity_name, entity_def in entities.items():
        atom_id = entity_def.get("atom", "")
        for state in entity_def.get("states", []):
            state_lower = state.lower()
            vocab["state"].append(VEntry(state,       state, W_FIELD_VALUES, "field_values"))
            vocab["state"].append(VEntry(state_lower, state, W_FIELD_VALUES, "field_values"))
            # Compound forms
            vocab["state"].append(VEntry(f"{state_lower} {entity_name}",
                                         state, W_FIELD_VALUES, "field_values"))
            vocab["state"].append(VEntry(f"{entity_name} {state_lower}",
                                         state, 0.95, "field_values"))

    # A3 — column hints for multi-measure disambiguation
    for entity_name, entity_def in entities.items():
        measures = entity_def.get("measures", [])
        if len(measures) <= 1:
            continue
        for measure in measures:
            hint = _strip_unit_suffix(measure)
            if hint and hint not in {"total","all","base","current","value"}:
                vocab["column_hints"].append(VEntry(hint,                measure, W_STRUCTURAL, "structural"))
                vocab["column_hints"].append(VEntry(hint.replace("_"," "), measure, W_STRUCTURAL, "structural"))

    # A4 — time scopes from seed
    time_aliases = {
        "this_month":   ["this month","current month","month to date","mtd"],
        "last_month":   ["last month","previous month"],
        "this_quarter": ["this quarter","current quarter","quarter to date","qtd"],
        "last_quarter": ["last quarter","previous quarter"],
        "ytd":          ["ytd","year to date","this year","current year"],
        "this_year":    ["this year","current year","year to date","ytd"],
        "last_year":    ["last year","previous year"],
        "all_time":     ["all time","ever","total","overall","all","to date"],
        "last_30_days": ["last 30 days","past 30 days","trailing 30 days"],
        "last_7_days":  ["last 7 days","past week","last week"],
        "last_90_days": ["last 90 days","past 90 days","trailing 90 days"],
    }
    for ts in seed.get("time_scopes", []):
        vocab["time"].append(VEntry(ts,                 ts, W_STRUCTURAL, "seed"))
        vocab["time"].append(VEntry(ts.replace("_"," "), ts, W_STRUCTURAL, "seed"))
        for alias in time_aliases.get(ts, []):
            vocab["time"].append(VEntry(alias, ts, W_STRUCTURAL, "seed"))

    # A5 — series axes from seed
    for axis in seed.get("series_axes", []):
        vocab["scope"].append(VEntry(axis, axis, W_STRUCTURAL, "seed"))
        readable = axis.replace("by_","").replace("_"," ")
        if readable:
            vocab["scope"].append(VEntry(f"by {readable}",       axis, W_STRUCTURAL, "seed"))
            vocab["scope"].append(VEntry(f"per {readable}",      axis, 0.95,         "seed"))
            vocab["scope"].append(VEntry(f"grouped by {readable}",axis, W_STRUCTURAL, "seed"))

    # A6 — KPI names from seed
    for kpi in seed.get("kpi_definitions", []):
        name = kpi.get("name", "")
        if not name:
            continue
        vocab["kpi"].append(VEntry(name,                    name, W_STRUCTURAL, "seed"))
        vocab["kpi"].append(VEntry(name.replace("_"," "),   name, W_STRUCTURAL, "seed"))

    return vocab


# ── Phase B: Gap analysis ──────────────────────────────────────────────────────

def _known_terms(vocab: Dict[str, List[VEntry]]) -> set:
    return {e.term for entries in vocab.values() for e in entries}


def _phase_b(seed: Dict, vocab: Dict[str, List[VEntry]]) -> List[Dict]:
    """Find vocabulary gaps — what's in seed but not yet covered."""
    known  = _known_terms(vocab)
    gaps   = []
    for entity_name, entity_def in seed.get("entities", {}).items():
        # Missing entity aliases
        if entity_name.lower() not in known:
            gaps.append({"type":"ENTITY_ALIAS","term":entity_name,"entity":entity_name,"impact":5})
        # Missing state synonyms
        for state in entity_def.get("states", []):
            if state.lower() not in known:
                gaps.append({"type":"STATE_SYNONYM","term":state,"entity":entity_name,"impact":3})
        # Missing measure synonyms
        for measure in entity_def.get("measures", []):
            readable = measure.replace("_"," ").lower()
            if readable not in known:
                gaps.append({"type":"MEASURE_SYNONYM","term":measure,"entity":entity_name,"impact":2})
    gaps.sort(key=lambda g: g["impact"], reverse=True)
    return gaps


# ── Phase C: AI synonym expansion ─────────────────────────────────────────────

def _ai_entity_aliases(vertical: str, entity_name: str,
                        known_entities: List[str]) -> List[str]:
    """Ask Claude for natural language aliases for this entity."""
    prompt = (
        f"I am building a GPL vocabulary for the '{vertical}' business domain.\n"
        f"The entity '{entity_name}' needs natural language aliases — words and phrases "
        f"business users naturally use to refer to this concept.\n"
        f"Other entities in this domain: {known_entities}\n\n"
        f"Return ONLY a JSON array of alias strings, no other text:\n"
        f'["alias1", "alias2", ...]'
    )
    try:
        resp = _get_client().messages.create(
            model=_get_model(), max_tokens=300,
            messages=[{"role":"user","content":prompt}],
            system="You are a business vocabulary expert. Return ONLY valid JSON arrays."
        )
        text = resp.content[0].text.strip().replace("```json","").replace("```","").strip()
        aliases = json.loads(text)
        return [str(a).lower().strip() for a in aliases if a][:8]
    except Exception as e:
        log.warning(f"vocab_agent: entity alias failed for {entity_name}: {e}")
        return []


def _ai_state_synonyms(vertical: str, entity_name: str,
                        state: str, all_states: List[str]) -> List[str]:
    """Ask Claude for synonyms of a state value, grounded against known states."""
    prompt = (
        f"For '{entity_name}' in the '{vertical}' domain, the actual status value is: '{state}'\n"
        f"All actual status values: {all_states}\n\n"
        f"What natural language words/phrases would business users say to mean '{state}'?\n"
        f"IMPORTANT: synonyms must map ONLY to '{state}', not to any other status.\n"
        f"Return ONLY a JSON array, no other text (empty array if no natural synonyms):\n"
        f'["synonym1", "synonym2"]'
    )
    try:
        resp = _get_client().messages.create(
            model=_get_model(), max_tokens=200,
            messages=[{"role":"user","content":prompt}],
            system="You are a business vocabulary expert. Return ONLY valid JSON arrays."
        )
        text = resp.content[0].text.strip().replace("```json","").replace("```","").strip()
        syns = json.loads(text)
        return [str(s).lower().strip() for s in syns if s][:5]
    except Exception as e:
        log.warning(f"vocab_agent: state synonym failed for {state}: {e}")
        return []


def _ai_measure_synonyms(vertical: str, entity_name: str,
                          measure: str) -> List[str]:
    """Ask Claude for business synonyms for a measure column."""
    readable = measure.replace("_"," ")
    prompt = (
        f"For '{entity_name}' in the '{vertical}' domain, the measure column is: '{measure}'\n"
        f"What words/phrases would business users say when asking about this metric?\n"
        f"Return ONLY a JSON array, no other text:\n"
        f'["synonym1", "synonym2"]'
    )
    try:
        resp = _get_client().messages.create(
            model=_get_model(), max_tokens=200,
            messages=[{"role":"user","content":prompt}],
            system="You are a business vocabulary expert. Return ONLY valid JSON arrays."
        )
        text = resp.content[0].text.strip().replace("```json","").replace("```","").strip()
        syns = json.loads(text)
        return [str(s).lower().strip() for s in syns if s][:6]
    except Exception as e:
        log.warning(f"vocab_agent: measure synonym failed for {measure}: {e}")
        return []


def _phase_c(vertical: str, seed: Dict, vocab: Dict[str, List[VEntry]],
              field_values: Dict) -> Tuple[Dict[str, List[VEntry]], Dict]:
    """AI-assisted synonym expansion. Returns updated vocab + aliases dict.

    Incremental: loads the existing dialect's aliases block and skips any
    entity/state/measure already processed. Only new atoms pay the LLM cost.
    """
    import json as _json
    from pathlib import Path as _Path

    entities    = seed.get("entities", {})
    known       = _known_terms(vocab)
    new_entries = 0
    entity_list = list(entities.keys())

    # Load existing aliases from the current dialect file (if any)
    dialect_path = _Path(__file__).parent.parent / "data" / "dialects" / f"{vertical}_dialect.json"
    existing_aliases: Dict = {"entity_aliases": {}, "state_synonyms": {}, "measure_synonyms": {}}
    if dialect_path.exists():
        try:
            existing_dialect = _json.loads(dialect_path.read_text(encoding="utf-8"))
            existing_aliases = existing_dialect.get("aliases", existing_aliases)
        except Exception:
            pass  # Fresh run — no existing dialect

    # Seed the output aliases from what we already have
    aliases = {
        "entity_aliases":  dict(existing_aliases.get("entity_aliases", {})),
        "state_synonyms":  dict(existing_aliases.get("state_synonyms", {})),
        "measure_synonyms": dict(existing_aliases.get("measure_synonyms", {})),
    }

    for entity_name, entity_def in entities.items():
        atom_id = entity_def.get("atom", "")

        # Entity aliases — skip if already done for this entity
        if entity_name not in aliases["entity_aliases"]:
            log.info(f"[VocabAgent] Phase C: entity aliases for '{entity_name}'")
            entity_aliases = []
            for alias in _ai_entity_aliases(vertical, entity_name, entity_list):
                if alias and alias not in known:
                    vocab["entity"].append(VEntry(alias, entity_name, W_AI_ENTITY, "ai"))
                    known.add(alias)
                    entity_aliases.append(alias)
                    new_entries += 1
            if entity_aliases:
                aliases["entity_aliases"][entity_name] = entity_aliases
        else:
            log.info(f"[VocabAgent] Phase C: entity aliases for '{entity_name}' — cached, skipping")
            for alias in aliases["entity_aliases"].get(entity_name, []):
                if alias not in known:
                    vocab["entity"].append(VEntry(alias, entity_name, W_AI_ENTITY, "ai"))
                    known.add(alias)

        # State synonyms
        states = entity_def.get("states", [])
        grounded: set = set()
        for key, vals in field_values.items():
            if key.startswith(f"{atom_id}."):
                grounded.update(str(v).strip() for v in vals)
        grounded.update(s.lower() for s in states)

        for state in states:
            if state not in aliases["state_synonyms"]:
                log.info(f"[VocabAgent] Phase C: state synonyms for '{state}'")
                state_syns = []
                for syn in _ai_state_synonyms(vertical, entity_name, state, states):
                    if not syn or syn in known:
                        continue
                    if state.lower() in grounded or state in grounded:
                        vocab["state"].append(VEntry(syn, state, W_AI_STATE, "ai"))
                        known.add(syn)
                        state_syns.append(syn)
                        new_entries += 1
                if state_syns:
                    aliases["state_synonyms"][state] = state_syns
            else:
                log.info(f"[VocabAgent] Phase C: state synonyms for '{state}' — cached, skipping")
                for syn in aliases["state_synonyms"].get(state, []):
                    if syn not in known and (state.lower() in grounded or state in grounded):
                        vocab["state"].append(VEntry(syn, state, W_AI_STATE, "ai"))
                        known.add(syn)

        # Measure synonyms
        for measure in entity_def.get("measures", []):
            if measure not in aliases["measure_synonyms"]:
                log.info(f"[VocabAgent] Phase C: measure synonyms for '{measure}'")
                meas_syns = []
                for syn in _ai_measure_synonyms(vertical, entity_name, measure):
                    if not syn or syn in known:
                        continue
                    vocab["column_hints"].append(VEntry(syn, measure, W_AI_MEASURE, "ai"))
                    known.add(syn)
                    meas_syns.append(syn)
                    new_entries += 1
                if meas_syns:
                    aliases["measure_synonyms"][measure] = meas_syns
            else:
                log.info(f"[VocabAgent] Phase C: measure synonyms for '{measure}' — cached, skipping")
                for syn in aliases["measure_synonyms"].get(measure, []):
                    if syn not in known:
                        vocab["column_hints"].append(VEntry(syn, measure, W_AI_MEASURE, "ai"))
                        known.add(syn)

    log.info(f"[VocabAgent] Phase C complete — {new_entries} new entries (rest cached)")
    return vocab, aliases


# ── Phase D: Predictive vocabulary ────────────────────────────────────────────

def _phase_d(seed: Dict, vocab: Dict[str, List[VEntry]]) -> Dict[str, List[VEntry]]:
    """Add question forms for KPI names and cross-domain refs."""
    known = _known_terms(vocab)
    for kpi in seed.get("kpi_definitions", []):
        name = kpi.get("name", "")
        if not name:
            continue
        readable = name.replace("_"," ")
        for phrase in [f"what is our {readable}", f"show {readable}", f"get {readable}"]:
            if phrase not in known:
                vocab["kpi"].append(VEntry(phrase, name, W_PREDICTED, "predicted"))
    # Cross-domain entity names
    for ref in seed.get("cross_domain_refs", []):
        ref_entity = ref.get("ref_entity", "")
        if ref_entity and ref_entity.lower() not in known:
            vocab["entity"].append(VEntry(ref_entity, ref_entity, W_PREDICTED, "predicted"))
            vocab["entity"].append(VEntry(_pluralize(ref_entity), ref_entity, W_PREDICTED, "predicted"))
    return vocab


# ── Phase E: Validation + write ───────────────────────────────────────────────

def _deduplicate(entries: List[VEntry]) -> List[VEntry]:
    """Keep highest-weight entry per term."""
    seen: Dict[str, VEntry] = {}
    for e in entries:
        if e.term not in seen or e.weight > seen[e.term].weight:
            seen[e.term] = e
    return list(seen.values())


def _grounding_check(vocab: Dict[str, List[VEntry]], seed: Dict,
                      field_values: Dict) -> Tuple[Dict[str, List[VEntry]], int]:
    """Drop state entries whose canonical doesn't exist in real data."""
    violations = 0
    # Build full set of grounded state values
    grounded: set = set()
    for entity_def in seed.get("entities", {}).values():
        for s in entity_def.get("states", []):
            grounded.add(s.lower())
        atom_id = entity_def.get("atom", "")
        for key, vals in field_values.items():
            if key.startswith(f"{atom_id}."):
                grounded.update(str(v).strip().lower() for v in vals)

    cleaned = []
    for e in vocab.get("state", []):
        canonical_lower = e.canonical.lower()
        if e.source in ("field_values", "structural"):
            cleaned.append(e)
        elif canonical_lower in grounded:
            cleaned.append(e)
        else:
            log.warning(f"[VocabAgent] Grounding violation: '{e.term}' → '{e.canonical}' — dropped")
            violations += 1

    vocab["state"] = cleaned
    return vocab, violations


def _measure_coverage(vocab: Dict[str, List[VEntry]], seed: Dict) -> float:
    known = _known_terms(vocab)
    total = covered = 0
    for entity_name, entity_def in seed.get("entities", {}).items():
        total += 1
        if entity_name.lower() in known: covered += 1
        for state in entity_def.get("states", []):
            total += 1
            if state.lower() in known: covered += 1
        for measure in entity_def.get("measures", []):
            total += 1
            hint = _strip_unit_suffix(measure).lower()
            if hint in known or measure.lower() in known: covered += 1
    for ts in seed.get("time_scopes", []):
        total += 1
        if ts.lower() in known: covered += 1
    return round(covered / max(total, 1), 3)


def _phase_e(vertical: str, seed: Dict, vocab: Dict[str, List[VEntry]],
              aliases: Dict, field_values: Dict) -> Dict:
    """Validate, deduplicate, measure coverage, write dialect file."""
    # Deduplicate each slot
    for slot in vocab:
        vocab[slot] = _deduplicate(vocab[slot])

    # Grounding check
    vocab, violations = _grounding_check(vocab, seed, field_values)

    # Coverage
    coverage = _measure_coverage(vocab, seed)

    # Serialize
    vocab_serialized = {
        slot: [e.to_dict() for e in entries]
        for slot, entries in vocab.items()
    }

    meta = {
        "phase_a_entries":      sum(len(v) for v in vocab.values()),
        "grounding_violations": violations,
        "total_entries":        sum(len(v) for v in vocab.values()),
        "coverage":             coverage,
    }

    dialect = {
        "domain":              vertical,
        "version":             "2.0",
        "generated_at":        _now(),
        "generated_from_seed": f"{vertical}_seed.json",
        "coverage":            coverage,
        "vocabulary":          vocab_serialized,
        "aliases":             aliases,
        "meta":                meta,
    }

    out_path = _paths.DIALECTS_DIR / f"{vertical}_dialect.json"
    out_path.write_text(json.dumps(dialect, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info(
        f"[VocabAgent] Phase E: domain={vertical} "
        f"coverage={coverage:.1%} entries={meta['total_entries']} "
        f"violations={violations} → {out_path}"
    )
    return dialect


# ── Main runner ────────────────────────────────────────────────────────────────

def run(vertical: str, skip_ai: bool = False) -> Dict[str, Any]:
    """
    Run VocabularyAgent for a vertical.

    Args:
        vertical: e.g. "supply_chain", "hr", "finance"
        skip_ai:  If True, skip Phase C (AI expansion). Faster, cheaper,
                  but vocabulary coverage will be lower.

    Returns:
        {status, vertical, dialect_path, coverage, total_entries,
         entity_entries, state_entries, measure_entries, violations}
    """
    log.info(f"[VocabAgent] Starting — vertical={vertical} skip_ai={skip_ai}")

    seed         = _load_seed(vertical)
    field_values = _load_field_values()
    atoms        = _load_atoms()

    # Phase A
    vocab = _phase_a(vertical, seed, field_values)
    log.info(f"[VocabAgent] Phase A: {sum(len(v) for v in vocab.values())} entries")

    # Phase B
    gaps = _phase_b(seed, vocab)
    log.info(f"[VocabAgent] Phase B: {len(gaps)} gaps found")

    # Phase C
    aliases = {"entity_aliases": {}, "state_synonyms": {}, "measure_synonyms": {}}
    if not skip_ai:
        vocab, aliases = _phase_c(vertical, seed, vocab, field_values)

    # Phase D
    vocab = _phase_d(seed, vocab)
    log.info(f"[VocabAgent] Phase D: {sum(len(v) for v in vocab.values())} entries")

    # Phase E
    dialect = _phase_e(vertical, seed, vocab, aliases, field_values)

    return {
        "status":          "success",
        "vertical":        vertical,
        "dialect_path":    str(_paths.DIALECTS_DIR / f"{vertical}_dialect.json"),
        "coverage":        dialect["coverage"],
        "total_entries":   dialect["meta"]["total_entries"],
        "entity_entries":  len(dialect["vocabulary"].get("entity", [])),
        "state_entries":   len(dialect["vocabulary"].get("state", [])),
        "measure_entries": len(dialect["vocabulary"].get("measure", [])),
        "time_entries":    len(dialect["vocabulary"].get("time", [])),
        "scope_entries":   len(dialect["vocabulary"].get("scope", [])),
        "kpi_entries":     len(dialect["vocabulary"].get("kpi", [])),
        "violations":      dialect["meta"]["grounding_violations"],
        "gaps_found":      len(gaps),
        "skip_ai":         skip_ai,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run VocabularyAgent")
    parser.add_argument("vertical", help="e.g. supply_chain")
    parser.add_argument("--skip-ai", action="store_true", help="Skip Phase C AI expansion")
    args = parser.parse_args()
    result = run(vertical=args.vertical, skip_ai=args.skip_ai)
    print(json.dumps(result, indent=2))
