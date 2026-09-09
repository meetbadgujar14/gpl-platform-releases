"""
agents/domain_goals_agent.py
=============================
Generates Waves A-E (domain expert goals) for a vertical via Claude.

Routing strategy (entity-count aware):
  <= SINGLE_CALL_THRESHOLD entities  -> one Claude call, all 5 waves together
  >  SINGLE_CALL_THRESHOLD entities  -> one Claude call per wave (5 calls)

This keeps per-call response size manageable regardless of vertical size.

Prompt is kept lean: only entity name, record_type, measures, states are sent.
Grain keys, dimensions, time_col are excluded — Claude doesn't need them to
write goal text, and excluding them cuts prompt tokens by ~60%.

Anti-hallucination validation before writing:
  - canonical_id tokens must come from real seed vocabulary only
  - depends_on CIDs must exist in Wave 1-9 set
  - No duplicates within A-E or against Wave 1-9
  - goal text must be non-empty

Writes:
  data/goals/{vertical}/wave_A.json ... wave_E.json
  data/goals/{vertical}/generation_summary.json  (updated with A-E counts)
"""

import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

sys.path.insert(0, str(Path(__file__).parent.parent))

from anthropic import Anthropic
from core.config import settings
import core.paths as _paths

log = logging.getLogger(__name__)

def _get_client():
    """Build a fresh client per call so a runtime API-key update takes effect immediately."""
    return Anthropic(api_key=settings.ANTHROPIC_API_KEY)


def _get_model():
    return settings.ANTHROPIC_MODEL

# Verticals with more entities than this get one call per wave instead of all-in-one
SINGLE_CALL_THRESHOLD = 10

_WAVE_LABELS = {
    "A": "Operational KPIs",
    "B": "Financial KPIs",
    "C": "Composite performance",
    "D": "Alert and trigger goals",
    "E": "Optimization and recommendations",
}

_WAVE_DESCRIPTIONS = {
    "A": (
        "Operational KPIs requiring domain expertise. SLA compliance rate, "
        "on-time fulfilment rate, process cycle efficiency, order-to-ship time, "
        "first-pass yield. Operational health metrics that reveal whether "
        "the business is running smoothly."
    ),
    "B": (
        "Financial KPIs with business-judgment definitions. Revenue at risk "
        "(orders on hold), order value concentration by segment, revenue mix "
        "by region, high-value order threshold counts. Financial intelligence "
        "goals that inform commercial decisions."
    ),
    "C": (
        "Composite performance goals combining multiple dimensions. "
        "Orders completed vs cancelled vs shipped breakdown, customer segment "
        "performance summary, priority-weighted order pipeline. Balanced "
        "scorecard style goals for executive overview."
    ),
    "D": (
        "Monitoring goals for specific known states that warrant operational attention. "
        "These are COUNT or SUM goals for states that already exist as enum values in the schema "
        "(e.g. 'On Hold', 'Cancelled', 'Processing') scoped to a specific time period. "
        "Example: total value of On Hold orders this quarter, count of Cancelled orders this month. "
        "Do NOT generate goals that reference undefined thresholds, alert levels, or acceptable limits — "
        "those cannot be compiled without a threshold value that does not exist in the schema."
    ),
    "E": (
        "Optimization and recommendation goals that surface actionable insights. "
        "Customer segment with highest cancellation rate, region with lowest "
        "order completion rate, optimal order priority mix."
    ),
}

# Complexity assigned by code — never by Claude
_WAVE_COMPLEXITY = {"A": "KPI", "B": "KPI", "C": "KPI", "D": "KPI", "E": "KPI"}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clean(v: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", str(v).lower().strip()).strip("_")


def _load_seed(vertical: str) -> Dict:
    path = _paths.SEEDS_DIR / f"{vertical}_seed.json"
    if not path.exists():
        raise ValueError(f"No seed file for '{vertical}'. Run SeedAgent first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _build_valid_tokens(seed: Dict) -> Dict[str, Any]:
    """Closed vocabulary extracted from seed — the only tokens Claude may use."""
    domain   = seed.get("domain", "")
    entities = seed.get("entities", {})

    valid_entities:  List[str]            = list(entities.keys())
    valid_measures:  Dict[str, List[str]] = {}
    valid_states:    Dict[str, List[str]] = {}
    valid_times      = [t for t in seed.get("time_scopes", []) if t != "all_time"]
    valid_kpi_names  = [k.get("name", "") for k in seed.get("kpi_definitions", [])]

    for name, edef in entities.items():
        valid_measures[name] = edef.get("measures", []) + ["count"]
        valid_states[name]   = edef.get("states", [])

    return {
        "domain":          domain,
        "valid_entities":  valid_entities,
        "valid_measures":  valid_measures,
        "valid_states":    valid_states,
        "valid_times":     valid_times,
        "valid_kpi_names": valid_kpi_names,
    }


def _lean_entity_block(seed: Dict) -> str:
    """
    Slim entity summary for the prompt — only what Claude needs to write goals.
    Excludes grain_key, dimensions, time_col, atom reference.
    Cuts prompt token count by ~60% vs. the full entity block.
    """
    entities = seed.get("entities", {})
    lean = {
        name: {
            "record_type": edef.get("record_type"),
            "measures":    edef.get("measures", []),
            "states":      edef.get("states", []),
        }
        for name, edef in entities.items()
    }
    return json.dumps(lean, indent=2)


def _build_prompt(
    seed: Dict,
    tokens: Dict,
    wave_19_cids: Set[str],
    waves: List[str],          # which waves this call should generate
) -> str:
    domain      = tokens["domain"]
    kpi_names   = tokens["valid_kpi_names"]
    cid_sample  = sorted(list(wave_19_cids))[:10]

    wave_specs = "\n\n".join(
        f"Wave {w} — {_WAVE_LABELS[w]}:\n{_WAVE_DESCRIPTIONS[w]}"
        for w in waves
    )

    # Build response format instruction based on which waves we're asking for
    if len(waves) == 5:
        response_format = (
            '{\n'
            '  "wave_A": [ ...goal objects... ],\n'
            '  "wave_B": [ ...goal objects... ],\n'
            '  "wave_C": [ ...goal objects... ],\n'
            '  "wave_D": [ ...goal objects... ],\n'
            '  "wave_E": [ ...goal objects... ]\n'
            '}'
        )
    else:
        w = waves[0]
        response_format = f'{{\n  "wave_{w}": [ ...goal objects... ]\n}}'

    tokens_block = json.dumps({
        "domain":         tokens["domain"],
        "valid_entities": tokens["valid_entities"],
        "valid_measures": tokens["valid_measures"],
        "valid_states":   tokens["valid_states"],
        "valid_times":    tokens["valid_times"],
    }, indent=2)

    return f"""You are generating domain expert business goals for the '{domain}' vertical.

Waves 1-9 already cover all algebraic combinations (count, total, average, growth, KPIs).
The waves you are generating require business judgment — they cannot be enumerated from grammar.
Do NOT repeat or rephrase anything in Waves 1-9.

ENTITIES (lean summary — record_type, measures, states only):
{_lean_entity_block(seed)}

KPI NAMES ALREADY IN WAVE 9 (do not repeat):
{json.dumps(kpi_names, indent=2)}

VALID TOKEN VOCABULARY (canonical_ids must use ONLY these tokens):
{tokens_block}

SAMPLE OF EXISTING CANONICAL IDs (do not duplicate):
{json.dumps(cid_sample, indent=2)}

WAVE DEFINITIONS (generate ONLY these waves):
{wave_specs}

GOAL OBJECT SCHEMA:
{{
  "goal":                "Natural language business question",
  "canonical_id":        "domain_entity_measure_state_time_unit using ONLY valid tokens",
  "depends_on":          [],
  "composition_formula": "",
  "slots": {{
    "domain": "{domain}",
    "entity": "<entity name>",
    "measure": "<measure name>",
    "state": "<exact enum value from vocabulary OR 'all' if no filter>",
    "scope": "<total|max|min|by_<field_name>>",
    "time": "<all_time|this_month|last_month|this_quarter|last_quarter|ytd>",
    "unit": "<currency|count|percent|days>",
    "series": "<scalar|by_month|by_status|by_<field_name>>"
  }}
}}

RULES:
1. canonical_id uses ONLY tokens from the vocabulary above. No invented fields.
2. depends_on: only reference CIDs from the existing sample. Use [] if none.
3. composition_formula: only if depends_on is non-empty. Otherwise "".
4. Do not repeat Waves 1-9 goals (count, total, average, growth, max/min, time-scoped, KPIs).
5. Generate only genuinely useful goals. No padding or trivial variations.
6. wave field: letter only ({"" .join(f'"{w}"' for w in waves)}).
7. CRITICAL — EVERY goal must be fully computable from the schema alone.
   The state slot must be one of the exact enum values in the vocabulary (e.g. "Completed", "On Hold").
   Do NOT generate goals that reference undefined thresholds, vague concepts, or values not in the schema.
   BAD:  "How many orders exceed a high-value threshold?" — threshold not defined in schema.
   BAD:  "Which customers have disproportionate cancellations?" — disproportionate is undefined.
   BAD:  "Orders above acceptable limit" — acceptable limit is not in the schema.
   GOOD: "How many Cancelled orders are there this quarter?" — Cancelled is a known enum value.
   GOOD: "What is the total order amount for On Hold orders?" — On Hold is a known enum value.
8. For Wave D specifically: use only goals with a specific known state value AND a specific time scope.
   Every Wave D goal must have both state (from the enum above) AND time (from time_scopes above) filled.
9. SCOPE and SERIES slots must be set correctly for the goal intent:
   scope=total    → simple aggregate (total, count, average of all rows)
   scope=max      → "highest", "largest", "top", "most", "worst performing", "over-spend"
   scope=min      → "lowest", "smallest", "least", "best value", "under-spend"
   scope=by_<X>   → "by carrier", "by region", "grouped by", "breakdown by", "per X"
   series=scalar  → single number result
   series=by_month → trend over time, monthly breakdown
   series=by_<X>  → grouped series result
   CRITICAL: Do NOT default scope to "total" when the goal asks for highest/lowest/which/ranking.
   BAD:  scope=total for "which carrier has the highest cost per km"
   GOOD: scope=max   for "which carrier has the highest cost per km"

Respond with ONLY valid JSON, no markdown, no explanation:
{response_format}"""


def _salvage_truncated_json(raw: str, waves: List[str]) -> Optional[Dict]:
    """Extract complete wave arrays from a truncated JSON response."""
    salvaged = {}
    for wave in waves:
        key     = f"wave_{wave}"
        pattern = rf'"{key}"\s*:\s*(\[.*?\])'
        match   = re.search(pattern, raw, re.DOTALL)
        if not match:
            continue
        try:
            goals = json.loads(match.group(1))
            salvaged[key] = goals
            log.info(f"[domain_goals_agent] Salvaged {len(goals)} goals from {key}")
        except json.JSONDecodeError:
            log.warning(f"[domain_goals_agent] Could not salvage {key} — array incomplete")
    return salvaged if salvaged else None


def _call_claude(
    seed: Dict,
    tokens: Dict,
    wave_19_cids: Set[str],
    waves: List[str],
) -> Dict:
    """Make one Claude call for the specified waves. Returns parsed dict."""
    prompt = _build_prompt(seed, tokens, wave_19_cids, waves)

    response = _get_client().messages.create(
        model=_get_model(),
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$",        "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning(
            f"[domain_goals_agent] JSON parse failed for waves={waves} ({e}). "
            "Attempting salvage."
        )
        salvaged = _salvage_truncated_json(raw, waves)
        if salvaged:
            return salvaged
        log.error(f"[domain_goals_agent] Salvage also failed for waves={waves}")
        return {}


def _enforce_unit_suffix(cid: str, slots: Dict) -> str:
    """
    Ensure the canonical_id ends with the unit token from slots.

    Claude sometimes appends descriptive suffixes (_rate, _share, _alert,
    _yield, _summary etc.) instead of the unit. Since the compiler decodes
    slots from the CID, the unit must be the final token.

    Strategy: if the CID does not already end with the correct unit token,
    simply append it. We never strip CID content — descriptive tokens are
    harmless and preserve readability; the compiler reads unit from slots,
    not from the CID suffix.
    """
    unit = _clean(slots.get("unit", "count"))
    last = cid.split("_")[-1]
    if last == unit:
        return cid  # already correct
    return f"{cid}_{unit}"


def _validate_and_fix(
    goal: Dict,
    wave: str,
    tokens: Dict,
    wave_19_cids: Set[str],
    seen_cids: Set[str],
) -> Optional[str]:
    """
    Validate one A-E goal. Returns None if valid, error string if it should be dropped.
    Also stamps complexity and wave from our closed enums (overwrites Claude's values).

    Validation is intentionally light for A-E goals — these waves use compound
    descriptive measure names (e.g. "on_time_fulfillment_rate") that are not in
    the seed vocabulary. Deep token-splitting would reject all of them.

    We validate what we can reliably check:
      - goal text exists
      - canonical_id exists and is non-empty
      - canonical_id starts with the correct domain prefix
      - second token (entity) is a real entity from the seed
      - not a duplicate of Wave 1-9 or another A-E goal
    """
    cid       = goal.get("canonical_id", "").strip()
    goal_text = goal.get("goal", "").strip()
    slots     = goal.get("slots", {})

    if not goal_text: return "empty goal text"
    if not cid:       return "empty canonical_id"

    # Normalise CID unit suffix BEFORE dedup — so the corrected CID is
    # what gets compared against wave_19_cids and seen_cids.
    normalised_cid = _enforce_unit_suffix(cid, slots)
    if normalised_cid != cid:
        log.info(
            f"[domain_goals_agent] CID unit suffix normalised: "
            f"{cid} → {normalised_cid}"
        )
        goal["canonical_id"] = normalised_cid
        cid = normalised_cid

    if cid in wave_19_cids: return f"duplicates Wave 1-9 CID: {cid}"
    if cid in seen_cids:    return f"duplicate within A-E: {cid}"

    # Must start with domain prefix
    domain_prefix = _clean(tokens["domain"]) + "_"
    if not cid.startswith(domain_prefix):
        return f"canonical_id must start with '{domain_prefix}', got: {cid}"

    # Second token must be a real entity
    remainder = cid[len(domain_prefix):]
    entity_token = remainder.split("_")[0] if remainder else ""
    valid_entity_tokens = {_clean(e) for e in tokens["valid_entities"]}
    if entity_token not in valid_entity_tokens:
        return (
            f"canonical_id entity token '{entity_token}' is not a known entity. "
            f"Known: {valid_entity_tokens}"
        )

    # Validate depends_on CIDs exist in wave 1-9 if referenced
    for dep in goal.get("depends_on", []):
        if dep not in wave_19_cids:
            # Don't drop — just clear the dependency to avoid broken references
            log.warning(
                f"[domain_goals_agent] {cid}: depends_on '{dep}' not in Wave 1-9, clearing"
            )
            goal["depends_on"]          = []
            goal["composition_formula"] = ""
            break

    # Stamp from our closed enums — overwrite whatever Claude put
    goal["complexity"]          = _WAVE_COMPLEXITY[wave]
    goal["wave"]                = wave
    goal["join"]                = None
    goal.setdefault("composition_formula", "")
    goal.setdefault("slots", {})

    return None  # valid


def generate_domain_goals(
    vertical:      str,
    wave_19_cids:  List[str],
    only_atom_ids: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Generate Waves A-E for a vertical.

    Routing:
      <= SINGLE_CALL_THRESHOLD entities  -> 1 Claude call (all 5 waves)
      >  SINGLE_CALL_THRESHOLD entities  -> 5 Claude calls (1 per wave)

    Args:
        vertical:      e.g. "supply_chain"
        wave_19_cids:  canonical_ids already generated in Waves 1-9
        only_atom_ids: If given, Claude only reasons about entities backed by
                        these atoms, and the result is MERGED into the
                        existing wave_A..E.json files — goals belonging to
                        atoms outside this set are kept byte-for-byte from
                        the previous run instead of being regenerated (which
                        would otherwise produce slightly different
                        canonical_ids purely from LLM non-determinism, and
                        break the compiler's skip-cache for unrelated
                        goals). If None, or if no entity maps into the
                        given atoms, falls back to the full unscoped
                        behavior — overwrite everything, as before.

    Returns:
        {status, vertical, waves_ae, total_ae_goals, dropped, calls_made}
    """
    seed   = _load_seed(vertical)
    cids19 = set(wave_19_cids)

    # Scope entities to the affected atom set, if provided. Fall back to the
    # full seed if scoping would leave nothing to generate — better to
    # regenerate everything than to silently generate nothing.
    scoped_seed  = seed
    is_scoped    = False
    if only_atom_ids:
        scoped_entities = {
            name: edef for name, edef in seed.get("entities", {}).items()
            if edef.get("atom") in only_atom_ids
        }
        if scoped_entities:
            scoped_seed = dict(seed)
            scoped_seed["entities"] = scoped_entities
            is_scoped = True
        else:
            log.warning(
                "[domain_goals_agent] only_atom_ids given but no entities "
                "matched — falling back to full unscoped generation"
            )

    entity_to_atom = {
        name: edef.get("atom") for name, edef in seed.get("entities", {}).items()
    }

    # ── Validate entity→atom mapping before generating goals ─────────────────
    # Entities whose atoms don't exist will cause no_atom_found compile errors.
    # Remove them from the seed so Claude never generates goals for them.
    from services.atom_registry import get_all_atoms
    existing_atom_ids = {a["canonical_id"] for a in get_all_atoms()}
    invalid_entities  = {
        name for name, atom_cid in entity_to_atom.items()
        if atom_cid and atom_cid not in existing_atom_ids
    }
    if invalid_entities:
        log.warning(
            f"[domain_goals_agent] Skipping {len(invalid_entities)} entities "
            f"whose atoms don't exist: {sorted(invalid_entities)}. "
            f"Run VerticalSchemaAgent first to create missing atoms."
        )
        # Remove invalid entities from seed before building prompt
        clean_entities = {
            name: edef for name, edef in scoped_seed.get("entities", {}).items()
            if name not in invalid_entities
        }
        scoped_seed = dict(scoped_seed)
        scoped_seed["entities"] = clean_entities

    tokens       = _build_valid_tokens(scoped_seed)
    entity_count = len(scoped_seed.get("entities", {}))

    use_single_call = entity_count <= SINGLE_CALL_THRESHOLD
    all_waves       = ["A", "B", "C", "D", "E"]

    log.info(
        f"[domain_goals_agent] vertical={vertical} entities={entity_count} "
        f"scoped={is_scoped} strategy={'single_call' if use_single_call else 'per_wave'}"
    )

    # Fetch raw goal lists from Claude
    raw_by_wave: Dict[str, List] = {}

    if use_single_call:
        parsed = _call_claude(scoped_seed, tokens, cids19, all_waves)
        for w in all_waves:
            raw_by_wave[w] = parsed.get(f"wave_{w}", [])
        calls_made = 1
    else:
        for w in all_waves:
            parsed = _call_claude(scoped_seed, tokens, cids19, [w])
            raw_by_wave[w] = parsed.get(f"wave_{w}", [])
        calls_made = 5

    # Validate, write wave files
    out_dir = _paths.GOALS_DIR / vertical
    out_dir.mkdir(parents=True, exist_ok=True)

    waves_ae:   Dict[str, int] = {}
    total_ae    = 0
    all_dropped = []
    seen_cids:  Set[str] = set()

    for wave in all_waves:
        key       = f"wave_{wave}"
        raw_goals = raw_by_wave.get(wave, [])

        # ── Load whatever's already on disk for this wave, and split into
        # "keep as-is" (belongs to an atom outside the affected set) vs.
        # "being replaced" (belongs to an affected atom, or scoping is off).
        kept_goals: List[Dict] = []
        if is_scoped:
            existing_path = out_dir / f"{key}.json"
            if existing_path.exists():
                try:
                    existing = json.loads(existing_path.read_text(encoding="utf-8"))
                    for g in existing.get("goals", []):
                        g_entity = g.get("slots", {}).get("entity", "")
                        g_atom   = entity_to_atom.get(g_entity)
                        # Keep it if its atom is known and NOT in the affected
                        # set. If we can't tell which atom it belongs to,
                        # keep it too — safer to preserve than to silently drop.
                        if g_atom is None or g_atom not in only_atom_ids:
                            kept_goals.append(g)
                            seen_cids.add(g.get("canonical_id", ""))
                except Exception as e:
                    log.warning(f"[domain_goals_agent] Could not read existing {key}.json "
                                f"for merge, treating as empty: {e}")

        valid   = list(kept_goals)
        dropped = []

        for g in raw_goals:
            if not isinstance(g, dict):
                dropped.append({"reason": "not a dict", "raw": str(g)[:80]})
                continue
            err = _validate_and_fix(g, wave, tokens, cids19, seen_cids)
            if err:
                dropped.append({"reason": err, "cid": g.get("canonical_id", "")})
                log.warning(f"[domain_goals_agent] {key} dropped: {err}")
            else:
                seen_cids.add(g["canonical_id"])
                valid.append(g)

        count         = len(valid)
        waves_ae[key] = count
        total_ae     += count
        all_dropped.extend(dropped)

        (out_dir / f"{key}.json").write_text(json.dumps({
            "wave":         wave,
            "goal_count":   count,
            "generated_at": _now(),
            "goals":        valid,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

        log.info(f"[domain_goals_agent] {key}: {count} valid "
                 f"({len(kept_goals)} kept unchanged, {count - len(kept_goals)} new/regenerated), "
                 f"{len(dropped)} dropped")

    # Update generation_summary.json
    summary_path = out_dir / "generation_summary.json"
    summary: Dict = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    summary["generated_at"] = _now()
    summary["total_goals"]  = summary.get("total_goals", 0) + total_ae
    existing_waves          = summary.get("waves", {})
    existing_waves.update(waves_ae)
    summary["waves"]        = existing_waves

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info(
        f"[domain_goals_agent] Done — vertical={vertical} "
        f"ae_goals={total_ae} dropped={len(all_dropped)} calls={calls_made}"
    )

    return {
        "status":         "success",
        "vertical":       vertical,
        "waves_ae":       waves_ae,
        "total_ae_goals": total_ae,
        "dropped":        all_dropped,
        "calls_made":     calls_made,
    }
