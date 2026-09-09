"""
compiler/wizard_steps.py
=========================
Phase 5 of Branch B — the LLM Wizard.

Replaces the single large _build_llm() call with 5-8 small constrained
calls. Each step presents the LLM with a closed enum and asks it to pick
exactly one value. The LLM cannot hallucinate because every option is
derived from real schema data (atom fields, field_values.json,
canonical_index).

Steps:
  5a  table selection      — which atom/table to query
  5b  aggregation          — SUM, COUNT_DISTINCT, AVG, MAX, MIN
  5c  column               — which field to aggregate
  5d  filter field         — which dimension column to filter on (or "none")
  5e  filter value         — exact value from field_values.json (if 5d != "none")
  5f  date column          — which time field to use (if time-scoped)
  5g  time scope           — which time window (if time-scoped)
  5h  supporting metrics   — numerator/denominator CIDs (for ratio goals only)

Each step returns a single string selected from the enum.
run_wizard() orchestrates all steps and returns a wizard_state dict that
the assembler uses to build the formula_line.

Cost:  ~$0.015–0.040 per goal (5–8 calls × ~$0.003–0.005 each)
LLM:   constrained selection only — never free-form generation
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from anthropic import Anthropic
from core.config import settings

log = logging.getLogger(__name__)

# ── Time scope constants ───────────────────────────────────────────────────────

TIME_SCOPES = [
    "all_time",
    "this_month",
    "last_month",
    "this_quarter",
    "last_quarter",
    "this_year",
    "last_year",
    "ytd",
]

# Aggregation ordered by general preference (measure hint re-orders at runtime)
_AGG_ALL = ["COUNT_DISTINCT", "COUNT", "SUM", "AVG", "MAX", "MIN"]

# Noise states that mean "no filter"
_NOISE_STATES = {"all", "base", "total", ""}

# ── Client helpers ─────────────────────────────────────────────────────────────

def _client() -> Anthropic:
    return Anthropic(api_key=settings.ANTHROPIC_API_KEY)


def _model() -> str:
    return settings.ANTHROPIC_MODEL


# ── Core constrained call ──────────────────────────────────────────────────────

def _pick(
    step_name: str,
    question: str,
    options: List[str],
    context: str,
    goal_text: str,
) -> str:
    """
    Make one constrained LLM call.

    Sends a short prompt with:
      - the goal text for context
      - any relevant context (entity, slots, prior selections)
      - a single question
      - a numbered list of valid options

    The LLM must respond with exactly one option string.
    If the response is not in the enum, we default to options[0].

    Args:
        step_name:  label for logging (e.g. "5a_table")
        question:   the specific question for this step
        options:    closed enum — LLM must pick from this list
        context:    extra context lines (prior selections, atom info)
        goal_text:  the natural language goal being compiled

    Returns:
        The selected option string (always a member of options).
    """
    if not options:
        raise ValueError(f"[wizard/{step_name}] Empty options list — cannot pick")

    if len(options) == 1:
        log.debug(f"[wizard/{step_name}] Only one option — auto-selecting: {options[0]}")
        return options[0]

    numbered = "\n".join(f"{i+1}. {o}" for i, o in enumerate(options))

    prompt = f"""You are a GPL formula compiler making a single constrained selection.

GOAL: {goal_text}
{context}

QUESTION: {question}

OPTIONS (pick exactly one):
{numbered}

Reply with ONLY the exact option string, nothing else. No explanation, no punctuation."""

    try:
        response = _client().messages.create(
            model=_model(),
            max_tokens=60,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip().strip("\"'")

        # Exact match first
        if raw in options:
            log.debug(f"[wizard/{step_name}] Selected: {raw}")
            return raw

        # Case-insensitive match
        raw_lower = raw.lower()
        for opt in options:
            if opt.lower() == raw_lower:
                log.debug(f"[wizard/{step_name}] Case-matched: {opt}")
                return opt

        # Number match — LLM returned "1" meaning first option
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                log.debug(f"[wizard/{step_name}] Index-matched [{idx}]: {options[idx]}")
                return options[idx]
        except ValueError:
            pass

        # Substring match — LLM returned something like "option_name (explanation)"
        for opt in options:
            if opt in raw or raw in opt:
                log.debug(f"[wizard/{step_name}] Substring-matched: {opt}")
                return opt

        log.warning(
            f"[wizard/{step_name}] LLM response '{raw}' not in enum. "
            f"Defaulting to: {options[0]}"
        )
        return options[0]

    except Exception as e:
        log.error(f"[wizard/{step_name}] LLM call failed: {e}. Defaulting to: {options[0]}")
        return options[0]



# ── JSON-returning constrained call ───────────────────────────────────────────

def _pick_json(step_name: str, prompt: str) -> Any:
    """
    One constrained LLM call returning structured JSON.
    Used for composite column selection — multiple (agg, col) pairs in one call.
    """
    import json as _json
    try:
        response = _client().messages.create(
            model=_model(),
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = _json.loads(raw)
        log.debug(f"[wizard/{step_name}] JSON response: {parsed}")
        return parsed
    except Exception as e:
        log.warning(f"[wizard/{step_name}] JSON parse failed: {e}")
        return None


def _is_composite_goal(goal: Dict) -> bool:
    """
    True if this goal requires MEASURE_MULTI (multiple columns from same table).
    Signals: CID contains "_composite_", or goal text matches known patterns.
    """
    cid       = goal.get("canonical_id", "").lower()
    goal_text = goal.get("goal", "").lower()
    if "_composite_" in cid:
        return True
    composite_phrases = [
        "showing gross", "showing count and", "showing quantity",
        "gross amount, discount", "quantity returned, return",
        "count and freight", "gross, discount", "qty", "summary showing",
        "breakdown showing",
    ]
    return any(p in goal_text for p in composite_phrases)


# ── Step helpers ───────────────────────────────────────────────────────────────

def _dim_fields(atom: Dict) -> List[str]:
    """Dimension-role fields only — valid filter candidates."""
    return [
        f["name"] for f in atom.get("fields", [])
        if f.get("role") in ("dimension", "state", "status")
        and not f.get("fk_target_atom")
    ]


def _measure_fields(atom: Dict) -> List[str]:
    """Measure-role fields."""
    return [f["name"] for f in atom.get("fields", []) if f.get("role") == "measure"]


def _pk_fields(atom: Dict) -> List[str]:
    """Primary key fields."""
    return [f["name"] for f in atom.get("fields", []) if f.get("role") == "primary_key"]


def _time_fields(atom: Dict) -> List[str]:
    """Time-role fields."""
    return [f["name"] for f in atom.get("fields", []) if f.get("role") == "time"]


def _all_field_names(atom: Dict) -> List[str]:
    return [f["name"] for f in atom.get("fields", [])]


def _atom_fv(atom: Dict, field_values: Dict) -> Dict[str, List]:
    """
    Slice of field_values for this atom only.
    Keys: column_name → [enum values]
    """
    prefix = atom.get("canonical_id", "") + "."
    return {
        k[len(prefix):]: v
        for k, v in field_values.items()
        if k.startswith(prefix)
    }


def _ordered_agg(measure: str, atom: Dict = None) -> List[str]:
    """
    Order aggregations by appropriateness for this measure.
    Most likely correct answer appears first — encodes expert knowledge
    as enum ordering so the LLM is biased toward the right answer.

    Key distinction (from the GPL formal definition):
      measure=count             → COUNT_DISTINCT first  (counting entities)
      measure=quantity_on_hand  → SUM first             (summing numeric values)
      measure=revenue/cost      → SUM first             (monetary values)
      measure=average/avg_*     → AVG first
      measure=max_*/highest     → MAX first
      measure=min_*/lowest      → MIN first

    When atom is provided, also checks the actual field role — if the
    field is role=measure (not role=primary_key), SUM/AVG are appropriate
    even if unit=count.
    """
    from compiler.slot_constants import is_quantity_measure
    m = measure.lower()

    # Explicit count of entities → COUNT_DISTINCT
    # For record-type atoms (ERP_record, WMS_record), multiple rows exist per entity
    # so COUNT would overcount — COUNT_DISTINCT on the PK is always correct.
    # For dimension/state atoms, COUNT and COUNT_DISTINCT are equivalent (one row per entity).
    if m == "count":
        if atom and atom.get("record_type", "record") == "record":
            # Force COUNT_DISTINCT first — plain COUNT is wrong on record tables
            return ["COUNT_DISTINCT", "COUNT", "SUM", "AVG", "MAX", "MIN"]
        return ["COUNT_DISTINCT", "COUNT", "SUM", "AVG", "MAX", "MIN"]

    # Aggregation prefix hints
    if m.startswith("average") or m.startswith("avg"):
        return ["AVG", "SUM", "COUNT_DISTINCT", "MAX", "MIN", "COUNT"]
    if any(x in m for x in ("max", "highest", "largest")):
        return ["MAX", "SUM", "AVG", "COUNT_DISTINCT", "MIN", "COUNT"]
    if any(x in m for x in ("min", "lowest", "smallest")):
        return ["MIN", "SUM", "AVG", "COUNT_DISTINCT", "MAX", "COUNT"]

    # Quantity measure (numeric column to be summed, not a count of entities)
    # e.g. quantity_on_hand, total_trips_completed, payload_capacity_kg
    if is_quantity_measure(m):
        return ["SUM", "AVG", "MAX", "MIN", "COUNT_DISTINCT", "COUNT"]

    # Check atom field definition if available — if measure matches a
    # field with role=measure, SUM is appropriate
    if atom:
        measure_fields = {f["name"].lower() for f in atom.get("fields", [])
                         if f.get("role") == "measure"}
        if m in measure_fields or any(m in fn for fn in measure_fields):
            return ["SUM", "AVG", "MAX", "MIN", "COUNT_DISTINCT", "COUNT"]

    # Currency / financial values → SUM first
    return ["SUM", "COUNT_DISTINCT", "AVG", "MAX", "MIN", "COUNT"]


def _ordered_columns(atom: Dict, agg: str) -> List[str]:
    """
    Order columns by appropriateness for the chosen aggregation.
    For COUNT/COUNT_DISTINCT: pk fields first.
    For SUM/AVG/MAX/MIN: measure fields first.
    """
    pks      = _pk_fields(atom)
    measures = _measure_fields(atom)
    dims     = _dim_fields(atom)
    times    = _time_fields(atom)
    others   = [
        f["name"] for f in atom.get("fields", [])
        if f["name"] not in pks + measures + dims + times
    ]

    if agg in ("COUNT_DISTINCT", "COUNT"):
        ordered = pks + measures + dims + times + others
    else:
        ordered = measures + pks + dims + times + others

    # Deduplicate while preserving order
    seen = set()
    result = []
    for c in ordered:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


# ── Wizard Steps ───────────────────────────────────────────────────────────────

def step_5a_table(
    goal: Dict,
    atoms_all: Dict[str, Dict],
    slots: Dict
) -> Tuple[str, Dict]:
    """
    Step 5a — Table selection.

    Enum: all atom canonical_ids that match the goal's domain + entity.
    Fallback: all atoms in domain if entity match is empty.

    Returns: (selected_atom_cid, atom_dict)
    """
    goal_text = goal.get("goal", "")
    entity    = slots.get("entity", "").lower()
    domain    = slots.get("domain", "").lower()

    # Build candidate list — prefer entity match
    candidates = []
    for cid, atom in atoms_all.items():
        cid_l = cid.lower()
        if domain in cid_l and entity in cid_l:
            candidates.append(cid)

    # Fallback: domain only
    if not candidates:
        for cid in atoms_all:
            if domain in cid.lower():
                candidates.append(cid)

    # Last resort: all atoms
    if not candidates:
        candidates = list(atoms_all.keys())

    if not candidates:
        raise ValueError(f"No atoms available for entity='{entity}' domain='{domain}'")

    context = f"ENTITY: {entity}\nDOMAIN: {domain}"

    selected_cid = _pick(
        "5a_table",
        "Which table (atom) should we query to answer this goal?",
        candidates,
        context,
        goal_text,
    )

    atom = atoms_all.get(selected_cid)
    if atom is None:
        # Safety: pick first candidate
        selected_cid = candidates[0]
        atom = atoms_all[selected_cid]

    return selected_cid, atom


def step_5b_aggregation(
    goal: Dict,
    atom: Dict,
    slots: Dict,
    selected_table: str
) -> str:
    """
    Step 5b — Aggregation selection.

    Enum ordered by measure AND atom field definitions — correctly
    distinguishes measure=count (COUNT_DISTINCT) from numeric measure
    columns like quantity_on_hand (SUM).
    """
    goal_text = goal.get("goal", "")
    measure   = slots.get("measure", "count")

    options = _ordered_agg(measure, atom)
    _rec_type = atom.get("record_type", "record")
    _count_note = (
        "\nIMPORTANT: This is a RECORD-type table (multiple rows per entity). "
        "For counting goals, always use COUNT_DISTINCT on the primary key — "
        "plain COUNT will overcount because each entity has multiple rows."
        if _rec_type == "record" else ""
    )
    context = (
        f"TABLE: {selected_table}\n"
        f"MEASURE HINT: {measure}\n"
        f"RECORD TYPE: {_rec_type}{_count_note}"
    )

    return _pick(
        "5b_aggregation",
        "Which aggregation function should we use?",
        options,
        context,
        goal_text,
    )


def step_5c_column(
    goal: Dict,
    atom: Dict,
    slots: Dict,
    selected_table: str,
    agg: str
) -> str:
    """
    Step 5c — Column selection.

    Enum: real column names from atom, ordered by appropriateness for the
    chosen aggregation. LLM cannot hallucinate a column that doesn't exist.
    """
    goal_text = goal.get("goal", "")
    measure   = slots.get("measure", "count")

    options = _ordered_columns(atom, agg)
    if not options:
        options = _all_field_names(atom)
    if not options:
        raise ValueError(f"No columns in atom {selected_table}")

    context = (
        f"TABLE: {selected_table}\n"
        f"AGGREGATION: {agg}\n"
        f"MEASURE HINT: {measure}\n"
        f"ALL COLUMNS: {_all_field_names(atom)}"
    )

    return _pick(
        "5c_column",
        f"Which column should we apply {agg} to?",
        options,
        context,
        goal_text,
    )



def step_5c_multi_columns(
    goal: Dict,
    atom: Dict,
    slots: Dict,
    selected_table: str,
) -> List[Tuple[str, str]]:
    """
    Step 5c (composite variant) — Multi-column selection.

    For composite goals that need multiple aggregations from the same table.
    Makes ONE LLM call asking Claude to return ALL required (agg, col) pairs
    as structured JSON.

    Returns:
        List of (agg, column) tuples, e.g.:
        [("SUM", "gross_amount"), ("SUM", "discount_amount"), ("SUM", "net_amount")]
    """
    import json as _json

    goal_text    = goal.get("goal", "")
    measure_cols = _measure_fields(atom)
    all_cols     = _all_field_names(atom)
    all_aggs     = ["SUM", "COUNT_DISTINCT", "COUNT", "AVG", "MAX", "MIN"]

    prompt = f"""You are a GPL formula compiler selecting columns for a COMPOSITE metric.

GOAL: {goal_text}

TABLE: {selected_table}
AVAILABLE MEASURE COLUMNS: {measure_cols}
ALL COLUMNS: {all_cols}
VALID AGGREGATIONS: {all_aggs}

This goal requires MULTIPLE aggregations from the same table in one formula.
Identify ALL columns the goal explicitly asks for and the correct aggregation for each.

Rules:
- Only pick columns that EXIST in AVAILABLE MEASURE COLUMNS or ALL COLUMNS
- For amount/value/cost columns: use SUM
- For count/quantity columns: use SUM or COUNT_DISTINCT (COUNT_DISTINCT if counting entities)
- Include EVERY column the goal mentions — do not omit any
- Do NOT include columns the goal does not mention

Respond with ONLY a JSON array of objects, no explanation, no markdown:
[{{"agg": "SUM", "col": "gross_amount"}}, {{"agg": "SUM", "col": "discount_amount"}}]"""

    result = _pick_json("5c_multi_columns", prompt)

    # Parse and validate the response
    if result and isinstance(result, list):
        valid = []
        _rec_type = atom.get("record_type", "dimension")
        _pk_cols   = set(_pk_fields(atom))
        _id_patterns = ("_id", "shipment_id", "invoice_id", "return_id",
                        "order_id", "item_id", "buyer_id")
        for item in result:
            if isinstance(item, dict):
                agg = item.get("agg", "SUM").upper()
                col = item.get("col", "")
                if col not in all_cols or agg not in all_aggs:
                    continue
                # Force COUNT_DISTINCT for PK/ID columns on record-type tables
                if (agg == "COUNT" and _rec_type == "record" and
                        (col in _pk_cols or any(p in col for p in _id_patterns))):
                    agg = "COUNT_DISTINCT"
                    log.info(f"[wizard/5c_multi_columns] Upgraded COUNT → COUNT_DISTINCT for {col} on record table")
                valid.append((agg, col))
        if valid:
            log.info(f"[wizard/5c_multi_columns] Selected {len(valid)} columns: {valid}")
            return valid

    # Fallback: return all measure columns with SUM
    log.warning(f"[wizard/5c_multi_columns] JSON parse failed — falling back to all measure cols")
    return [("SUM", col) for col in (measure_cols or all_cols[:3])]


def step_5d_filter_field(
    goal: Dict,
    atom: Dict,
    slots: Dict,
    selected_table: str
) -> str:
    """
    Step 5d — Filter field selection.

    Enum: "none" + dimension/state columns only.
    LLM cannot try to filter on a numeric measure column.

    Returns "none" if no filter needed.
    """
    goal_text = goal.get("goal", "")
    state     = slots.get("state", "all")

    # If state is noise, hint toward "none" but still ask
    state_hint = f"STATE SLOT: {state}"
    if state in _NOISE_STATES:
        state_hint += " (no specific filter state — likely 'none')"

    dims    = _dim_fields(atom)
    # Also include any field with status/state/stage in name
    extras  = [
        f["name"] for f in atom.get("fields", [])
        if any(kw in f.get("name", "").lower() for kw in ("status", "state", "stage", "phase", "type", "category"))
        and f["name"] not in dims
    ]
    options = ["none"] + dims + extras

    # Deduplicate
    seen = set()
    deduped = []
    for o in options:
        if o not in seen:
            seen.add(o)
            deduped.append(o)
    options = deduped

    context = (
        f"TABLE: {selected_table}\n"
        f"{state_hint}\n"
        f"DIMENSION COLUMNS: {dims}"
    )

    return _pick(
        "5d_filter_field",
        "Should we filter rows? If yes, which column should we filter on? Pick 'none' for no filter.",
        options,
        context,
        goal_text,
    )


def step_5e_filter_value(
    goal: Dict,
    atom: Dict,
    slots: Dict,
    selected_table: str,
    filter_field: str,
    field_values: Dict
) -> Optional[str]:
    """
    Step 5e — Filter value selection.

    THE MOST IMPORTANT STEP. Enum comes exclusively from field_values.json.
    LLM cannot hallucinate a value that isn't in the actual data.

    Returns None if no values available for this field.
    """
    goal_text = goal.get("goal", "")
    state     = slots.get("state", "all")

    # Get enum from field_values for this atom + column
    fv        = _atom_fv(atom, field_values)
    values    = fv.get(filter_field, [])

    # Also try with atom canonical_id prefix directly
    if not values:
        fv_key = f"{atom.get('canonical_id', '')}.{filter_field}"
        values = field_values.get(fv_key, [])

    if not values:
        log.warning(
            f"[wizard/5e_filter_value] No field_values for "
            f"{atom.get('canonical_id')}.{filter_field}"
        )
        return None

    # Convert to strings and deduplicate
    options = list(dict.fromkeys(str(v) for v in values))

    # Order: put state-matching value first as a hint
    if state and state not in _NOISE_STATES:
        state_lower = state.lower()
        reordered = []
        remainder = []
        for o in options:
            if o.lower() == state_lower or state_lower in o.lower():
                reordered.append(o)
            else:
                remainder.append(o)
        options = reordered + remainder

    context = (
        f"TABLE: {selected_table}\n"
        f"FILTER COLUMN: {filter_field}\n"
        f"STATE HINT: {state}"
    )

    return _pick(
        "5e_filter_value",
        f"Which value should we filter '{filter_field}' on?",
        options,
        context,
        goal_text,
    )


def step_5f_date_column(
    goal: Dict,
    atom: Dict,
    slots: Dict,
    selected_table: str,
) -> Optional[str]:
    """
    Step 5f — Date column selection.

    Enum: time-role fields only.
    Skipped if no time-role fields exist.
    """
    goal_text  = goal.get("goal", "")
    time_scope = slots.get("time", "all_time")

    time_cols = _time_fields(atom)

    # Also grab any column with date/time in its name
    extras = [
        f["name"] for f in atom.get("fields", [])
        if any(kw in f.get("name", "").lower() for kw in ("date", "time", "at", "created", "updated", "timestamp"))
        and f["name"] not in time_cols
    ]

    options = time_cols + extras

    # Deduplicate
    seen = set()
    deduped = []
    for o in options:
        if o not in seen:
            seen.add(o)
            deduped.append(o)
    options = deduped

    if not options:
        return None

    if len(options) == 1:
        return options[0]

    context = (
        f"TABLE: {selected_table}\n"
        f"TIME SCOPE: {time_scope}\n"
        f"TIME COLUMNS: {options}"
    )

    return _pick(
        "5f_date_column",
        "Which date/time column should we use for the time window?",
        options,
        context,
        goal_text,
    )


def step_5g_time_scope(
    goal: Dict,
    slots: Dict,
    selected_table: str,
    atom: Dict
) -> str:
    """
    Step 5g — Time scope selection.

    Enum: valid time scopes. The slots time value is prioritised
    (appears first in the list) so the LLM is biased toward it.

    If slots already has a known time scope, auto-selects without LLM call.
    """
    goal_text    = goal.get("goal", "")
    slot_time    = slots.get("time", "all_time")

    # If slots already resolved a known time scope, use it directly
    if slot_time in TIME_SCOPES and slot_time != "all_time":
        log.debug(f"[wizard/5g_time_scope] Auto-selected from slots: {slot_time}")
        return slot_time

    # Put slot_time first as a strong hint
    options = [slot_time] if slot_time in TIME_SCOPES else []
    for ts in TIME_SCOPES:
        if ts not in options:
            options.append(ts)

    context = (
        f"TABLE: {selected_table}\n"
        f"RECORD TYPE: {atom.get('record_type', 'record')}\n"
        f"TIME HINT FROM SLOTS: {slot_time}"
    )

    return _pick(
        "5g_time_scope",
        "Which time window should this metric cover?",
        options,
        context,
        goal_text,
    )


def step_5h_support(
    goal: Dict,
    slots: Dict,
    canonical_index: Dict,
    selected_table: str,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Step 5h — Supporting metrics for ratio/composition goals.

    Only called when unit=percent or measure suggests a ratio.
    Picks numerator CID and denominator CID from canonical_index.

    Returns (numerator_cid, denominator_cid) or (None, None).
    """
    goal_text = goal.get("goal", "")
    unit      = slots.get("unit", "count")
    entity    = slots.get("entity", "").lower()
    domain    = slots.get("domain", "").lower()

    if unit != "percent" and "ratio" not in slots.get("measure", "").lower():
        return None, None

    # Filter relevant CIDs — same entity/domain, already compiled
    relevant = [
        cid for cid in canonical_index
        if domain in cid and entity in cid
    ]

    if len(relevant) < 2:
        log.debug(f"[wizard/5h_support] Not enough compiled CIDs for ratio — skipping")
        return None, None

    # Cap to avoid huge enum
    if len(relevant) > 20:
        relevant = relevant[:20]

    context = (
        f"TABLE: {selected_table}\n"
        f"UNIT: {unit} (this is a ratio/percent goal)\n"
        f"DOMAIN: {domain}, ENTITY: {entity}"
    )

    numerator_cid = _pick(
        "5h_numerator",
        "Which compiled metric should be the NUMERATOR (the specific subset being measured)?",
        relevant,
        context,
        goal_text,
    )

    denominator_cid = _pick(
        "5h_denominator",
        "Which compiled metric should be the DENOMINATOR (the total/base for the percentage)?",
        relevant,
        context,
        goal_text,
    )

    return numerator_cid, denominator_cid


# ── Main orchestrator ──────────────────────────────────────────────────────────

def run_wizard(
    goal:            Dict,
    atoms_all:       Dict[str, Dict],
    field_values:    Dict,
    canonical_index: Dict,
    hints:           List[Dict] = None,   # accepted for API compat, reserved for future use
) -> Dict[str, Any]:
    """
    Run all wizard steps for a goal and return wizard_state.

    The wizard_state dict contains every selection made, ready for the
    assembler to build the formula_line.

    Args:
        goal:            goal object (goal text + slots + canonical_id)
        atoms_all:       all atoms in the vertical {cid: atom_dict}
        field_values:    full field_values.json {atom_cid.column: [values]}
        canonical_index: already-compiled aterms {cid: {...}}

    Returns:
        wizard_state dict:
          table          — selected atom canonical_id
          atom           — full atom dict
          agg            — aggregation function string
          column         — column to aggregate
          filter_field   — column to filter on (or None)
          filter_value   — exact value string (or None)
          filter         — [{field, op, value}] or []
          date_col       — date column (or None)
          time_scope     — time scope string
          numerator_cid  — for ratio goals (or None)
          denominator_cid— for ratio goals (or None)
          record_type    — atom record_type
          steps_taken    — count of LLM calls made
          needs_dedup    — True if atom is state/snapshot type
    """
    slots      = goal.get("slots", {})
    time_scope = slots.get("time", "all_time")
    unit       = slots.get("unit", "count")
    state      = slots.get("state", "all")
    steps      = 0

    # ── Step 5a: Table ─────────────────────────────────────────────────────────
    selected_table, atom = step_5a_table(goal, atoms_all, slots)
    steps += 1

    record_type  = atom.get("record_type", "record")
    needs_dedup  = record_type in ("state", "snapshot")

    # ── Composite detection ───────────────────────────────────────────────────
    is_composite = _is_composite_goal(goal)

    # ── Step 5b: Aggregation ───────────────────────────────────────────────────
    agg = step_5b_aggregation(goal, atom, slots, selected_table)
    steps += 1

    # ── Step 5c: Column (single) or Multi-columns (composite) ─────────────────
    multi_columns = None   # list of (agg, col) tuples — set only for composite goals
    if is_composite:
        multi_columns = step_5c_multi_columns(goal, atom, slots, selected_table)
        # Use first column as the primary for fallback/logging
        column = multi_columns[0][1] if multi_columns else "id"
        steps += 1
        log.info(f"[wizard] Composite goal detected — multi_columns={multi_columns}")
    else:
        column = step_5c_column(goal, atom, slots, selected_table, agg)
        steps += 1

    # ── Step 5d: Filter field ──────────────────────────────────────────────────
    filter_field = None
    filter_value = None
    built_filter = []

    # Skip filter step if state is noise (no filter expected)
    if state in _NOISE_STATES:
        filter_field = "none"
    else:
        filter_field = step_5d_filter_field(goal, atom, slots, selected_table)
        steps += 1

    # ── Step 5e: Filter value ──────────────────────────────────────────────────
    if filter_field and filter_field != "none":
        filter_value = step_5e_filter_value(
            goal, atom, slots, selected_table, filter_field, field_values
        )
        steps += 1

        if filter_value is not None:
            built_filter = [{"field": filter_field, "op": "=", "value": filter_value}]

    # ── Step 5f: Date column ───────────────────────────────────────────────────
    date_col = None
    is_time_scoped = time_scope not in ("all_time", "alltime", "")

    if is_time_scoped or needs_dedup:
        date_col = step_5f_date_column(goal, atom, slots, selected_table)
        if date_col:
            steps += 1

    # ── Step 5g: Time scope ────────────────────────────────────────────────────
    final_time_scope = "all_time"
    if is_time_scoped:
        final_time_scope = step_5g_time_scope(goal, slots, selected_table, atom)
        steps += 1
    else:
        final_time_scope = time_scope  # "all_time"

    # ── Step 5h: Supporting metrics (ratio/percent goals only) ─────────────────
    numerator_cid   = None
    denominator_cid = None

    if unit == "percent" or "ratio" in slots.get("measure", "").lower():
        numerator_cid, denominator_cid = step_5h_support(
            goal, slots, canonical_index, selected_table
        )
        if numerator_cid:
            steps += 2

    log.info(
        f"[wizard] Completed {steps} steps for {goal.get('canonical_id', '')} | "
        f"table={selected_table} agg={agg} col={column} "
        f"filter={built_filter} time={final_time_scope}"
    )

    return {
        "table":           selected_table,
        "atom":            atom,
        "agg":             agg,
        "column":          column,
        "multi_columns":   multi_columns,
        "is_composite":    is_composite,
        "filter_field":    filter_field if filter_field != "none" else None,
        "filter_value":    filter_value,
        "filter":          built_filter,
        "date_col":        date_col,
        "time_scope":      final_time_scope,
        "numerator_cid":   numerator_cid,
        "denominator_cid": denominator_cid,
        "record_type":     record_type,
        "needs_dedup":     needs_dedup,
        "steps_taken":     steps,
    }
