"""
slot_constants.py
==================
Single source of truth for the 8-slot canonical ID system.

Used by BOTH goal generation and compilation — they must stay in sync,
otherwise a goal's ID and its compiled aterm's ID could drift apart and
break the bijection property (same question → same ID → same cached answer).

The 8 slots:
  domain, entity, measure, state, scope, time, unit, series

Noise values represent "default / nothing special happening" for a slot
and are stripped from the final canonical ID to keep it short and to
ensure equivalent goals always produce the identical ID.
"""

import re
from typing import Dict

# ── Noise values stripped from canonical ID ────────────────────────────────────
NOISE_STATE  = {"all", "base", "total", ""}
NOISE_SCOPE  = {"total", "all", "base", ""}
NOISE_TIME   = {"all_time", "alltime", ""}
NOISE_SERIES = {"scalar", ""}

# Matches {token} placeholders inside a composition_formula string — used both
# when resolving a KPI's AI-suggested component labels to real canonical IDs
# (seed_generator.py) and when substituting real oracle values during
# composition (compiler/composition_engine.py). Single source of truth so the
# two stay in sync on what counts as a valid placeholder token.
PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z0-9_]+)\}")

# ── Unit inference hints ────────────────────────────────────────────────────────
CURRENCY_HINTS   = {"amount", "cost", "price", "revenue", "value", "total",
                    "balance", "fee", "tax", "profit", "sales", "spend"}
COUNT_MEASURES   = {"count"}
PERCENT_MEASURES = {"rate", "ratio", "percent", "growth", "margin"}
DAY_MEASURES     = {"duration", "days", "lead_time", "cycle_time"}

# Quantity words — numeric measure columns that are unit=count but aggregated
# with SUM (not COUNT_DISTINCT). Examples: quantity_on_hand, units_shipped,
# num_items, qty_reserved. Distinguished from measure=count (counting entities).
QUANTITY_HINTS   = {"quantity", "qty", "units", "num_items", "num_units",
                    "volume", "weight", "capacity", "stock", "inventory",
                    "headcount", "trips", "distance", "mileage", "tonnage"}


def clean_token(v: str) -> str:
    """Normalise a single slot value to a safe snake_case ID token."""
    return re.sub(r"[^a-z0-9_]", "_", str(v).lower().strip()).strip("_")


def infer_unit(measure: str) -> str:
    """
    Infer the unit (currency/percent/days/count) from a measure name.

    Returns one of: currency | percent | days | count

    Note: both entity-counting (measure=count → COUNT_DISTINCT) and
    quantity-summing (measure=quantity_on_hand → SUM) return unit=count,
    but they need different aggregations. Use infer_aggregation() or
    is_quantity_measure() to distinguish them at compile time.
    """
    m = measure.lower()
    if m in COUNT_MEASURES:
        return "count"
    if any(h in m for h in PERCENT_MEASURES):
        return "percent"
    if any(h in m for h in DAY_MEASURES):
        return "days"
    if any(h in m for h in CURRENCY_HINTS):
        return "currency"
    if any(h in m for h in QUANTITY_HINTS):
        return "count"   # quantity — numeric but unit=count, aggregated with SUM
    return "count"


def is_quantity_measure(measure: str) -> bool:
    """
    Returns True if this measure is a numeric quantity column (SUM aggregation)
    rather than an entity count (COUNT_DISTINCT aggregation).

    Examples:
      quantity_on_hand    → True  (SUM the values)
      total_trips_completed → True (SUM or AVG the values)
      count               → False (COUNT_DISTINCT the pk)
      num_orders          → False (COUNT_DISTINCT — "num" = counting entities)

    Used by the wizard Step 5b to correctly order aggregations.
    """
    m = measure.lower()
    if m in COUNT_MEASURES:
        return False
    if m.startswith("num_") or m.startswith("number_of_"):
        return False  # "num_orders" means count of orders, not sum of a value
    return any(h in m for h in QUANTITY_HINTS)


def infer_aggregation(measure: str) -> str:
    """
    Return the aggregation type for a measure name.
    Always returns uppercase to match the aggregate() function's expected values.

    Key rule: rate/percent/margin columns store a pre-computed ratio per row
    (e.g. on_time_delivery_rate = 94.7 per supplier).  Summing them produces
    nonsense (2554%).  The correct aggregation is AVG.
    """
    m = measure.lower()
    if m == "count":
        return "COUNT_DISTINCT"
    if m.startswith("max_") or m == "max":
        return "MAX"
    if m.startswith("min_") or m == "min":
        return "MIN"
    if m.startswith("average_") or m.startswith("avg_"):
        return "AVG"
    # Pre-computed ratio/rate/percent columns must be averaged, not summed.
    # A SUM of per-row rates is mathematically undefined and always > 100.
    if any(h in m for h in PERCENT_MEASURES):   # rate, ratio, percent, growth, margin
        return "AVG"
    return "SUM"


def build_canonical_id(
    domain: str,
    entity: str,
    measure: str,
    state:  str = "all",
    scope:  str = "total",
    time:   str = "all_time",
    unit:   str = "count",
    series: str = "scalar",
) -> str:
    """
    Build the 8-slot canonical ID, stripping noise (default) values.

    Example:
        build_canonical_id("sales", "order", "count", state="pending")
        → "sales_order_count_pending_count"
        (scope/time/series omitted because they're at their noise/default values)
    """
    domain_t  = clean_token(domain)
    entity_t  = clean_token(entity)
    measure_t = clean_token(measure)

    # Strip measure when it is identical to entity — composed/KPI metrics use
    # the entity name as their measure (e.g. shipment_delivery_success_rate),
    # which would otherwise produce a redundant double token in the ID.
    parts = [domain_t, entity_t]
    if measure_t != entity_t:
        parts.append(measure_t)

    if state not in NOISE_STATE:
        parts.append(clean_token(state))
    if scope not in NOISE_SCOPE:
        parts.append(clean_token(scope))
    if time not in NOISE_TIME:
        parts.append(clean_token(time))

    # Suppress unit when it is identical to measure — avoids redundant tokens
    # like "count_count" (measure=count, unit=count) or "currency_currency".
    unit_t = clean_token(unit)
    if unit_t != measure_t:
        parts.append(unit_t)

    if series not in NOISE_SERIES:
        parts.append(clean_token(series))

    return "_".join(p for p in parts if p)


def decode_canonical_id(cid: str, atom_lookup=None) -> Dict[str, str]:
    """
    Decode a canonical ID back into its slot values.

    Because noise values are stripped during encoding, this is NOT a
    perfect inverse — slots at their default value won't be present in
    the ID and will be filled back in as their noise/default value.

    This is primarily useful for compilation: given a goal's stored
    canonical_id (or one a user's question resolved to), recover enough
    slot structure to route through the decision tree.

    NOTE: In practice, goals/aterms store the full `slots` dict alongside
    the canonical_id (see goal_generator.py's _goal_obj), so decoding from
    the ID string alone is a fallback path, not the primary path.
    """
    tokens = cid.split("_")
    if len(tokens) < 3:
        return {}

    domain = tokens[0]
    entity = tokens[1]
    # measure is harder to isolate generically since slot values can
    # themselves contain underscores. Prefer using the stored `slots`
    # dict on the goal/aterm object instead of this decoder when available.
    return {
        "domain": domain,
        "entity": entity,
        "state":  "all",
        "scope":  "total",
        "time":   "all_time",
        "unit":   "count",
        "series": "scalar",
    }
