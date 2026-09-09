"""
compiler/wizard_assembler.py
==============================
Phase 6 of Branch B — the Assembler.

Takes wizard_state (all selections from Phase 5) and traverses the
decision tree to emit a formula_line string.

Zero LLM calls. Pure deterministic Python.

Decision tree mirrors dt_operator_selection_v1 logic:
  needs_dedup + all_time      → MEASURE_SNAPSHOT_DEDUPED
  needs_dedup + time_scoped   → MEASURE_SNAPSHOT_DEDUPED (with date params)
  ratio/percent               → RATIO formula using numerator/denominator aterms
  time_scoped + this/last_month → MEASURE_MONTH_FIXED
  time_scoped + quarter/year  → MEASURE_QUARTER_FIXED
  time_scoped + ytd           → MEASURE_YTD
  grouped (scope/series)      → MEASURE_GROUPED
  default                     → MEASURE
"""

import logging
from typing import Any, Dict, Optional

from compiler.operators.gpl_operators import (
    MEASURE,
    MEASURE_SNAPSHOT_DEDUPED,
    MEASURE_SNAPSHOT_DEDUPED_MONTH,
    MEASURE_SNAPSHOT_DEDUPED_QUARTER,
    MEASURE_SNAPSHOT_DEDUPED_YTD,
    MEASURE_MONTH_FIXED,
    MEASURE_QUARTER_FIXED,
    MEASURE_YTD,
    MEASURE_GROUPED,
    MEASURE_GROUPED_YTD,
    MEASURE_ROW_RATIO,
    MEASURE_MULTI,
    MEASURE_MULTI_GROUPED,
    # ── New operators (v29) ──────────────────────────────────────────────
    MEASURE_STDDEV,
    MEASURE_CV,
    MEASURE_IQR,
    MEASURE_SKEWNESS,
    MEASURE_PERCENTILE_N,
    MEASURE_MOVING_AVERAGE,
    MEASURE_RUNNING_TOTAL,
    MEASURE_LINEAR_TREND,
    MEASURE_WINDOW_RANK,
    MEASURE_PARETO_RATIO,
    MEASURE_HERFINDAHL,
    MEASURE_GINI,
    MEASURE_SET_NEW,
    MEASURE_SET_RETAINED,
    MEASURE_SET_CHURNED,
    MEASURE_COHORT_RETENTION,
    MEASURE_CORRELATION,
    build_namespace,
)
from compiler._shared import pk_col, time_col

log = logging.getLogger(__name__)

# Time scopes that map to MEASURE_MONTH_FIXED
_MONTH_SCOPES    = {"this_month", "last_month"}
# Time scopes that map to MEASURE_QUARTER_FIXED
_QUARTER_SCOPES  = {"this_quarter", "last_quarter", "this_year", "last_year"}
# Time scopes for MEASURE_YTD
_YTD_SCOPES      = {"ytd", "year_to_date"}



# ── Composite unit derivation ─────────────────────────────────────────────────

_CURRENCY_PATTERNS = ("amount", "cost", "value", "revenue", "price", "fee",
                       "salary", "income", "spend", "credit", "discount", "tax",
                       "gross", "net", "freight", "insurance")
_COUNT_PATTERNS    = ("_id", "count", "qty", "quantity", "num_", "number",
                       "total_items", "units")
_DAYS_PATTERNS     = ("days", "hours", "duration", "age", "lead_time")
_RATIO_PATTERNS    = ("rate", "ratio", "percent", "pct", "utilization",
                       "utilisation", "efficiency")


def _col_unit(col: str, agg: str) -> str:
    """Infer the unit of a single (agg, col) pair from name patterns."""
    c = col.lower()
    a = agg.upper()
    if a in ("COUNT", "COUNT_DISTINCT"):
        return "count"
    if any(p in c for p in _DAYS_PATTERNS):
        return "days"
    if any(p in c for p in _RATIO_PATTERNS):
        return "ratio"
    if any(p in c for p in _CURRENCY_PATTERNS):
        return "currency"
    if any(p in c for p in _COUNT_PATTERNS):
        return "count"
    # Default: trust the agg
    return "currency" if a == "SUM" else "count"


def _derive_composite_unit(multi_columns) -> str:
    """
    Derive the most accurate unit for a MEASURE_MULTI result.

    Rules:
      - All columns same unit → use that unit
      - Mixed units           → "composite" (genuinely mixed, no better term)
    """
    if not multi_columns:
        return "composite"
    units = {_col_unit(col, agg) for agg, col in multi_columns}
    if len(units) == 1:
        return units.pop()
    return "composite"

def assemble(
    wizard_state: Dict[str, Any],
    goal:         Dict,
    csv_root:     str,
    canonical_index: Dict,
) -> Dict[str, Any]:
    """
    Build and execute a formula from wizard_state.

    Args:
        wizard_state:    output of run_wizard() — all slot selections
        goal:            original goal object (for slots + text)
        csv_root:        path to mock data CSVs
        canonical_index: compiled aterms (for ratio lookups)

    Returns:
        {
          oracle_value:  float,
          formula_line:  str,
          operator_used: str,
        }

    Raises:
        ValueError if formula execution fails or oracle is invalid.
    """
    atom         = wizard_state["atom"]
    table        = wizard_state["table"]
    agg          = wizard_state["agg"]
    column       = wizard_state["column"]
    filters      = wizard_state["filter"]
    date_col     = wizard_state["date_col"]
    time_scope   = wizard_state["time_scope"]
    needs_dedup  = wizard_state["needs_dedup"]
    num_cid      = wizard_state.get("numerator_cid")
    den_cid      = wizard_state.get("denominator_cid")
    multi_columns = wizard_state.get("multi_columns")   # list[(agg,col)] or None
    is_composite  = wizard_state.get("is_composite", False)

    slots        = goal.get("slots", {})
    unit         = slots.get("unit", "count")
    scope        = slots.get("scope", "total")
    series       = slots.get("series", "scalar")

    # ── scope="average" overrides agg to AVG ──────────────────────────────────
    if scope == "average" and agg not in ("AVG",):
        log.info(f"[assembler] scope='average' → overriding agg {agg!r} → 'AVG'")
        agg = "AVG"

    # ── agg_override slot: goal can force a specific aggregation ──────────────
    _agg_override = slots.get("agg_override", "")
    if _agg_override and _agg_override != agg:
        log.info(f"[assembler] agg_override slot: {agg!r} → {_agg_override!r}")
        agg = _agg_override

    # ── count_all_rows: use COUNT(*) — all rows, not COUNT_DISTINCT(pk) ────────
    # Used when the goal needs a row count denominator for average computations.
    if slots.get("count_all_rows"):
        agg    = "COUNT"
        column = None   # COUNT(*) — no column needed
        log.info("[assembler] count_all_rows=True → COUNT(*) all rows")


    atom_cid     = atom.get("canonical_id", table)
    grain_keys   = atom.get("grain_keys") or [pk_col(atom)]
    dedup_sort   = atom.get("dedup_sort_col") or date_col or time_col(atom) or ""

    is_time_scoped = time_scope not in ("all_time", "alltime", "")
    # ── scope="row_ratio": per-row ratio between two columns ──────────────────
    if scope == "row_ratio":
        _num_col   = slots.get("ratio_numerator", column)
        _den_col   = slots.get("ratio_denominator", "")
        _label_col = slots.get("ratio_label_col", None)
        if not _den_col:
            raise ValueError(f"[assembler] scope='row_ratio' requires ratio_denominator slot")
        result = MEASURE_ROW_RATIO(
            atom_cid, _num_col, _den_col,
            label_col=_label_col,
            filters=filters,
            top_n=10,
            descending=True,
        )
        log.info(f"[assembler] MEASURE_ROW_RATIO: {_num_col}/{_den_col} label={_label_col}")
        _ef = (
            f"MEASURE_ROW_RATIO('{atom_cid}', '{_num_col}', '{_den_col}', "
            f"label_col={_label_col!r}, filters={repr(filters)}, top_n=10, descending=True)"
        )
        return _result(result, "MEASURE_ROW_RATIO", _ef)
    is_grouped     = (
        (scope not in ("total", "all", "base", ""))
        and scope.startswith("by_")
    ) or (
        series not in ("scalar", "")
        and series.startswith("by_")
    )

    # ── CID-based grouping detection (slot fallback) ───────────────────────────
    # If slots missed the grouping intent but the CID or goal text reveals it,
    # detect it here and resolve the correct group_by column from atom fields.
    # Signals: CID contains "_by_" segment, or goal text contains "which"/"by"/"breakdown"
    _cid = goal.get("canonical_id", "")
    _goal_text = goal.get("goal", "").lower()
    _by_signals = ["which ", "breakdown", "composition", "by employment", "by carrier",
                   "by state", "by type", "by category", "by driver", "by vehicle",
                   "per carrier", "per driver", "each state", "each type"]

    if not is_grouped:
        # Check CID for _by_ segment
        _cid_parts = _cid.split("_")
        _by_idx = None
        for i, part in enumerate(_cid_parts):
            if part == "by" and i + 1 < len(_cid_parts):
                _by_idx = i
                break

        # Check goal text signals
        _goal_signals = any(s in _goal_text for s in _by_signals)

        if _by_idx is not None or _goal_signals:
            # Try to resolve the group_by column from the atom's dimension fields
            _dim_fields = [f["name"] for f in atom.get("fields", [])
                          if f.get("role") in ("dimension", "state", "status")]

            _group_col = None

            # Strategy 1: match CID segment after "by" against atom field names
            if _by_idx is not None:
                _by_segment = "_".join(_cid_parts[_by_idx + 1:])  # e.g. "state", "carrier", "employment_type"
                # Try progressively shorter suffixes
                for length in range(len(_cid_parts) - _by_idx - 1, 0, -1):
                    _candidate = "_".join(_cid_parts[_by_idx + 1: _by_idx + 1 + length])
                    for dim in _dim_fields:
                        if _candidate in dim or dim.endswith(_candidate):
                            _group_col = dim
                            break
                    if _group_col:
                        break

            # Strategy 2: match goal text keywords against field names
            if not _group_col:
                _kw_map = {
                    "employment type": "employment_status",
                    "employment_type": "employment_status",
                    "carrier": "carrier_id",
                    "state": "shipment_status",
                    "incident state": "incident_status",
                    "shipment state": "shipment_status",
                    "driver type": "employment_status",
                }
                for kw, field_hint in _kw_map.items():
                    if kw in _goal_text:
                        for dim in _dim_fields:
                            if field_hint in dim or dim == field_hint:
                                _group_col = dim
                                break
                        if _group_col:
                            break

            # Strategy 3: pick the most semantically relevant status/state dimension
            if not _group_col:
                for dim in _dim_fields:
                    if any(kw in dim for kw in ("status", "state", "type", "category")):
                        _group_col = dim
                        break

            if _group_col:
                # Override is_grouped and inject the scope
                is_grouped = True
                scope = f"by_{_group_col}"
                log.info(
                    f"[assembler] CID/goal grouping detected for '{_cid}' → "
                    f"group_by={_group_col} (slots had scope='{slots.get('scope','?')}')"
                )
            else:
                log.warning(
                    f"[assembler] Grouping signals detected for '{_cid}' "
                    f"but could not resolve group_by column from atom fields: {_dim_fields}"
                )

    # ── Composite multi-column goals ───────────────────────────────────────────
    if is_composite and multi_columns:
        # Derive accurate unit from actual column types — not from slots
        derived_unit = _derive_composite_unit(multi_columns)
        log.info(f"[assembler] Composite unit derived: {derived_unit} from {multi_columns}")

        if is_grouped:
            group_axis = scope[3:] if scope.startswith("by_") else (
                series[3:] if series.startswith("by_") else None
            )
            if not group_axis:
                group_axis = next(
                    (f["name"] for f in atom.get("fields", [])
                     if f.get("role") in ("dimension", "state", "status")),
                    "id"
                )
            result = MEASURE_MULTI_GROUPED(
                atom_cid,
                columns=multi_columns,
                group_by_col=group_axis,
                filters=filters or None,
            )
            log.info(f"[assembler] MEASURE_MULTI_GROUPED cols={multi_columns} group_by={group_axis}")
            r = _result(result, "MEASURE_MULTI_GROUPED", result["formula_line"])
            r["derived_unit"] = derived_unit
            return r
        else:
            result = MEASURE_MULTI(
                atom_cid,
                columns=multi_columns,
                filters=filters or None,
            )
            log.info(f"[assembler] MEASURE_MULTI cols={multi_columns}")
            r = _result(result, "MEASURE_MULTI", result["formula_line"])
            r["derived_unit"] = derived_unit
            return r

    # ── Ratio / percent goals ──────────────────────────────────────────────────
    # ── Percent / ratio goals ─────────────────────────────────────────────────
    if unit == "percent":
        # Check if a real state filter exists — if so, compute the ratio
        # directly (filtered numerator / unfiltered denominator) instead of
        # using pre-compiled aterms which don't have the filter applied.
        _state = slots.get("state", "")
        _NOISE = {"all", "base", "total", ""}
        _has_state_filter = bool(_state) and _state not in _NOISE and filters

        if _has_state_filter:
            # Direct ratio: filtered measure / total measure * 100
            # If slots["measure"] is a numeric value/currency column (not the PK),
            # use SUM(measure_col). Otherwise fall back to COUNT_DISTINCT(pk).
            from compiler.operators.gpl_operators import MEASURE as _MEASURE_DIRECT
            _pk = (atom.get("grain_keys") or [None])[0] or column
            _measure_slot = slots.get("measure", "")
            _atom_fields = {f["name"]: f for f in atom.get("fields", [])}
            _measure_field = _atom_fields.get(_measure_slot, {})
            _is_value_col = (
                _measure_slot
                and _measure_slot != _pk
                and _measure_field.get("type") in ("number", "float", "decimal", "currency")
                and not any(kw in _measure_slot for kw in ("_id", "_count", "_qty", "_num"))
            )
            if _is_value_col:
                _ratio_agg = "SUM"
                _ratio_col = _measure_slot
                log.info(f"[assembler] Direct ratio: using SUM({_ratio_col}) (value column)")
            else:
                _ratio_agg = "COUNT_DISTINCT"
                _ratio_col = _pk
                log.info(f"[assembler] Direct ratio: using COUNT_DISTINCT({_ratio_col}) (pk column)")

            # denominator_measure slot allows a different column for the denominator
            # e.g. numerator=SUM(credit_amount) / denominator=SUM(return_value)
            _den_measure = slots.get("denominator_measure", "")
            _den_col = _den_measure if _den_measure and _den_measure in _atom_fields else _ratio_col
            _den_agg  = "SUM" if (_den_col != _pk) else _ratio_agg
            if _den_col != _ratio_col:
                log.info(f"[assembler] Direct ratio: denominator overridden to SUM({_den_col}) via denominator_measure slot")

            # denominator_same_filter=True passes the same state filter to denominator
            # e.g. SUM(credit_amount, Completed) / SUM(return_value, Completed)
            _den_filters = filters if slots.get("denominator_same_filter") else None
            if _den_filters:
                log.info(f"[assembler] Direct ratio: denominator uses same filters as numerator")

            if is_time_scoped and date_col:
                if time_scope in _MONTH_SCOPES:
                    _which = "last_month" if "last" in time_scope else "this_month"
                    num_r = MEASURE_MONTH_FIXED(atom_cid, date_col, _ratio_agg, _ratio_col, _which, filters)
                    den_r = MEASURE_MONTH_FIXED(atom_cid, date_col, _den_agg,   _den_col,   _which, _den_filters)
                elif time_scope in _QUARTER_SCOPES:
                    num_r = MEASURE_QUARTER_FIXED(atom_cid, date_col, _ratio_agg, _ratio_col, time_scope, filters)
                    den_r = MEASURE_QUARTER_FIXED(atom_cid, date_col, _den_agg,   _den_col,   time_scope, _den_filters)
                elif time_scope in _YTD_SCOPES:
                    num_r = MEASURE_YTD(atom_cid, date_col, _ratio_agg, _ratio_col, filters)
                    den_r = MEASURE_YTD(atom_cid, date_col, _den_agg,   _den_col,   _den_filters)
                else:
                    num_r = _MEASURE_DIRECT(atom_cid, _ratio_agg, _ratio_col, filters)
                    den_r = _MEASURE_DIRECT(atom_cid, _den_agg,   _den_col,   _den_filters)
            else:
                num_r = _MEASURE_DIRECT(atom_cid, _ratio_agg, _ratio_col, filters)
                den_r = _MEASURE_DIRECT(atom_cid, _den_agg,   _den_col,   _den_filters)

            _num_v  = float(num_r["oracle_value"])
            _den_v  = float(den_r["oracle_value"])
            _oracle = round((_num_v / _den_v * 100) if _den_v else 0.0, 6)
            _ef_num = num_r["formula_line"]
            _ef_den = den_r["formula_line"]
            _formula = f"RATIO: {_ef_num} / {_ef_den} * 100"
            _exec    = f"(({_ef_num}) / ({_ef_den}) * 100) if ({_ef_den}) else 0.0"
            log.info(f"[assembler] Direct state-filtered RATIO: {_oracle:.2f}% (state={_state})")
            return {
                "oracle_value":  _oracle,
                "formula_line":  _formula,
                "exec_formula":  _exec,
                "operator_used": "RATIO_FILTERED",
            }

        if num_cid and den_cid:
            num_entry = canonical_index.get(num_cid, {})
            den_entry = canonical_index.get(den_cid, {})
            num_v = float(num_entry.get("oracle_value", 0))
            den_v = float(den_entry.get("oracle_value", 0))
            oracle = round((num_v / den_v * 100) if den_v else 0.0, 6)
            formula_line = (
                f"RATIO: {num_entry.get('formula_line', num_cid)} "
                f"/ {den_entry.get('formula_line', den_cid)} * 100"
            )
            log.debug(f"[assembler] RATIO (pre-compiled aterms): {oracle}")
            return {
                "oracle_value":  oracle,
                "formula_line":  formula_line,
                "operator_used": "RATIO",
            }



    # ── State/snapshot tables — deduplication required ─────────────────────────
    if needs_dedup:
        if not dedup_sort:
            dedup_sort = time_col(atom) or ""

        if is_time_scoped and date_col:
            # Time-scoped dedup: filter to window → dedup within window → aggregate
            # Use dedicated operators that implement the correct order
            _fr = repr(filters) if filters else 'None'

            if time_scope in _MONTH_SCOPES:
                which = "last_month" if "last" in time_scope else "this_month"
                result = MEASURE_SNAPSHOT_DEDUPED_MONTH(
                    atom_cid, date_col, dedup_sort, grain_keys, agg, column, which, filters
                )
                _ef = (f"MEASURE_SNAPSHOT_DEDUPED_MONTH('{atom_cid}', '{date_col}', "
                       f"'{dedup_sort}', {grain_keys!r}, '{agg}', '{column}', '{which}', {_fr})")
                log.debug(f"[assembler] MEASURE_SNAPSHOT_DEDUPED_MONTH which={which}")
                return _result(result, "MEASURE_SNAPSHOT_DEDUPED_MONTH", _ef)

            elif time_scope in _QUARTER_SCOPES:
                result = MEASURE_SNAPSHOT_DEDUPED_QUARTER(
                    atom_cid, date_col, dedup_sort, grain_keys, agg, column, time_scope, filters
                )
                _ef = (f"MEASURE_SNAPSHOT_DEDUPED_QUARTER('{atom_cid}', '{date_col}', "
                       f"'{dedup_sort}', {grain_keys!r}, '{agg}', '{column}', '{time_scope}', {_fr})")
                log.debug(f"[assembler] MEASURE_SNAPSHOT_DEDUPED_QUARTER scope={time_scope}")
                return _result(result, "MEASURE_SNAPSHOT_DEDUPED_QUARTER", _ef)

            elif time_scope in _YTD_SCOPES:
                result = MEASURE_SNAPSHOT_DEDUPED_YTD(
                    atom_cid, date_col, dedup_sort, grain_keys, agg, column, filters
                )
                _ef = (f"MEASURE_SNAPSHOT_DEDUPED_YTD('{atom_cid}', '{date_col}', "
                       f"'{dedup_sort}', {grain_keys!r}, '{agg}', '{column}', {_fr})")
                log.debug(f"[assembler] MEASURE_SNAPSHOT_DEDUPED_YTD")
                return _result(result, "MEASURE_SNAPSHOT_DEDUPED_YTD", _ef)

            else:
                # Unknown time scope — fall back to standard dedup (all time)
                log.warning(f"[assembler] Unknown time_scope '{time_scope}' for dedup atom — using all_time dedup")

        # All-time dedup (no time scope or unrecognised scope)
        result = MEASURE_SNAPSHOT_DEDUPED(
            atom_cid, dedup_sort, grain_keys, agg, column, filters
        )
        log.debug(f"[assembler] MEASURE_SNAPSHOT_DEDUPED")
        _fr = repr(filters) if filters else 'None'
        _ef = f"MEASURE_SNAPSHOT_DEDUPED('{atom_cid}', '{dedup_sort}', {grain_keys!r}, '{agg}', '{column}', {_fr})"
        return _result(result, "MEASURE_SNAPSHOT_DEDUPED", _ef)

    # ── Grouped goals ──────────────────────────────────────────────────────────
    if is_grouped:
        axis = ""
        if scope.startswith("by_"):
            axis = scope[3:]
        elif series.startswith("by_"):
            axis = series[3:]

        time_axes = {"month", "quarter", "week", "year", "day"}
        _fr = repr(filters) if filters else 'None'

        if axis in time_axes:
            if not date_col:
                raise ValueError(f"Grouped by time axis '{axis}' but no date column found")
            result = MEASURE_GROUPED(
                atom_cid, agg, column,
                filters=filters,
                time_bucket={"date_col": date_col, "granularity": axis},
            )
            log.debug(f"[assembler] MEASURE_GROUPED (time-bucket) by={axis}")
            _ef = f"MEASURE_GROUPED('{atom_cid}', '{agg}', '{column}', group_by_col='{axis}', filters={_fr})"
            return _result(result, "MEASURE_GROUPED", _ef)

        # ── Grouped + YTD: apply YTD date scope before grouping ──────────────
        if is_time_scoped and date_col and time_scope in _YTD_SCOPES:
            result = MEASURE_GROUPED_YTD(
                atom_cid, date_col, agg, column,
                group_by_col=axis,
                filters=filters,
            )
            log.info(f"[assembler] MEASURE_GROUPED_YTD by={axis} scope={time_scope}")
            _ef = (
                f"MEASURE_GROUPED_YTD('{atom_cid}', '{date_col}', '{agg}', '{column}', "
                f"group_by_col='{axis}', filters={_fr})"
            )
            return _result(result, "MEASURE_GROUPED_YTD", _ef)

        # ── Plain grouped ────────────────────────────────────────────────────
        result = MEASURE_GROUPED(
            atom_cid, agg, column,
            group_by_col=axis,
            filters=filters,
        )
        log.debug(f"[assembler] MEASURE_GROUPED by={axis}")
        _ef = f"MEASURE_GROUPED('{atom_cid}', '{agg}', '{column}', group_by_col='{axis}', filters={_fr})"
        return _result(result, "MEASURE_GROUPED", _ef)

    # ── Time-scoped goals ──────────────────────────────────────────────────────
    if is_time_scoped:
        if not date_col:
            raise ValueError(
                f"Time-scoped goal ({time_scope}) but no date column selected in Step 5f"
            )

        if time_scope in _MONTH_SCOPES:
            which = "last_month" if "last" in time_scope else "this_month"
            result = MEASURE_MONTH_FIXED(atom_cid, date_col, agg, column, which, filters)
            log.debug(f"[assembler] MEASURE_MONTH_FIXED which={which}")
            _fr = repr(filters) if filters else 'None'
            _ef = f"MEASURE_MONTH_FIXED('{atom_cid}', '{date_col}', '{agg}', '{column}', '{which}', {_fr})"
            return _result(result, "MEASURE_MONTH_FIXED", _ef)

        if time_scope in _QUARTER_SCOPES:
            result = MEASURE_QUARTER_FIXED(atom_cid, date_col, agg, column, time_scope, filters)
            log.debug(f"[assembler] MEASURE_QUARTER_FIXED scope={time_scope}")
            _fr = repr(filters) if filters else 'None'
            _ef = f"MEASURE_QUARTER_FIXED('{atom_cid}', '{date_col}', '{agg}', '{column}', '{time_scope}', {_fr})"
            return _result(result, "MEASURE_QUARTER_FIXED", _ef)

        if time_scope in _YTD_SCOPES:
            result = MEASURE_YTD(atom_cid, date_col, agg, column, filters)
            log.debug(f"[assembler] MEASURE_YTD")
            _fr = repr(filters) if filters else 'None'
            _ef = f"MEASURE_YTD('{atom_cid}', '{date_col}', '{agg}', '{column}', {_fr})"
            return _result(result, "MEASURE_YTD", _ef)

        # Unknown time scope — fall through to MEASURE with a warning
        log.warning(
            f"[assembler] Unrecognised time_scope='{time_scope}' — "
            f"falling back to MEASURE (all_time)"
        )

    # ── New statistical / analytical operators ─────────────────────────────────
    _measure_slot = slots.get("measure", "").lower().replace(" ", "_")
    _fr = repr(filters) if filters else 'None'

    # Statistical distribution
    if any(k in _measure_slot for k in ("std", "stddev", "standard_deviation", "deviation")):
        result = MEASURE_STDDEV(atom_cid, column, filters)
        return _result(result, "MEASURE_STDDEV",
                       f"MEASURE_STDDEV('{atom_cid}', '{column}', {_fr})")

    if any(k in _measure_slot for k in ("cv", "coefficient_of_variation", "relative_std")):
        result = MEASURE_CV(atom_cid, column, filters)
        return _result(result, "MEASURE_CV",
                       f"MEASURE_CV('{atom_cid}', '{column}', {_fr})")

    if any(k in _measure_slot for k in ("iqr", "interquartile", "quartile_range")):
        result = MEASURE_IQR(atom_cid, column, filters)
        return _result(result, "MEASURE_IQR",
                       f"MEASURE_IQR('{atom_cid}', '{column}', {_fr})")

    if any(k in _measure_slot for k in ("skew", "skewness", "asymmetry")):
        result = MEASURE_SKEWNESS(atom_cid, column, filters)
        return _result(result, "MEASURE_SKEWNESS",
                       f"MEASURE_SKEWNESS('{atom_cid}', '{column}', {_fr})")

    if any(k in _measure_slot for k in ("percentile", "p50", "p75", "p90", "p95", "p99", "quantile", "median")):
        n = float(wizard_state.get("percentile_n", slots.get("percentile_n", 50)))
        result = MEASURE_PERCENTILE_N(atom_cid, column, n, filters)
        return _result(result, "MEASURE_PERCENTILE_N",
                       f"MEASURE_PERCENTILE_N('{atom_cid}', '{column}', {n}, {_fr})")

    # Temporal window
    if any(k in _measure_slot for k in ("moving_average", "rolling_avg", "rolling_average", "ma_")):
        periods = int(wizard_state.get("periods", slots.get("periods", 3)))
        result = MEASURE_MOVING_AVERAGE(atom_cid, date_col, column, periods, agg, filters)
        return _result(result, "MEASURE_MOVING_AVERAGE",
                       f"MEASURE_MOVING_AVERAGE('{atom_cid}', '{date_col}', '{column}', {periods}, '{agg}', {_fr})")

    if any(k in _measure_slot for k in ("running_total", "cumulative", "cumsum", "running_sum")):
        result = MEASURE_RUNNING_TOTAL(atom_cid, date_col, column, filters)
        return _result(result, "MEASURE_RUNNING_TOTAL",
                       f"MEASURE_RUNNING_TOTAL('{atom_cid}', '{date_col}', '{column}', {_fr})")

    if any(k in _measure_slot for k in ("trend", "slope", "linear_trend", "regression")):
        result = MEASURE_LINEAR_TREND(atom_cid, date_col, column, agg, filters)
        return _result(result, "MEASURE_LINEAR_TREND",
                       f"MEASURE_LINEAR_TREND('{atom_cid}', '{date_col}', '{column}', '{agg}', {_fr})")

    # Window rank
    if any(k in _measure_slot for k in ("rank", "ranking", "window_rank", "position")):
        entity_col   = wizard_state.get("entity_col", slots.get("entity_col", ""))
        entity_value = wizard_state.get("entity_value", slots.get("entity_value", ""))
        result = MEASURE_WINDOW_RANK(atom_cid, column, entity_col, entity_value, agg, filters)
        return _result(result, "MEASURE_WINDOW_RANK",
                       f"MEASURE_WINDOW_RANK('{atom_cid}', '{column}', '{entity_col}', '{entity_value}', '{agg}', {_fr})")

    # Concentration
    if any(k in _measure_slot for k in ("pareto", "concentration", "top_pct", "80_20")):
        entity_col = wizard_state.get("entity_col", slots.get("entity_col", ""))
        top_pct    = float(wizard_state.get("top_pct", slots.get("top_pct", 20.0)))
        result = MEASURE_PARETO_RATIO(atom_cid, entity_col, column, top_pct, filters)
        return _result(result, "MEASURE_PARETO_RATIO",
                       f"MEASURE_PARETO_RATIO('{atom_cid}', '{entity_col}', '{column}', {top_pct}, {_fr})")

    if any(k in _measure_slot for k in ("herfindahl", "hhi", "market_concentration")):
        entity_col = wizard_state.get("entity_col", slots.get("entity_col", ""))
        result = MEASURE_HERFINDAHL(atom_cid, entity_col, column, filters)
        return _result(result, "MEASURE_HERFINDAHL",
                       f"MEASURE_HERFINDAHL('{atom_cid}', '{entity_col}', '{column}', {_fr})")

    if any(k in _measure_slot for k in ("gini", "inequality", "gini_coefficient")):
        entity_col = wizard_state.get("entity_col", slots.get("entity_col", ""))
        result = MEASURE_GINI(atom_cid, entity_col, column, filters)
        return _result(result, "MEASURE_GINI",
                       f"MEASURE_GINI('{atom_cid}', '{entity_col}', '{column}', {_fr})")

    # Entity set operators
    _w1s = wizard_state.get("window1_start", slots.get("window1_start", ""))
    _w1e = wizard_state.get("window1_end",   slots.get("window1_end",   ""))
    _w2s = wizard_state.get("window2_start", slots.get("window2_start", ""))
    _w2e = wizard_state.get("window2_end",   slots.get("window2_end",   ""))
    _ecol = wizard_state.get("entity_col",   slots.get("entity_col",    ""))

    if any(k in _measure_slot for k in ("new_customers", "new_entities", "set_new", "acquired")):
        result = MEASURE_SET_NEW(atom_cid, _ecol, date_col, _w1s, _w1e, _w2s, _w2e, filters)
        return _result(result, "MEASURE_SET_NEW",
                       f"MEASURE_SET_NEW('{atom_cid}', '{_ecol}', '{date_col}', '{_w1s}', '{_w1e}', '{_w2s}', '{_w2e}', {_fr})")

    if any(k in _measure_slot for k in ("retained", "returning", "set_retained")):
        result = MEASURE_SET_RETAINED(atom_cid, _ecol, date_col, _w1s, _w1e, _w2s, _w2e, filters)
        return _result(result, "MEASURE_SET_RETAINED",
                       f"MEASURE_SET_RETAINED('{atom_cid}', '{_ecol}', '{date_col}', '{_w1s}', '{_w1e}', '{_w2s}', '{_w2e}', {_fr})")

    if any(k in _measure_slot for k in ("churned", "lost", "set_churned", "attrition")):
        result = MEASURE_SET_CHURNED(atom_cid, _ecol, date_col, _w1s, _w1e, _w2s, _w2e, filters)
        return _result(result, "MEASURE_SET_CHURNED",
                       f"MEASURE_SET_CHURNED('{atom_cid}', '{_ecol}', '{date_col}', '{_w1s}', '{_w1e}', '{_w2s}', '{_w2e}', {_fr})")

    if any(k in _measure_slot for k in ("cohort", "retention_rate", "cohort_retention")):
        cohort_start   = wizard_state.get("cohort_start",   slots.get("cohort_start",   ""))
        cohort_end     = wizard_state.get("cohort_end",     slots.get("cohort_end",     ""))
        retention_days = int(wizard_state.get("retention_days", slots.get("retention_days", 30)))
        result = MEASURE_COHORT_RETENTION(atom_cid, _ecol, date_col, cohort_start, cohort_end, retention_days, filters)
        return _result(result, "MEASURE_COHORT_RETENTION",
                       f"MEASURE_COHORT_RETENTION('{atom_cid}', '{_ecol}', '{date_col}', '{cohort_start}', '{cohort_end}', {retention_days}, {_fr})")

    # Correlation
    if any(k in _measure_slot for k in ("correlation", "corr", "pearson", "spearman")):
        col_a  = wizard_state.get("col_a",  slots.get("col_a",  column))
        col_b  = wizard_state.get("col_b",  slots.get("col_b",  ""))
        method = wizard_state.get("corr_method", slots.get("corr_method", "pearson"))
        result = MEASURE_CORRELATION(atom_cid, col_a, col_b, method, filters)
        return _result(result, "MEASURE_CORRELATION",
                       f"MEASURE_CORRELATION('{atom_cid}', '{col_a}', '{col_b}', '{method}', {_fr})")

    # ── Default: simple MEASURE ────────────────────────────────────────────────
    result = MEASURE(atom_cid, agg, column, filters)
    log.debug(f"[assembler] MEASURE")
    filter_repr = repr(filters) if filters else 'None'
    exec_f = f"MEASURE('{atom_cid}', '{agg}', '{column}', {filter_repr})"
    return _result(result, "MEASURE", exec_f)


def _result(
    operator_result: Dict,
    operator_name:   str,
    exec_formula:    str = "",
) -> Dict[str, Any]:
    """
    Validate and normalise operator output.

    formula_line is set to exec_formula — the Python-executable call string.
    This ensures the aterm shipped to the customer runtime is always executable.
    exec_formula is also stored separately for backward compatibility.
    """
    if not isinstance(operator_result, dict):
        raise ValueError(
            f"[assembler] Operator {operator_name} returned non-dict: "
            f"{type(operator_result)}"
        )

    oracle = operator_result.get("oracle_value")
    if oracle is None:
        raise ValueError(f"[assembler] Operator {operator_name} returned None oracle")

    try:
        oracle = float(oracle) if not isinstance(oracle, dict) else oracle
    except (TypeError, ValueError) as e:
        raise ValueError(f"[assembler] Oracle not numeric: {oracle} — {e}")

    # For Branch B: formula_line IS the executable formula.
    # exec_formula is kept for backward compatibility but both point to the same string.
    formula_line = exec_formula or operator_result.get("formula_line", "")
    if not formula_line:
        raise ValueError(f"[assembler] Operator {operator_name} returned empty formula_line")

    return {
        "oracle_value":  oracle,
        "formula_line":  formula_line,
        "exec_formula":  formula_line,
        "operator_used": operator_name,
    }
