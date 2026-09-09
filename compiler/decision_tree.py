"""
decision_tree.py
=================
Deterministic routing: given a goal's slots + the atom's record_type,
always return the SAME operator name for the SAME inputs. No AI, no
fuzzy matching — this is what makes Branch A compilation reproducible
and free.

This module only decides WHICH operator to call. The actual calling
(building the right arguments and invoking the operator function) is
done by the Branch A compiler orchestrator (next build step).
"""

from typing import Dict


# Operator names this tree can emit — must match function names in
# compiler/operators/*.py
OPERATOR_MEASURE                  = "MEASURE"
OPERATOR_MEASURE_SNAPSHOT_DEDUPED = "MEASURE_SNAPSHOT_DEDUPED"
OPERATOR_MEASURE_MONTH_FIXED      = "MEASURE_MONTH_FIXED"
OPERATOR_MEASURE_QUARTER_FIXED    = "MEASURE_QUARTER_FIXED"
OPERATOR_MEASURE_YTD              = "MEASURE_YTD"
OPERATOR_MEASURE_GROUPED          = "MEASURE_GROUPED"

# ── New operators (v29) ────────────────────────────────────────────────────────
OPERATOR_MEASURE_STDDEV           = "MEASURE_STDDEV"
OPERATOR_MEASURE_CV               = "MEASURE_CV"
OPERATOR_MEASURE_IQR              = "MEASURE_IQR"
OPERATOR_MEASURE_SKEWNESS         = "MEASURE_SKEWNESS"
OPERATOR_MEASURE_PERCENTILE_N     = "MEASURE_PERCENTILE_N"
OPERATOR_MEASURE_MOVING_AVERAGE   = "MEASURE_MOVING_AVERAGE"
OPERATOR_MEASURE_RUNNING_TOTAL    = "MEASURE_RUNNING_TOTAL"
OPERATOR_MEASURE_LINEAR_TREND     = "MEASURE_LINEAR_TREND"
OPERATOR_MEASURE_WINDOW_RANK      = "MEASURE_WINDOW_RANK"
OPERATOR_MEASURE_PARETO_RATIO     = "MEASURE_PARETO_RATIO"
OPERATOR_MEASURE_HERFINDAHL       = "MEASURE_HERFINDAHL"
OPERATOR_MEASURE_GINI             = "MEASURE_GINI"
OPERATOR_MEASURE_SET_NEW          = "MEASURE_SET_NEW"
OPERATOR_MEASURE_SET_RETAINED     = "MEASURE_SET_RETAINED"
OPERATOR_MEASURE_SET_CHURNED      = "MEASURE_SET_CHURNED"
OPERATOR_MEASURE_COHORT_RETENTION = "MEASURE_COHORT_RETENTION"
OPERATOR_MEASURE_CORRELATION      = "MEASURE_CORRELATION"

# Goals that cannot be handled algebraically — must fall through to
# Branch B (LLM wizard) or Branch C (composition)
ROUTE_COMPOSITION   = "ROUTE_COMPOSITION"     # averages, growth, KPIs — depend on other aterms
ROUTE_LLM_WIZARD    = "ROUTE_LLM_WIZARD"      # complex / unsupported pattern

# ── Measure slot keywords for new operator routing ─────────────────────────────
_STDDEV_KEYWORDS      = {"std", "stddev", "standard_deviation", "deviation"}
_CV_KEYWORDS          = {"cv", "coefficient_of_variation", "relative_std"}
_IQR_KEYWORDS         = {"iqr", "interquartile", "quartile_range"}
_SKEWNESS_KEYWORDS    = {"skew", "skewness", "asymmetry"}
_PERCENTILE_KEYWORDS  = {"percentile", "p50", "p75", "p90", "p95", "p99", "quantile", "median"}
_MOVING_AVG_KEYWORDS  = {"moving_average", "rolling_avg", "rolling_average", "ma_"}
_RUNNING_TOTAL_KEYWORDS = {"running_total", "cumulative", "cumsum", "running_sum"}
_TREND_KEYWORDS       = {"trend", "slope", "linear_trend", "regression"}
_RANK_KEYWORDS        = {"rank", "ranking", "window_rank", "position"}
_PARETO_KEYWORDS      = {"pareto", "concentration", "top_pct", "80_20"}
_HHI_KEYWORDS         = {"herfindahl", "hhi", "market_concentration"}
_GINI_KEYWORDS        = {"gini", "inequality", "gini_coefficient"}
_SET_NEW_KEYWORDS     = {"new_customers", "new_entities", "set_new", "acquired"}
_SET_RETAINED_KEYWORDS= {"retained", "returning", "set_retained"}
_SET_CHURNED_KEYWORDS = {"churned", "lost", "set_churned", "attrition"}
_COHORT_KEYWORDS      = {"cohort", "retention_rate", "cohort_retention"}
_CORRELATION_KEYWORDS = {"correlation", "corr", "pearson", "spearman"}


def select_operator(slots: Dict, record_type: str, complexity: str = "PRIMITIVE") -> str:
    """
    Decide which operator (or routing) a goal should use.

    Args:
        slots:       the goal's 8-slot dict (domain, entity, measure, state,
                     scope, time, unit, series)
        record_type: the atom's record_type ("record" | "state" | "dimension"
                     | "event" | "snapshot")
        complexity:  the goal's complexity tag from goal generation
                     ("PRIMITIVE" | "FILTERED" | "TIME_SCOPED" | "COMPOSED_1"
                     | "COMPOSED_2" | "CROSS_TABLE" | "SERIES" | "KPI")

    Returns:
        One of the OPERATOR_* / ROUTE_* constants above.
    """
    # ── Composed goals never execute against raw data — they read
    #    other already-compiled aterms and do arithmetic. Branch C
    #    handles these, not Branch A.
    if complexity in ("COMPOSED_1", "COMPOSED_2", "KPI"):
        return ROUTE_COMPOSITION

    time  = slots.get("time", "all_time")
    series = slots.get("series", "scalar")
    scope = slots.get("scope", "total")

    # ── Grouped / series / cross-table goals
    if series != "scalar" or (scope != "total" and scope.startswith("by_")):
        return OPERATOR_MEASURE_GROUPED

    # ── State tables always dedup first, regardless of time window.
    #    (A state table with a time filter would need dedup THEN time
    #    filter THEN state filter — not yet supported, routes to LLM
    #    wizard for now since correctness here needs careful ordering.)
    if record_type == "state":
        if time != "all_time":
            return ROUTE_LLM_WIZARD  # dedup + time-window combo — not yet algebraic
        return OPERATOR_MEASURE_SNAPSHOT_DEDUPED

    # ── Time-scoped goals on record/event tables
    if time == "this_month" or time == "last_month":
        return OPERATOR_MEASURE_MONTH_FIXED
    if time in ("ytd", "year_to_date"):
        return OPERATOR_MEASURE_YTD
    if time in ("this_quarter", "last_quarter", "this_year", "last_year"):
        return OPERATOR_MEASURE_QUARTER_FIXED
    if time not in ("all_time", "alltime", ""):
        # this_quarter, last_quarter, this_year, last_year, last_12_months,
        # trailing windows etc — not yet covered by a dedicated operator
        return ROUTE_LLM_WIZARD

    # ── New statistical / analytical operators — route by measure slot keywords
    measure = slots.get("measure", "").lower().replace(" ", "_")

    if any(k in measure for k in _STDDEV_KEYWORDS):
        return OPERATOR_MEASURE_STDDEV
    if any(k in measure for k in _CV_KEYWORDS):
        return OPERATOR_MEASURE_CV
    if any(k in measure for k in _IQR_KEYWORDS):
        return OPERATOR_MEASURE_IQR
    if any(k in measure for k in _SKEWNESS_KEYWORDS):
        return OPERATOR_MEASURE_SKEWNESS
    if any(k in measure for k in _PERCENTILE_KEYWORDS):
        return OPERATOR_MEASURE_PERCENTILE_N
    if any(k in measure for k in _MOVING_AVG_KEYWORDS):
        return OPERATOR_MEASURE_MOVING_AVERAGE
    if any(k in measure for k in _RUNNING_TOTAL_KEYWORDS):
        return OPERATOR_MEASURE_RUNNING_TOTAL
    if any(k in measure for k in _TREND_KEYWORDS):
        return OPERATOR_MEASURE_LINEAR_TREND
    if any(k in measure for k in _RANK_KEYWORDS):
        return OPERATOR_MEASURE_WINDOW_RANK
    if any(k in measure for k in _PARETO_KEYWORDS):
        return OPERATOR_MEASURE_PARETO_RATIO
    if any(k in measure for k in _HHI_KEYWORDS):
        return OPERATOR_MEASURE_HERFINDAHL
    if any(k in measure for k in _GINI_KEYWORDS):
        return OPERATOR_MEASURE_GINI
    if any(k in measure for k in _SET_NEW_KEYWORDS):
        return OPERATOR_MEASURE_SET_NEW
    if any(k in measure for k in _SET_RETAINED_KEYWORDS):
        return OPERATOR_MEASURE_SET_RETAINED
    if any(k in measure for k in _SET_CHURNED_KEYWORDS):
        return OPERATOR_MEASURE_SET_CHURNED
    if any(k in measure for k in _COHORT_KEYWORDS):
        return OPERATOR_MEASURE_COHORT_RETENTION
    if any(k in measure for k in _CORRELATION_KEYWORDS):
        return OPERATOR_MEASURE_CORRELATION

    # ── Simplest case: no time window, no grouping, no dedup needed
    return OPERATOR_MEASURE
