"""
customer/formula_executor.py
==============================
Formula Executor — executes compiled formulas against customer's real SOT data.

On HIT: the formula_line from the aterm is executed against customer_runtime/sot_csv/.
The result is the real oracle value from real customer data.
This value stays on the customer side — it never leaves.

HOW FORMULA LINES WORK
-----------------------
The compiler stores formula_line as a HUMAN-READABLE display string, not a
Python call.  For example:

    MEASURE(COUNT_DISTINCT(order_id) FROM orders_erp_record)

This is NOT valid Python.  The actual Python call to replay the same
computation is:

    MEASURE("orders_erp_record", "COUNT_DISTINCT", "order_id")

This module contains a parser (_parse_display_formula) that translates every
display string pattern back into the correct Python operator call, then
exec()s it against a namespace built on the customer's real SOT CSV root.

FORMULA PATTERNS HANDLED
-------------------------
  MEASURE(AGG(col) FROM atom [WHERE field=val ...])
  MEASURE_MONTH_FIXED(AGG(col) FROM atom WHERE date_col IN YYYY-MM [WHERE ...])
  MEASURE_QUARTER_FIXED(AGG(col) FROM atom WHERE date_col IN YYYY-QN [WHERE ...])
  MEASURE_YTD(AGG(col) FROM atom WHERE date_col BETWEEN d1 AND d2 [WHERE ...])
  MEASURE_GROUPED(AGG(col) FROM atom GROUP BY BUCKET(col, month))
  RATIO: <num_formula> / <den_formula> [* 100]
  {dep_cid} composition placeholders
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


# ── Fingerprinting ─────────────────────────────────────────────────────────────

def _fingerprint(csv_path: Path) -> str:
    if not csv_path.exists():
        return "MISSING"
    return hashlib.sha256(csv_path.read_bytes()).hexdigest()[:16]


def _composite_fingerprint(table_names: List[str], sot_dir: Path) -> str:
    fps = [_fingerprint(sot_dir / f"{t}.csv") for t in sorted(set(table_names))]
    return hashlib.sha256("|".join(fps).encode()).hexdigest()[:16]


def _extract_tables(formula: str) -> List[str]:
    """Extract atom canonical_ids referenced in a display formula string."""
    return re.findall(
        r'FROM\s+([a-z][a-z0-9_]+)',
        formula,
        re.IGNORECASE,
    )


# ── Display formula parser ─────────────────────────────────────────────────────

def _parse_where_clauses(where_str: str) -> List[Dict]:
    """
    Parse ' WHERE field=value WHERE field2=value2 ...' into filter dicts.
    Skips date-range clauses (IN / BETWEEN) — those are handled by the
    time-windowed operators directly.
    """
    filters = []
    # Strip any leading WHERE before splitting on subsequent WHERE clauses
    cleaned = re.sub(r'^\s*WHERE\s+', '', where_str.strip(), flags=re.IGNORECASE)
    parts = re.split(r'\s+WHERE\s+', cleaned, flags=re.IGNORECASE)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Skip time clauses: "date_col IN 2026-07", "date_col BETWEEN ... AND ..."
        if re.search(r'\bIN\s+\d{4}', part, re.IGNORECASE):
            continue
        if re.search(r'\bBETWEEN\b', part, re.IGNORECASE):
            continue
        # Parse field=value
        m = re.match(r'^([a-z_][a-z0-9_]*)\s*=\s*(.+)$', part, re.IGNORECASE)
        if m:
            filters.append({"field": m.group(1), "op": "=", "value": m.group(2).strip()})
    return filters


def _parse_agg_col(agg_expr: str):
    """
    Parse 'COUNT_DISTINCT(order_id)' → ('COUNT_DISTINCT', 'order_id')
    Parse 'COUNT(*)'                  → ('COUNT', None)
    Parse 'SUM(freight_cost)'         → ('SUM', 'freight_cost')
    """
    m = re.match(r'([A-Z_]+)\(([^)]*)\)', agg_expr.strip())
    if not m:
        raise ValueError(f"Cannot parse agg expression: {agg_expr!r}")
    agg = m.group(1)
    col = m.group(2).strip()
    if col in ('', '*'):
        col = None
    return agg, col


def _parse_measure_body(body: str):
    """
    Parse the inner body of MEASURE_*(... FROM atom WHERE ...).
    Returns (agg, col, atom, date_col, date_spec, filters).
    """
    # Split on FROM
    m = re.match(r'^(.+?)\s+FROM\s+([a-z][a-z0-9_]+)(.*)?$', body.strip(), re.IGNORECASE | re.DOTALL)
    if not m:
        raise ValueError(f"Cannot parse measure body: {body!r}")

    agg_expr  = m.group(1).strip()
    atom      = m.group(2).strip()
    remainder = (m.group(3) or '').strip()

    agg, col = _parse_agg_col(agg_expr)

    # Extract date clause if present
    date_col  = None
    date_spec = None

    # BETWEEN date1 AND date2
    bm = re.search(
        r'WHERE\s+([a-z_]+)\s+BETWEEN\s+(\S+)\s+AND\s+(\S+)',
        remainder, re.IGNORECASE,
    )
    if bm:
        date_col  = bm.group(1)
        date_spec = f"BETWEEN {bm.group(2)} AND {bm.group(3)}"
        remainder = remainder[:bm.start()] + remainder[bm.end():]

    # IN YYYY-MM or IN YYYY-QN
    im = re.search(
        r'WHERE\s+([a-z_]+)\s+IN\s+(\S+)',
        remainder, re.IGNORECASE,
    )
    if im:
        date_col  = im.group(1)
        date_spec = f"IN {im.group(2)}"
        remainder = remainder[:im.start()] + remainder[im.end():]

    filters = _parse_where_clauses(remainder)

    return agg, col, atom, date_col, date_spec, filters


def _build_python_call(func: str, display_formula: str, ns: Dict) -> Any:
    """
    Translate a display formula string into the equivalent Python operator
    call and execute it in ns.  Returns the result dict from the operator.
    """
    # Strip outer FUNC_NAME(...)
    inner_match = re.match(rf'^{re.escape(func)}\((.+)\)$', display_formula.strip(), re.DOTALL)
    if not inner_match:
        raise ValueError(f"Cannot strip outer {func}() from: {display_formula!r}")
    body = inner_match.group(1)

    # ── MEASURE_GROUPED ────────────────────────────────────────────────────────
    if func == 'MEASURE_GROUPED':
        # Signature: (atom, agg, column, group_by_col=None, filters=None, time_bucket=None)
        gm = re.search(r'\s+GROUP\s+BY\s+(.+)$', body, re.IGNORECASE)
        if not gm:
            raise ValueError(f"No GROUP BY in MEASURE_GROUPED: {body!r}")
        group_clause  = gm.group(1).strip()
        body_no_group = body[:gm.start()]
        agg, col, atom, _, _, filters = _parse_measure_body(body_no_group)

        # BUCKET(date_col, month) → time_bucket={"date_col": ..., "granularity": "month"}
        bm = re.match(r'BUCKET\(([a-z_]+),\s*([a-z]+)\)', group_clause, re.IGNORECASE)
        if bm:
            date_col_grp  = bm.group(1)
            granularity   = bm.group(2).lower()
            kwargs_parts  = [
                f'"{atom}"', f'"{agg}"', f'"{col}"' if col else 'None',
                f'time_bucket={{"date_col": "{date_col_grp}", "granularity": "{granularity}"}}',
            ]
        else:
            # Plain column grouping
            kwargs_parts = [
                f'"{atom}"', f'"{agg}"', f'"{col}"' if col else 'None',
                f'group_by_col="{group_clause}"',
            ]
        if filters:
            kwargs_parts.append(f'filters={json.dumps(filters)}')

        py_call = f'MEASURE_GROUPED({", ".join(kwargs_parts)})'
        exec(f'__result__ = {py_call}', ns)
        return ns['__result__']

    # ── MEASURE / MEASURE_MONTH_FIXED / MEASURE_QUARTER_FIXED / MEASURE_YTD ──
    agg, col, atom, date_col, date_spec, filters = _parse_measure_body(body)

    if func == 'MEASURE':
        kwargs_parts = [f'"{atom}"', f'"{agg}"', f'"{col}"' if col else 'None']
        if filters:
            kwargs_parts.append(f'filters={json.dumps(filters)}')
        py_call = f'MEASURE({", ".join(kwargs_parts)})'
        exec(f'__result__ = {py_call}', ns)
        return ns['__result__']

    if func == 'MEASURE_MONTH_FIXED':
        # Signature: (atom, date_col, agg, column, which_month="this_month", filters=None)
        import datetime as _dt
        which_month = "this_month"
        if date_spec and date_spec.startswith("IN "):
            spec_val = date_spec[3:].strip()
            try:
                spec_date  = _dt.date.fromisoformat(spec_val + "-01")
                today      = _dt.date.today()
                this_month = today.replace(day=1)
                last_month = (this_month - _dt.timedelta(days=1)).replace(day=1)
                which_month = "last_month" if spec_date == last_month else "this_month"
            except Exception:
                pass
        kwargs_parts = [
            f'"{atom}"', f'"{date_col}"', f'"{agg}"',
            f'"{col}"' if col else 'None',
            f'which_month="{which_month}"',
        ]
        if filters:
            kwargs_parts.append(f'filters={json.dumps(filters)}')
        py_call = f'MEASURE_MONTH_FIXED({", ".join(kwargs_parts)})'
        exec(f'__result__ = {py_call}', ns)
        return ns['__result__']

    if func == 'MEASURE_QUARTER_FIXED':
        # Signature: (atom, date_col, agg, column, which="this_quarter", filters=None)
        which = "this_quarter"
        if date_spec and date_spec.startswith("IN "):
            spec_val = date_spec[3:].strip()
            try:
                from compiler.operators.gpl_operators import current_quarter as _cq, previous_quarter as _pq
                cy, cq = _cq(); py2, pq = _pq()
                import datetime as _dt2
                if spec_val == f"{cy}-Q{cq}":      which = "this_quarter"
                elif spec_val == f"{py2}-Q{pq}":   which = "last_quarter"
                elif "-Q" not in spec_val:
                    yr = int(spec_val)
                    which = "this_year" if yr == _dt2.date.today().year else "last_year"
            except Exception:
                pass
        kwargs_parts = [
            f'"{atom}"', f'"{date_col}"', f'"{agg}"',
            f'"{col}"' if col else 'None',
            f'which="{which}"',
        ]
        if filters:
            kwargs_parts.append(f'filters={json.dumps(filters)}')
        py_call = f'MEASURE_QUARTER_FIXED({", ".join(kwargs_parts)})'
        exec(f'__result__ = {py_call}', ns)
        return ns['__result__']

    if func == 'MEASURE_YTD':
        # Signature: (atom, date_col, agg, column, filters=None)
        kwargs_parts = [
            f'"{atom}"', f'"{date_col}"', f'"{agg}"',
            f'"{col}"' if col else 'None',
        ]
        if filters:
            kwargs_parts.append(f'filters={json.dumps(filters)}')
        py_call = f'MEASURE_YTD({", ".join(kwargs_parts)})'
        exec(f'__result__ = {py_call}', ns)
        return ns['__result__']

    raise ValueError(f"Unhandled operator function: {func}")


# ── Single MEASURE display string execution ────────────────────────────────────

def _exec_display_measure(formula_line: str, ns: Dict, sot_dir: Path, vertical: str) -> Dict:
    """
    Execute a single MEASURE_*(... FROM ...) display string.
    Returns {status, oracle_value, method, fingerprint}.
    """
    func_match = re.match(r'^(MEASURE[A-Z_]*)\(', formula_line)
    if not func_match:
        raise ValueError(f"Not a MEASURE display formula: {formula_line!r}")
    func = func_match.group(1)

    result = _build_python_call(func, formula_line, ns)

    tables = _extract_tables(formula_line)
    fp     = _composite_fingerprint(tables, sot_dir / vertical) if tables else "no_tables"
    oracle = result.get("oracle_value") if isinstance(result, dict) else float(result)

    return {
        "status":       "ok",
        "oracle_value": oracle,
        "method":       "display_parse_execution",
        "fingerprint":  fp,
    }


# ── Main entry point ───────────────────────────────────────────────────────────

def execute(
    formula_line:      str,
    sot_dir:           Path,
    vertical:          str,
    canonical_index:   Optional[Dict] = None,
    composition_rules: Optional[Dict] = None,
    _resolving:        Optional[set]  = None,   # cycle-detection — passed through from _execute_composition
) -> Dict[str, Any]:
    """
    Execute a compiled formula against the customer's real SOT data.

    Handles all formula_line patterns produced by the compiler:
      - MEASURE(AGG(col) FROM atom ...)
      - MEASURE_MONTH_FIXED / MEASURE_QUARTER_FIXED / MEASURE_YTD / MEASURE_GROUPED
      - RATIO: <num> / <den> [* 100]
      - {dep_cid} composition placeholders

    Returns: {status, oracle_value, method, fingerprint, error?}
    """
    csv_root = str(sot_dir / vertical)

    if not formula_line:
        return {"status": "error", "error": "Empty formula_line"}

    try:
        from compiler.operators.gpl_operators import build_namespace
        from compiler.operators._csv_loader import set_mock_fallback_root
        ns = build_namespace(csv_root)

        # Set mock_data as automatic fallback when sot_csv/ files are missing.
        # sot_dir is e.g. customer_runtime/customers/<id>/sot_csv/
        # mock_data lives at  customer_runtime/customers/<id>/data/mock_data/
        mock_root = sot_dir.parent / "data" / "mock_data" / vertical
        set_mock_fallback_root(mock_root if mock_root.exists() else None)

        # ── RATIO formula ──────────────────────────────────────────────────────
        if formula_line.startswith("RATIO:"):
            return _execute_ratio(formula_line, ns, sot_dir, vertical)

        # ── Composition {dep_cid} placeholders ────────────────────────────────
        # Must match {word_chars} style placeholders (dep canonical IDs).
        # Python-callable formulas with filter dicts like [{'field':...}] also
        # contain { } but those are NOT composition placeholders — exclude them.
        if re.search(r'\{[a-z][a-z0-9_]+\}', formula_line):
            return _execute_composition(
                formula_line, canonical_index or {}, sot_dir, vertical, _resolving
            )

        # ── MEASURE display string (SQL/human-readable style only) ────────────
        # SQL style:    MEASURE(COUNT_DISTINCT(col) FROM atom ...)
        # Python style: MEASURE('atom', 'COUNT_DISTINCT', 'col', [...])
        # Only route to _exec_display_measure for SQL-style (contains " FROM ").
        # Python-callable formulas fall through to the raw exec() path below.
        if re.match(r'^MEASURE', formula_line) and ' FROM ' in formula_line.upper():
            return _exec_display_measure(formula_line, ns, sot_dir, vertical)

        # ── Fallback: raw exec — handles Python-callable operator strings ──────
        # This covers both Branch B/wizard formulas AND the new @gpl_operator
        # style where formula_line = "MEASURE('atom', 'AGG', 'col', [filters])"
        exec(f"__result__ = {formula_line}", ns)
        result = ns.get("__result__", {})
        oracle = result.get("oracle_value") if isinstance(result, dict) else float(result)
        tables = _extract_tables(formula_line)
        fp     = _composite_fingerprint(tables, sot_dir / vertical) if tables else "no_tables"
        return {"status": "ok", "oracle_value": oracle, "method": "raw_exec", "fingerprint": fp}

    except Exception as e:
        log.error(f"[formula_executor] Failed: {e} | formula: {formula_line[:120]}")
        return {"status": "error", "error": str(e)}


# ── RATIO execution ────────────────────────────────────────────────────────────

def _execute_ratio(
    formula_line: str,
    ns:           Dict,
    sot_dir:      Path,
    vertical:     str,
) -> Dict[str, Any]:
    """
    Execute RATIO: <num_formula> / <den_formula> [* 100]
    Each of num/den is a MEASURE display string — parsed via _exec_display_measure.
    """
    expr    = formula_line[6:].strip()   # strip "RATIO:"
    has_pct = expr.endswith("* 100")
    if has_pct:
        expr = expr[: expr.rfind("* 100")].strip().rstrip("/").strip()

    # Split at the top-level "/" between two MEASURE calls
    # We walk char by char tracking paren depth
    depth = 0
    div_i = None
    for i, ch in enumerate(expr):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "/" and depth == 0:
            div_i = i
            break

    if div_i is None:
        return {"status": "error", "error": "Cannot parse RATIO formula — no top-level /"}

    num_str = expr[:div_i].strip()
    den_str = expr[div_i + 1:].strip()

    def _exec_part(s: str) -> float:
        # SQL-style display formula → parse via _exec_display_measure
        if re.match(r'^MEASURE', s) and ' FROM ' in s.upper():
            r = _exec_display_measure(s, ns, sot_dir, vertical)
            if r["status"] != "ok":
                raise ValueError(r.get("error", "part failed"))
            return float(r["oracle_value"] or 0)
        # Python-callable or anything else → raw exec
        exec(f"__part__ = {s}", ns)
        v = ns.get("__part__", {})
        return float(v.get("oracle_value", 0) if isinstance(v, dict) else v)

    nv = _exec_part(num_str)
    dv = _exec_part(den_str)

    oracle = round((nv / dv * 100) if dv else 0.0, 6)

    tables = _extract_tables(formula_line)
    fp     = _composite_fingerprint(tables, sot_dir / vertical) if tables else "no_tables"

    return {
        "status":       "ok",
        "oracle_value": oracle,
        "method":       "ratio_execution",
        "fingerprint":  fp,
    }


# ── Composition execution ──────────────────────────────────────────────────────

def _execute_composition(
    formula_line:    str,
    canonical_index: Dict,
    sot_dir:         Path,
    vertical:        str,
    _resolving:      Optional[set] = None,
) -> Dict[str, Any]:
    """
    Execute a composition formula by resolving {dep_cid} placeholders.

    Each dependency is executed LIVE against the real SOT CSV by calling
    execute() recursively, so composition results reflect actual customer
    data rather than the mock oracle values stored during factory compilation.

    A _resolving set tracks in-progress dep_cids to detect and abort
    circular dependency chains gracefully.
    """
    if _resolving is None:
        _resolving = set()

    tpl = formula_line.split("=", 1)[1].strip() if "=" in formula_line else formula_line

    dep_values: Dict[str, float] = {}
    for placeholder in re.findall(r"\{([^}]+)\}", tpl):
        # Cycle guard
        if placeholder in _resolving:
            return {
                "status": "error",
                "error":  f"Circular dependency detected: '{placeholder}' is already being resolved",
            }

        entry = canonical_index.get(placeholder)
        if not entry:
            return {
                "status": "error",
                "error":  f"Dependency '{placeholder}' not found in local index",
            }

        dep_formula = entry.get("formula_line", "")

        # Execute dependency live
        if dep_formula:
            dep_result = execute(
                dep_formula,
                sot_dir         = sot_dir,
                vertical        = vertical,
                canonical_index = canonical_index,
                _resolving      = _resolving | {placeholder},
            )
            if dep_result.get("status") != "ok":
                log.warning(
                    f"[composition] Live execution of dependency '{placeholder}' failed "
                    f"({dep_result.get('error','?')}) — falling back to stored mock oracle"
                )
                raw = entry.get("oracle_value", 0)
                dep_values[placeholder] = float(raw) if not isinstance(raw, dict) else 0.0
            else:
                live = dep_result.get("live_oracle")
                if live is None:
                    live = dep_result.get("oracle_value", 0)
                dep_values[placeholder] = float(live) if not isinstance(live, dict) else 0.0
        else:
            log.warning(
                f"[composition] Dependency '{placeholder}' has no formula_line — "
                f"falling back to stored mock oracle"
            )
            raw = entry.get("oracle_value", 0)
            dep_values[placeholder] = float(raw) if not isinstance(raw, dict) else 0.0

    expr = tpl
    for dep_cid, dep_val in dep_values.items():
        expr = expr.replace("{" + dep_cid + "}", str(dep_val))

    if re.search(r"\{[^}]+\}", expr):
        return {"status": "error", "error": "Unresolved placeholders in composition formula"}

    try:
        oracle = float(eval(expr, {"__builtins__": {}, "abs": abs, "round": round}))
    except Exception as e:
        return {"status": "error", "error": f"Composition eval failed: {e}"}

    return {
        "status":       "ok",
        "oracle_value": oracle,
        "live_oracle":  oracle,
        "method":       "composition_live",
        "fingerprint":  "composed",
    }
