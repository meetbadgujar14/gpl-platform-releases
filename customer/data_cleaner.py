"""
customer/data_cleaner.py
========================
Standalone data-cleaning module applied before excel_parser results are
ingested.  Pure Python — no LLM, no I/O except the transformation log.

Public API
----------
    clean_sheet(sheet, options)
        → (cleaned_sheet, transformation_report)

    write_transformation_log(customer_data_dir, file_id, reports)
        → Path   (transformation_log.json inside customer data dir)

Transformation pipeline (execution order)
------------------------------------------
1. Column header cleanup  — strip whitespace, collapse spaces, remove BOM
2. Empty column detection — flag fully-empty columns, keep headers
3. Null standardisation   — normalise sentinel strings → ""
4. Whitespace cleanup     — strip/collapse all string values
5. Number normalisation   — strip $, ₹, €, £, ¥, commas, trailing %
6. Date standardisation   — resolve ambiguous dates → ISO 8601

Badge colour logic (for UI)
---------------------------
- "green"  → only whitespace / null transformations applied
- "amber"  → dates or numbers were changed
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ── Types ─────────────────────────────────────────────────────────────────────

Sheet = Dict[str, Any]          # ExcelSheet-compatible dict
Row   = Dict[str, Any]
TransformationReport = Dict[str, Any]

# ── Null sentinel set ──────────────────────────────────────────────────────────

_NULL_SENTINELS: set = {
    "-", "n/a", "na", "null", "none", "#n/a", "nan",
    "#null!", "#value!", "#ref!", "#div/0!", "#name?", "#num!", "#error!",
    "—", "–",   # em-dash / en-dash sometimes used as missing
}

# ── Date hint patterns for column-name-based disambiguation ──────────────────

_DATE_COL_HINTS = re.compile(
    r"(date|day|dob|born|expir|deliv|order|ship|arrival|depart|period|month|year|quarter)",
    re.IGNORECASE,
)

# Verticals where DD/MM is conventional
_DDMM_VERTICALS = {"logistics", "supply_chain", "supply chain", "manufacturing", "fleet"}


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Column header cleanup
# ─────────────────────────────────────────────────────────────────────────────

def _clean_header(name: str, is_first: bool = False) -> str:
    """Strip leading/trailing whitespace, collapse internal spaces, remove BOM."""
    if is_first:
        name = name.lstrip("\ufeff\ufffe")   # UTF-8 and UTF-16 BOMs
    name = name.strip()
    name = re.sub(r" {2,}", " ", name)
    return name


def _clean_headers(columns: List[str]) -> Tuple[List[str], Dict[str, str], int]:
    """
    Return (cleaned_columns, rename_map, cells_affected).
    rename_map: {original_col → cleaned_col} — only entries that actually changed.
    """
    cleaned: List[str] = []
    rename_map: Dict[str, str] = {}
    affected = 0
    for i, col in enumerate(columns):
        clean = _clean_header(col, is_first=(i == 0))
        cleaned.append(clean)
        if clean != col:
            rename_map[col] = clean
            affected += 1
    return cleaned, rename_map, affected


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Empty column detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_empty_columns(rows: List[Row], columns: List[str]) -> List[str]:
    """
    Return list of column names where every data value is empty / None / NaN.
    The header itself is kept (not dropped) — only flagged in the report.
    """
    empty_cols: List[str] = []
    for col in columns:
        if all(_value_is_empty(row.get(col)) for row in rows):
            empty_cols.append(col)
    return empty_cols


def _value_is_empty(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float):
        import math
        return math.isnan(v)
    if isinstance(v, str):
        return v.strip() == ""
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Null standardisation
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_null(v: Any) -> Tuple[Any, bool]:
    """
    Return (normalised_value, was_changed).
    Sentinel strings are mapped to ""; others are returned unchanged.
    """
    if not isinstance(v, str):
        return v, False
    stripped = v.strip()
    if stripped.lower() in _NULL_SENTINELS:
        return "", stripped != ""
    return v, False


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Whitespace cleanup
# ─────────────────────────────────────────────────────────────────────────────

def _clean_whitespace(v: str) -> Tuple[str, bool]:
    """
    Strip leading/trailing whitespace, collapse internal spaces,
    replace embedded \\r, \\n, \\t with a single space.
    Returns (cleaned, was_changed).
    """
    if not isinstance(v, str):
        return v, False
    original = v
    v = re.sub(r"[\r\n\t]+", " ", v)   # embedded control chars → space
    v = v.strip()
    v = re.sub(r" {2,}", " ", v)
    return v, v != original


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Number normalisation
# ─────────────────────────────────────────────────────────────────────────────

_CURRENCY_RE   = re.compile(r"^[\$₹€£¥\s]+")
_THOUSANDS_RE  = re.compile(r"(?<=\d),(?=\d{3})")
_TRAILING_PCT  = re.compile(r"%\s*$")
_NUMERIC_CHECK = re.compile(r"^-?[\d,]+(\.\d+)?%?$")


def _looks_numeric(s: str) -> bool:
    """Return True if the string looks like a number (possibly with symbols)."""
    stripped = re.sub(r"[\$₹€£¥\s,]", "", s)
    stripped = re.sub(r"%$", "", stripped)
    if not stripped:
        return False
    try:
        float(stripped)
        return True
    except ValueError:
        return False


def _normalise_number(v: Any) -> Tuple[Any, bool]:
    """
    Strip currency symbols, thousand separators, and trailing %.
    Only applied to cells that look numeric; strings are left untouched.
    Returns (normalised, was_changed).
    """
    if not isinstance(v, str) or v.strip() == "":
        return v, False
    s = v.strip()
    if not _looks_numeric(s):
        return v, False
    original = s
    s = _CURRENCY_RE.sub("", s)        # leading currency
    s = _THOUSANDS_RE.sub("", s)       # thousand commas
    s = _TRAILING_PCT.sub("", s).strip()   # trailing %
    s = s.strip()
    return s, s != original


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Date standardisation → ISO 8601
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that capture  (a, b, c)  where the order (DMY vs MDY) is ambiguous
# Group names: seg1, sep, seg2, sep2, seg3
_DATE_SEG_RE = re.compile(
    r"^(?P<seg1>\d{1,4})(?P<sep>[/\-\.])(?P<seg2>\d{1,2})(?P=sep)(?P<seg3>\d{2,4})$"
)

# Unambiguous formats (no confusion possible)
_ISO_RE   = re.compile(r"^\d{4}-\d{2}-\d{2}$")          # already ISO
_DDMMYYYY = re.compile(r"^\d{2}/\d{2}/\d{4}$")          # could be DMY or MDY — needs resolution


def _parse_date_segments(s: str) -> Optional[Tuple[int, int, int]]:
    """Return (seg1, seg2, seg3) integers if s matches a 3-segment date pattern."""
    m = _DATE_SEG_RE.match(s.strip())
    if not m:
        return None
    try:
        return int(m.group("seg1")), int(m.group("seg2")), int(m.group("seg3"))
    except ValueError:
        return None


def _to_iso(seg1: int, seg2: int, seg3: int, day_first: bool) -> Optional[str]:
    """
    Build an ISO-8601 date string.

    day_first=True  → interpret as DD/MM/YYYY  (seg1=day,  seg2=month, seg3=year)
    day_first=False → interpret as MM/DD/YYYY  (seg1=month,seg2=day,   seg3=year)

    Handles 2-digit years: 00-29 → 2000-2029, 30-99 → 1930-1999.
    Handles YYYY-MM-DD when seg1 > 31 (already a 4-digit year in front).
    """
    # If seg1 > 31 it must be the 4-digit year (YYYY-MM-DD or YYYY/MM/DD)
    if seg1 > 31:
        year, month, day = seg1, seg2, seg3
    else:
        year = seg3
        if year < 100:
            year += 2000 if year < 30 else 1900
        if day_first:
            day, month = seg1, seg2
        else:
            month, day = seg1, seg2

    try:
        d = date(year, month, day)
        return d.isoformat()
    except ValueError:
        return None


def _resolve_column_date_format(
    col_name: str,
    values: List[str],
    vertical: Optional[str],
    decisions: List[Dict],
) -> bool:
    """
    Resolve whether this column uses day-first (DD/MM) or month-first (MM/DD).

    Resolution order (per spec):
      1. Scan whole column — if any value has first segment > 12, day-first confirmed
      2. Column name hints (delivery_date, order_date, etc.)
      3. Vertical convention (logistics/supply_chain → DD/MM, retail → MM/DD)
      4. Global default → DD/MM

    Returns: day_first (bool)
    Side-effect: appends a decision entry to `decisions`.
    """
    parseable = [segs for v in values if (segs := _parse_date_segments(v)) is not None]

    # Step 1: any seg1 > 12 → unambiguously day-first
    forced_day_first = [s for s in parseable if s[0] > 12 and s[0] <= 31]
    if forced_day_first:
        decisions.append({
            "column": col_name,
            "decision": "day_first",
            "reason": f"column scan: {len(forced_day_first)} value(s) with first segment > 12",
        })
        return True

    # Step 2: column name hint
    if _DATE_COL_HINTS.search(col_name):
        # 'order_date', 'delivery_date' → treat as DD/MM (logistics convention);
        # 'created_at', generic → default
        vert_lower = (vertical or "").lower().replace(" ", "_")
        if any(v in vert_lower for v in _DDMM_VERTICALS):
            decisions.append({
                "column": col_name,
                "decision": "day_first",
                "reason": f"column name hint + vertical '{vertical}' convention DD/MM",
            })
            return True
        else:
            decisions.append({
                "column": col_name,
                "decision": "month_first",
                "reason": f"column name hint + vertical '{vertical}' default MM/DD",
            })
            return False

    # Step 3: vertical convention
    vert_lower = (vertical or "").lower().replace(" ", "_")
    if any(v in vert_lower for v in _DDMM_VERTICALS):
        decisions.append({
            "column": col_name,
            "decision": "day_first",
            "reason": f"vertical convention: '{vertical}' uses DD/MM",
        })
        return True
    if vertical:
        # Known vertical that is NOT in _DDMM_VERTICALS (e.g. retail_shopify) → MM/DD
        decisions.append({
            "column": col_name,
            "decision": "month_first",
            "reason": f"vertical convention: '{vertical}' uses MM/DD",
        })
        return False

    # Step 4: global default (no vertical known)
    decisions.append({
        "column": col_name,
        "decision": "day_first",
        "reason": "global default: DD/MM",
    })
    return True


def _standardise_dates_column(
    col_name: str,
    rows: List[Row],
    vertical: Optional[str],
) -> Tuple[List[Row], Dict]:
    """
    Standardise date values in one column to ISO 8601.
    Returns (updated_rows, column_report).
    """
    str_values = [
        str(row[col_name]).strip()
        for row in rows
        if col_name in row and not _value_is_empty(row[col_name])
    ]
    if not str_values:
        return rows, {}

    date_decisions: List[Dict] = []
    day_first = _resolve_column_date_format(col_name, str_values, vertical, date_decisions)

    changed = 0
    failed  = 0
    new_rows = []
    for row in rows:
        row = dict(row)
        v = row.get(col_name)
        if _value_is_empty(v):
            new_rows.append(row)
            continue
        s = str(v).strip()
        if _ISO_RE.match(s):
            new_rows.append(row)    # already ISO — skip
            continue
        segs = _parse_date_segments(s)
        if segs is None:
            new_rows.append(row)
            failed += 1
            continue
        iso = _to_iso(*segs, day_first=day_first)
        if iso:
            row[col_name] = iso
            if iso != s:
                changed += 1
        else:
            failed += 1
        new_rows.append(row)

    col_report = {
        "cells_changed": changed,
        "cells_unparseable": failed,
        "day_first": day_first,
        "date_decisions": date_decisions,
    }
    return new_rows, col_report


# ─────────────────────────────────────────────────────────────────────────────
# Column-level type detection for dates and numbers
# ─────────────────────────────────────────────────────────────────────────────

_DATE_FRAC_THRESHOLD   = 0.60   # ≥60 % of non-empty values look like dates
_NUMBER_FRAC_THRESHOLD = 0.80   # ≥80 % look numeric


def _is_date_string(s: str) -> bool:
    if _ISO_RE.match(s):
        return True
    return _parse_date_segments(s) is not None


def _column_looks_like_dates(rows: List[Row], col: str) -> bool:
    non_empty = [str(row[col]).strip() for row in rows if not _value_is_empty(row.get(col))]
    if not non_empty:
        return False
    hits = sum(1 for v in non_empty if _is_date_string(v))
    return hits / len(non_empty) >= _DATE_FRAC_THRESHOLD


def _column_looks_like_numbers(rows: List[Row], col: str) -> bool:
    non_empty = [str(row[col]).strip() for row in rows if not _value_is_empty(row.get(col))]
    if not non_empty:
        return False
    hits = sum(1 for v in non_empty if _looks_numeric(v))
    return hits / len(non_empty) >= _NUMBER_FRAC_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# Main public API
# ─────────────────────────────────────────────────────────────────────────────

def clean_sheet(
    sheet: Sheet,
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[Sheet, TransformationReport]:
    """
    Apply the full transformation pipeline to a single ExcelSheet dict.

    Args:
        sheet:   ExcelSheet-compatible dict with keys:
                   sheet_name, original_sheet_name, rows, columns, row_count
        options: optional overrides
                   vertical (str) — used for date convention resolution
                   skip_dates (bool) — skip step 6
                   skip_numbers (bool) — skip step 5

    Returns:
        (cleaned_sheet, transformation_report)

        transformation_report keys:
          sheet_name, total_cells_changed, badge_colour,
          steps: {
            headers:      { cells_affected, renames }
            empty_cols:   { columns }
            null_norm:    { cells_affected }
            whitespace:   { cells_affected }
            numbers:      { cells_affected, columns_affected }
            dates:        { cells_affected, columns_affected, decisions }
          }
    """
    opts      = options or {}
    vertical  = opts.get("vertical")
    skip_dates   = opts.get("skip_dates", False)
    skip_numbers = opts.get("skip_numbers", False)

    original_name = sheet.get("original_sheet_name", sheet.get("sheet_name", ""))
    columns = list(sheet.get("columns", []))
    rows    = [dict(r) for r in sheet.get("rows", [])]

    report: Dict[str, Any] = {
        "sheet_name": original_name,
        "total_cells_changed": 0,
        "badge_colour": "green",
        "steps": {},
    }

    # ── Step 1: Column headers ────────────────────────────────────────────────
    cleaned_cols, rename_map, header_affected = _clean_headers(columns)

    # Apply renames to all rows
    if rename_map:
        rows = [
            {rename_map.get(k, k): v for k, v in row.items()}
            for row in rows
        ]
    columns = cleaned_cols

    report["steps"]["headers"] = {
        "cells_affected": header_affected,
        "renames": rename_map,
    }

    # ── Step 2: Empty column detection ───────────────────────────────────────
    empty_cols = _detect_empty_columns(rows, columns)
    report["steps"]["empty_cols"] = {
        "columns": empty_cols,
    }

    # ── Step 3 + 4: Null normalisation + Whitespace cleanup (combined pass) ──
    null_affected  = 0
    ws_affected    = 0

    for row in rows:
        for col in list(row.keys()):
            v = row[col]
            # Null first
            v2, null_changed = _normalise_null(v)
            if null_changed:
                null_affected += 1
                row[col] = v2
                continue          # already "" — no further cleanup needed
            # Whitespace on strings
            if isinstance(v2, str):
                v3, ws_changed = _clean_whitespace(v2)
                if ws_changed:
                    ws_affected += 1
                    row[col] = v3

    report["steps"]["null_norm"]  = {"cells_affected": null_affected}
    report["steps"]["whitespace"] = {"cells_affected": ws_affected}

    # ── Step 5: Number normalisation ─────────────────────────────────────────
    num_affected = 0
    num_cols_affected: List[str] = []

    if not skip_numbers:
        for col in columns:
            if col in empty_cols:
                continue
            if not _column_looks_like_numbers(rows, col):
                continue
            col_changed = 0
            for row in rows:
                v = row.get(col)
                v2, changed = _normalise_number(v)
                if changed:
                    row[col] = v2
                    col_changed += 1
            if col_changed:
                num_affected += col_changed
                num_cols_affected.append(col)

    report["steps"]["numbers"] = {
        "cells_affected":   num_affected,
        "columns_affected": num_cols_affected,
    }

    # ── Step 6: Date standardisation ─────────────────────────────────────────
    date_affected = 0
    date_cols_affected: List[str] = []
    date_decisions: List[Dict]    = []

    if not skip_dates:
        for col in columns:
            if col in empty_cols:
                continue
            if not _column_looks_like_dates(rows, col):
                continue
            rows, col_report = _standardise_dates_column(col, rows, vertical)
            if col_report.get("cells_changed", 0):
                date_affected += col_report["cells_changed"]
                date_cols_affected.append(col)
            date_decisions.extend(col_report.get("date_decisions", []))

    report["steps"]["dates"] = {
        "cells_affected":   date_affected,
        "columns_affected": date_cols_affected,
        "decisions":        date_decisions,
    }

    # ── Badge colour ──────────────────────────────────────────────────────────
    if date_affected or num_affected:
        report["badge_colour"] = "amber"
    else:
        report["badge_colour"] = "green"

    # ── Total cells changed ───────────────────────────────────────────────────
    report["total_cells_changed"] = (
        header_affected + null_affected + ws_affected + num_affected + date_affected
    )

    # ── Build cleaned sheet ───────────────────────────────────────────────────
    cleaned_sheet = dict(sheet)
    cleaned_sheet["columns"]   = columns
    cleaned_sheet["rows"]      = rows
    cleaned_sheet["row_count"] = len(rows)

    log.info(
        f"[data_cleaner] '{original_name}' — "
        f"{report['total_cells_changed']} cell(s) changed "
        f"(badge={report['badge_colour']})"
    )
    return cleaned_sheet, report


# ─────────────────────────────────────────────────────────────────────────────
# Log persistence
# ─────────────────────────────────────────────────────────────────────────────

def write_transformation_log(
    customer_data_dir: Path,
    file_id: str,
    reports: List[TransformationReport],
) -> Path:
    """
    Persist per-upload transformation reports to transformation_log.json.
    Keyed by file_id; each entry summarises all sheets of that upload.

    Returns the path of the written log file.
    """
    log_path = customer_data_dir / "transformation_log.json"
    existing: Dict = {}
    if log_path.exists():
        try:
            existing = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    existing[file_id] = {
        "file_id":  file_id,
        "sheets":   reports,
        "total_cells_changed": sum(r.get("total_cells_changed", 0) for r in reports),
        "badge_colour": (
            "amber" if any(r.get("badge_colour") == "amber" for r in reports) else "green"
        ),
    }

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    log.info(
        f"[data_cleaner] transformation_log.json updated for file_id='{file_id}' "
        f"({len(reports)} sheet(s))"
    )
    return log_path
