"""
customer/excel_parser.py
=========================
Parses Excel workbooks (.xlsx / .xls) into the same flat-row format that
ingest_csv() already consumes — so every sheet becomes one virtual "CSV table".

Logic ported from BizzBrain app/workflows/excel/excel_parser.py with GPL-
specific changes:
  - Sheet names are normalised to snake_case table names (not preserved as-is)
  - Sheets with fewer than MIN_ROWS data rows are skipped (cover/summary sheets)
  - All-NaN rows and all-NaN columns are dropped (BizzBrain behaviour)
  - File size is capped at MAX_FILE_MB before any parsing starts
  - No domain detection, canonicalization, or atom-ID logic here — those are
    handled downstream by detect_vertical() + ingest_csv() exactly as for CSVs

Public API
----------
  parse_excel(file_bytes, filename)
      → List[ExcelSheet]   one entry per non-empty sheet

  ExcelSheet  (TypedDict)
      sheet_name   str          normalised snake_case name used as table_name
      original_sheet_name  str  raw sheet name from workbook
      rows         List[dict]   data rows with original column names as keys
      columns      List[str]    original column names (header row)
      row_count    int
"""

import io
import logging
import re
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_FILE_MB = 20
MIN_ROWS    = 1   # sheets with 0 data rows (header-only or totally empty) are dropped


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sheet_name_to_snake(name: str) -> str:
    """
    Convert an arbitrary Excel sheet name to a snake_case table name.

    Examples:
      'Fleet Operations'  → 'fleet_operations'
      'Supply-Chain Data' → 'supply_chain_data'
      'Sheet1'            → 'sheet1'
      'Q3 P&L (USD)'      → 'q3_pl_usd'
    """
    name = str(name).strip().lower()
    # Replace common separators and special chars with underscores
    name = re.sub(r"[\s\-&()/\\]+", "_", name)
    # Strip anything that is not alphanumeric or underscore
    name = re.sub(r"[^a-z0-9_]", "", name)
    # Collapse multiple underscores
    name = re.sub(r"_+", "_", name)
    # Strip leading/trailing underscores
    name = name.strip("_")
    # Fallback for names that become empty after cleaning (e.g. "---")
    return name or "sheet"


def _is_empty_value(v: Any) -> bool:
    """True for NaN, None, and empty strings."""
    if v is None:
        return True
    if isinstance(v, float):
        import math
        return math.isnan(v)
    if isinstance(v, str):
        return v.strip() == ""
    return False


def _row_is_all_empty(row: Dict[str, Any]) -> bool:
    return all(_is_empty_value(v) for v in row.values())


# ── Main parser ────────────────────────────────────────────────────────────────

class ExcelSheet(dict):
    """
    Dict subclass so callers can use either attribute or key access.
    Keys: sheet_name, original_sheet_name, rows, columns, row_count
    """


def parse_excel(file_bytes: bytes, filename: str) -> List[ExcelSheet]:
    """
    Parse every sheet in an Excel workbook.

    Args:
        file_bytes:  raw bytes of the .xlsx / .xls file
        filename:    original filename (used only for logging)

    Returns:
        List of ExcelSheet dicts, one per non-empty sheet, in workbook order.
        Sheets with 0 data rows after cleaning are silently dropped.

    Raises:
        ValueError: file too large, unreadable, or wrong format
    """
    # ── Size guard ─────────────────────────────────────────────────────────────
    mb = len(file_bytes) / (1024 * 1024)
    if mb > MAX_FILE_MB:
        raise ValueError(
            f"File '{filename}' is {mb:.1f} MB — maximum allowed is {MAX_FILE_MB} MB."
        )

    # ── Open workbook ──────────────────────────────────────────────────────────
    try:
        import pandas as pd
        xl = pd.ExcelFile(io.BytesIO(file_bytes))
    except Exception as e:
        raise ValueError(f"Cannot read '{filename}' as an Excel file: {e}") from e

    log.info(
        f"[excel_parser] '{filename}' — {len(xl.sheet_names)} sheet(s): "
        f"{xl.sheet_names}"
    )

    results: List[ExcelSheet] = []

    for raw_sheet_name in xl.sheet_names:
        try:
            df = xl.parse(raw_sheet_name)
        except Exception as e:
            log.warning(f"[excel_parser] Could not parse sheet '{raw_sheet_name}': {e} — skipping")
            continue

        # ── Drop all-NaN rows and all-NaN columns (BizzBrain behaviour) ────────
        df.dropna(how="all", inplace=True)
        df.dropna(axis=1, how="all", inplace=True)

        if df.empty:
            log.info(f"[excel_parser] Sheet '{raw_sheet_name}' is empty after NaN drop — skipping")
            continue

        # ── Normalise column names to strings (pandas may produce int headers) ─
        df.columns = [str(c) for c in df.columns]

        # ── Convert rows to plain dicts; stringify values for consistent typing ─
        original_cols = list(df.columns)
        rows: List[Dict[str, Any]] = []

        for _, row in df.iterrows():
            record = {col: row[col] for col in original_cols}
            # Skip rows that are entirely empty/NaN
            if _row_is_all_empty(record):
                continue
            # Normalise every cell value for downstream compatibility
            for k, v in record.items():
                if _is_empty_value(v):
                    record[k] = ""
                else:
                    # pandas.Timestamp (Excel date cells) must be converted to
                    # "YYYY-MM-DD" — NOT the default str() which produces
                    # "YYYY-MM-DD HH:MM:SS".  The time component breaks
                    # _infer_type()'s date regex, causing date columns to be
                    # classified as "string" instead of "date".
                    try:
                        import pandas as _pd
                        if isinstance(v, _pd.Timestamp):
                            record[k] = v.strftime("%Y-%m-%d")
                            continue
                    except ImportError:
                        pass
                    # Keep numeric types (int/float) as-is so _infer_type()
                    # can detect them correctly; leave strings unchanged.
            rows.append(record)

        if len(rows) < MIN_ROWS:
            log.info(
                f"[excel_parser] Sheet '{raw_sheet_name}' has {len(rows)} data row(s) "
                f"(minimum {MIN_ROWS}) — skipping"
            )
            continue

        snake_name = _sheet_name_to_snake(raw_sheet_name)

        log.info(
            f"[excel_parser] Sheet '{raw_sheet_name}' → table '{snake_name}': "
            f"{len(rows)} rows, {len(original_cols)} cols"
        )

        results.append(ExcelSheet({
            "sheet_name":          snake_name,
            "original_sheet_name": raw_sheet_name,
            "rows":                rows,
            "columns":             original_cols,
            "row_count":           len(rows),
        }))

    log.info(
        f"[excel_parser] '{filename}' parsed — "
        f"{len(results)} usable sheet(s) out of {len(xl.sheet_names)}"
    )
    return results
