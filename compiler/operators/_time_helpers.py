"""
compiler/operators/_time_helpers.py
=====================================
Date parsing and time-window helpers shared by all time-scoped operators.

All helpers work on plain Python datetime — no pandas dependency.
"""

import calendar
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


def parse_date(value) -> Optional[datetime]:
    """
    Best-effort parse of a date/datetime string from a CSV row.
    Tries the most common formats in order. Returns None on failure.
    """
    if not value:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)

    s = str(value).strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%S",   # ISO datetime
        "%Y-%m-%d %H:%M:%S",   # SQL datetime
        "%Y-%m-%d",            # ISO date only
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
    ):
        # Only truncate when the format is a datetime (has time component).
        # For date-only formats, use the full string — truncating by format
        # length gives "2024-03-" for "%Y-%m-%d" (len=8) which fails.
        try:
            candidate = s[:19] if "T" in fmt or " " in fmt else s
            return datetime.strptime(candidate, fmt)
        except (ValueError, IndexError):
            continue
    return None


def now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ── Month helpers ─────────────────────────────────────────────────────────────

def current_year_month(now: Optional[datetime] = None) -> Tuple[int, int]:
    n = now or now_utc()
    return n.year, n.month


def previous_year_month(now: Optional[datetime] = None) -> Tuple[int, int]:
    year, month = current_year_month(now)
    if month == 1:
        return year - 1, 12
    return year, month - 1


def filter_by_month(
    rows: List[Dict], date_col: str, year: int, month: int
) -> List[Dict]:
    out = []
    for r in rows:
        d = parse_date(r.get(date_col))
        if d and d.year == year and d.month == month:
            out.append(r)
    return out


# ── Quarter helpers ───────────────────────────────────────────────────────────

def current_quarter(now: Optional[datetime] = None) -> Tuple[int, int]:
    """Return (year, quarter 1-4)."""
    n = now or now_utc()
    return n.year, (n.month - 1) // 3 + 1


def previous_quarter(now: Optional[datetime] = None) -> Tuple[int, int]:
    year, q = current_quarter(now)
    if q == 1:
        return year - 1, 4
    return year, q - 1


def quarter_date_range(year: int, q: int) -> Tuple[datetime, datetime]:
    first_month = (q - 1) * 3 + 1
    last_month  = first_month + 2
    start = datetime(year, first_month, 1, 0, 0, 0)
    last_day = calendar.monthrange(year, last_month)[1]
    end   = datetime(year, last_month, last_day, 23, 59, 59)
    return start, end


def year_date_range(year: int) -> Tuple[datetime, datetime]:
    return datetime(year, 1, 1, 0, 0, 0), datetime(year, 12, 31, 23, 59, 59)


def filter_by_range(
    rows: List[Dict], date_col: str,
    start: datetime, end: datetime,
) -> List[Dict]:
    out = []
    for r in rows:
        d = parse_date(r.get(date_col))
        if d and start <= d <= end:
            out.append(r)
    return out


# ── YTD ───────────────────────────────────────────────────────────────────────

def filter_ytd(rows: List[Dict], date_col: str) -> List[Dict]:
    now   = now_utc()
    start = datetime(now.year, 1, 1, 0, 0, 0)
    return filter_by_range(rows, date_col, start, now)


# ── Time-bucket key for series grouping ──────────────────────────────────────

def bucket_date_key(value, granularity: str) -> Optional[str]:
    """
    Derive a grouping key for a calendar-bucket series axis.
    Used by MEASURE_GROUPED when the grouping is by_month / by_quarter /
    by_week / by_year rather than a literal data column.

    Returns a sortable string key, e.g. "2026-03" or "2026-Q1".
    """
    d = parse_date(value)
    if not d:
        return None

    if granularity in ("month", "by_month"):
        return d.strftime("%Y-%m")
    if granularity in ("quarter", "by_quarter"):
        q = (d.month - 1) // 3 + 1
        return f"{d.year}-Q{q}"
    if granularity in ("week", "by_week"):
        iso_year, iso_week, _ = d.isocalendar()
        return f"{iso_year}-W{iso_week:02d}"
    if granularity in ("year", "by_year"):
        return str(d.year)
    if granularity in ("day", "by_day"):
        return d.strftime("%Y-%m-%d")
    return None
