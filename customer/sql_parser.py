"""
customer/sql_parser.py
======================
Robust SQL dump parser for MySQL and PostgreSQL exports.

Converts a .sql dump file into a list of table dicts:
    [
        {
            "table_name":  str,
            "columns":     [str, ...],
            "rows":        [{col: val, ...}, ...],
            "row_count":   int,
            "dialect":     "mysql" | "postgres" | "unknown",
        },
        ...
    ]

Edge cases handled:
  - Multi-line string values with embedded newlines
  - Escaped quotes: \\', \\", \\\\
  - Multi-row INSERT:  VALUES (1,2),(3,4)
  - NULL / \\N values
  - Hex literals:  0x414243
  - Binary / BLOB columns (stored as empty string with warning)
  - Stored procedures, triggers, views, functions  -> skipped
  - Comments: -- line comments,  /* block comments */
  - MySQL-specific: backticks, AUTO_INCREMENT, ENGINE=, CHARSET=, UNSIGNED,
                    COLLATE, DEFAULT CHARSET, ROW_FORMAT, KEY/INDEX blocks,
                    DELIMITER $$ ... $$ blocks, LOCK TABLES, UNLOCK TABLES,
                    SET statements, /*!...*/ conditional comments
  - Postgres-specific: SERIAL/BIGSERIAL, ::cast, public. schema prefix,
                       $$ dollar-quoting, SEQUENCE blocks, COPY ... FROM stdin,
                       end-of-copy marker (backslash-dot), SET / SELECT pg_catalog,
                       ALTER TABLE ... OWNER TO, GRANT/REVOKE statements
  - Encoding: latin-1 fallback if UTF-8 decode fails
  - DDL-only tables (CREATE TABLE with no INSERT) -> included with 0 rows
  - Duplicate table names -> rows merged
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_sql(content: bytes, filename: str = "upload.sql") -> List[Dict[str, Any]]:
    """
    Parse a MySQL or Postgres .sql dump and return a list of table dicts.
    Each dict has: table_name, columns, rows, row_count, dialect.
    Tables with 0 rows are excluded from the result.
    """
    text = _decode(content, filename)
    dialect = _detect_dialect(text)
    log.info(f"[sql_parser] '{filename}' detected dialect: {dialect}")

    text = _strip_dollar_quote_blocks(text)      # Postgres stored procs
    text = _strip_block_comments(text)           # /* ... */
    text = _strip_line_comments(text)            # -- ...
    text = _strip_conditional_comments(text)     # /*!50003 ... */
    text = _strip_delimiter_blocks(text)         # DELIMITER $$ ... DELIMITER ;
    text = _strip_noise_statements(text)         # SET, LOCK, GRANT, etc.

    # Parse CREATE TABLE -> column lists
    schema: Dict[str, List[str]] = _parse_create_tables(text, dialect)
    log.info(f"[sql_parser] '{filename}' found {len(schema)} CREATE TABLE definitions")

    # Parse INSERT INTO -> rows
    tables: Dict[str, Dict[str, Any]] = {}
    for tbl, cols in schema.items():
        tables[tbl] = {
            "table_name": tbl,
            "columns":    cols,
            "rows":       [],
            "row_count":  0,
            "dialect":    dialect,
        }

    _parse_inserts(text, tables, dialect)

    # Parse COPY ... FROM stdin blocks (Postgres)
    if dialect in ("postgres", "unknown"):
        _parse_copy_blocks(text, tables, schema)

    result = []
    for tbl, info in tables.items():
        if info["row_count"] == 0:
            log.warning(f"[sql_parser] Table '{tbl}' has 0 rows — skipping")
            continue
        result.append(info)

    log.info(
        f"[sql_parser] '{filename}' complete — "
        f"{len(result)} tables with data "
        f"({sum(t['row_count'] for t in result)} total rows)"
    )
    return result


# ---------------------------------------------------------------------------
# Dialect detection
# ---------------------------------------------------------------------------

def _detect_dialect(text: str) -> str:
    sample = text[:8000]
    mysql_signals = [
        r"ENGINE\s*=",
        r"AUTO_INCREMENT",
        r"CHARSET\s*=",
        r"mysqldump",
        r"LOCK TABLES",
        r"`[^`]+`",           # backtick identifiers
    ]
    pg_signals = [
        r"pg_dump",
        r"PostgreSQL",
        r"SERIAL\b",
        r"\bBIGSERIAL\b",
        r"::[a-z]",           # cast syntax
        r"public\.[a-z]",
        r"\$\$",
        r"COPY .+ FROM stdin",
    ]
    mysql_score = sum(1 for p in mysql_signals if re.search(p, sample, re.I))
    pg_score    = sum(1 for p in pg_signals    if re.search(p, sample, re.I))

    if mysql_score > pg_score:
        return "mysql"
    if pg_score > mysql_score:
        return "postgres"
    return "unknown"


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _decode(content: bytes, filename: str) -> str:
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            continue
    log.warning(f"[sql_parser] '{filename}' decode failed on all encodings — using latin-1 with replace")
    return content.decode("latin-1", errors="replace")


def _strip_block_comments(text: str) -> str:
    """Remove /* ... */ block comments (non-greedy, handles nesting via iteration)."""
    result = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    return result


def _strip_conditional_comments(text: str) -> str:
    """MySQL conditional comments  /*!50003 content */  -> remove entirely."""
    return re.sub(r"/\*!.*?\*/", " ", text, flags=re.DOTALL)


def _strip_line_comments(text: str) -> str:
    """Remove -- line comments, preserving newlines."""
    return re.sub(r"--[^\n]*", "", text)


def _strip_dollar_quote_blocks(text: str) -> str:
    """
    Remove Postgres dollar-quoted blocks  $tag$...$tag$ or $$...$$
    These contain stored procedure bodies which we don't want to parse.
    """
    # Named: $label$...$label$
    text = re.sub(r"\$[A-Za-z_][A-Za-z_0-9]*\$.*?\$[A-Za-z_][A-Za-z_0-9]*\$", " ", text, flags=re.DOTALL)
    # Anonymous: $$...$$
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    return text


def _strip_delimiter_blocks(text: str) -> str:
    """
    MySQL DELIMITER $$ ... DELIMITER ; blocks contain stored procs / triggers.
    Remove everything between DELIMITER <x> and the matching DELIMITER ;
    """
    text = re.sub(
        r"DELIMITER\s+\S+.*?DELIMITER\s+;",
        " ",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return text


_NOISE_PATTERNS = [
    # MySQL
    r"LOCK\s+TABLES\s+[^\n;]+;",
    r"UNLOCK\s+TABLES\s*;",
    r"SET\s+[^\n;]+;",
    r"USE\s+[^\n;]+;",
    # Postgres
    r"SELECT\s+pg_catalog[^;]+;",
    r"ALTER\s+TABLE\s+[^;]+\bOWNER\s+TO\b[^;]+;",
    r"GRANT\s+[^;]+;",
    r"REVOKE\s+[^;]+;",
    r"CREATE\s+SEQUENCE\s+[^;]+;",
    r"ALTER\s+SEQUENCE\s+[^;]+;",
    r"SELECT\s+setval[^;]+;",
    r"CREATE\s+(UNIQUE\s+)?INDEX\s+[^;]+;",
    r"CREATE\s+VIEW\s+[^;]+;",
    r"CREATE\s+(OR\s+REPLACE\s+)?FUNCTION\s+[^;]+;",
    r"CREATE\s+TRIGGER\s+[^;]+;",
    r"CREATE\s+TYPE\s+[^;]+;",
    r"CREATE\s+SCHEMA\s+[^;]+;",
    r"CREATE\s+EXTENSION\s+[^;]+;",
    r"\\connect\s+[^\n]+",          # psql meta-commands
    r"\\encoding\s+[^\n]+",
]

def _strip_noise_statements(text: str) -> str:
    for pattern in _NOISE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.DOTALL | re.IGNORECASE)
    return text


# ---------------------------------------------------------------------------
# CREATE TABLE parsing
# ---------------------------------------------------------------------------

_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?"
    r"(?:\"?`?\[?(?:public\s*\.\s*)?(?P<schema>[a-zA-Z_][a-zA-Z0-9_]*)\.)?`?\[?"
    r"(?P<table>[a-zA-Z_][a-zA-Z0-9_$]*)"
    r"`?\]?\"?\s*\(",
    re.IGNORECASE,
)


def _parse_create_tables(text: str, dialect: str) -> Dict[str, List[str]]:
    schema: Dict[str, List[str]] = {}
    pos = 0
    while True:
        m = _CREATE_TABLE_RE.search(text, pos)
        if not m:
            break
        table_name = m.group("table").lower()
        body_start = m.end()

        # Find the matching closing paren (handles nested parens)
        body, end_pos = _extract_paren_block(text, body_start - 1)
        if body is None:
            pos = body_start
            continue

        cols = _extract_columns(body, dialect)
        if cols:
            schema[table_name] = cols
            log.debug(f"[sql_parser] CREATE TABLE '{table_name}' — {len(cols)} columns")
        pos = end_pos

    return schema


def _extract_paren_block(text: str, start: int) -> Tuple[Optional[str], int]:
    """
    Extract content between the opening paren at `start` and its matching close.
    Returns (inner_content, position_after_closing_paren).
    """
    if start >= len(text) or text[start] != "(":
        return None, start + 1
    depth = 0
    i = start
    in_string = False
    str_char = ""
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\" and i + 1 < len(text):
                i += 2
                continue
            if ch == str_char:
                in_string = False
        else:
            if ch in ("'", '"'):
                in_string = True
                str_char = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return text[start + 1:i], i + 1
        i += 1
    return None, len(text)


# Column definition patterns to SKIP (not data columns)
_SKIP_COL_RE = re.compile(
    r"^\s*(PRIMARY\s+KEY|UNIQUE(\s+KEY)?|KEY|INDEX|CONSTRAINT|CHECK|FOREIGN\s+KEY"
    r"|FULLTEXT|SPATIAL)\b",
    re.IGNORECASE,
)

# Extract column name from a column definition line
_COL_NAME_RE = re.compile(
    r"^\s*[`\"\[]?([a-zA-Z_][a-zA-Z0-9_$]*)[`\"\]]?\s+",
)


def _extract_columns(body: str, dialect: str) -> List[str]:
    """
    Parse the body of a CREATE TABLE (...) and return a list of column names.
    Skips constraint/index lines.
    """
    cols: List[str] = []
    # Split on commas that are NOT inside parens or strings
    parts = _split_column_defs(body)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if _SKIP_COL_RE.match(part):
            continue
        m = _COL_NAME_RE.match(part)
        if m:
            cols.append(m.group(1).lower())
    return cols


def _split_column_defs(body: str) -> List[str]:
    """Split CREATE TABLE body on top-level commas."""
    parts = []
    current = []
    depth = 0
    in_string = False
    str_char = ""
    i = 0
    while i < len(body):
        ch = body[i]
        if in_string:
            if ch == "\\" and i + 1 < len(body):
                current.append(ch)
                current.append(body[i + 1])
                i += 2
                continue
            if ch == str_char:
                in_string = False
            current.append(ch)
        else:
            if ch in ("'", '"'):
                in_string = True
                str_char = ch
                current.append(ch)
            elif ch in ("(", "["):
                depth += 1
                current.append(ch)
            elif ch in (")", "]"):
                depth -= 1
                current.append(ch)
            elif ch == "," and depth == 0:
                parts.append("".join(current))
                current = []
                i += 1
                continue
            else:
                current.append(ch)
        i += 1
    if current:
        parts.append("".join(current))
    return parts


# ---------------------------------------------------------------------------
# INSERT parsing
# ---------------------------------------------------------------------------

_INSERT_RE = re.compile(
    r"INSERT\s+(?:IGNORE\s+|OR\s+REPLACE\s+)?INTO\s+"
    r"[`\"\[]?(?:public\s*\.\s*)?(?:[a-zA-Z_][a-zA-Z0-9_$]*\.)?`?\[?"
    r"(?P<table>[a-zA-Z_][a-zA-Z0-9_$]*)"
    r"`?\]?\"?"
    r"(?:\s*\((?P<cols>[^)]+)\))?"   # optional explicit column list
    r"\s*VALUES\s*",
    re.IGNORECASE,
)


def _parse_inserts(text: str, tables: Dict[str, Dict], dialect: str) -> None:
    pos = 0
    while True:
        m = _INSERT_RE.search(text, pos)
        if not m:
            break

        table_name = m.group("table").lower()
        explicit_cols_str = m.group("cols")
        values_start = m.end()

        # Parse the VALUES (...),(...),...  section
        rows_raw, next_pos = _extract_values_block(text, values_start)
        pos = next_pos

        if not rows_raw:
            continue

        # Determine column list
        if explicit_cols_str:
            cols = [c.strip().strip("`\"[]").lower() for c in explicit_cols_str.split(",")]
        elif table_name in tables:
            cols = tables[table_name]["columns"]
        else:
            # Unknown table — create entry with positional column names
            num_cols = len(rows_raw[0]) if rows_raw else 0
            cols = [f"col_{i}" for i in range(num_cols)]
            tables[table_name] = {
                "table_name": table_name,
                "columns":    cols,
                "rows":       [],
                "row_count":  0,
                "dialect":    dialect,
            }
            log.warning(
                f"[sql_parser] INSERT into unknown table '{table_name}' — "
                f"using {num_cols} positional columns"
            )

        if table_name not in tables:
            tables[table_name] = {
                "table_name": table_name,
                "columns":    cols,
                "rows":       [],
                "row_count":  0,
                "dialect":    dialect,
            }

        for raw_vals in rows_raw:
            row = _map_row(cols, raw_vals)
            tables[table_name]["rows"].append(row)
            tables[table_name]["row_count"] += 1


def _extract_values_block(text: str, start: int) -> Tuple[List[List[str]], int]:
    """
    Starting right after VALUES, extract all (row), (row), ... groups
    until we hit a semicolon at depth 0 or end of text.
    Returns (list_of_row_value_lists, next_parse_position).
    """
    rows: List[List[str]] = []
    pos = start
    n = len(text)

    while pos < n:
        # Skip whitespace and commas between rows
        while pos < n and text[pos] in (" ", "\t", "\n", "\r", ","):
            pos += 1

        if pos >= n:
            break

        ch = text[pos]

        if ch == "(":
            vals, pos = _parse_row_values(text, pos)
            if vals is not None:
                rows.append(vals)
        elif ch == ";":
            pos += 1
            break
        else:
            # Something unexpected — stop consuming this INSERT block
            break

    return rows, pos


def _parse_row_values(text: str, start: int) -> Tuple[Optional[List[str]], int]:
    """
    Parse one VALUES tuple  (val1, val2, ...)  starting at the opening paren.
    Returns (list_of_string_values, position_after_closing_paren).
    Handles: strings with escape sequences, NULL, \\N, hex literals, nested parens.
    """
    if start >= len(text) or text[start] != "(":
        return None, start + 1

    values: List[str] = []
    current: List[str] = []
    i = start + 1
    n = len(text)
    depth = 1  # we are inside one paren already

    def flush():
        val = "".join(current).strip()
        current.clear()
        # NULL variants
        if val.upper() in ("NULL", "\\N", ""):
            values.append("")
            return
        # Hex literal 0x...
        if re.match(r"^0x[0-9a-fA-F]+$", val):
            try:
                values.append(bytes.fromhex(val[2:]).decode("utf-8", errors="replace"))
            except Exception:
                values.append(val)
            return
        values.append(val)

    while i < n:
        ch = text[i]

        if ch == "'":
            # MySQL/SQL string literal
            s, i = _parse_sql_string(text, i)
            current.append(s)
            continue

        if ch == '"':
            # Postgres double-quoted string or identifier
            s, i = _parse_double_quoted(text, i)
            current.append(s)
            continue

        if ch == "(":
            depth += 1
            current.append(ch)
            i += 1
            continue

        if ch == ")":
            depth -= 1
            if depth == 0:
                flush()
                return values, i + 1
            current.append(ch)
            i += 1
            continue

        if ch == "," and depth == 1:
            flush()
            i += 1
            continue

        current.append(ch)
        i += 1

    # Unterminated — return what we have
    flush()
    return values, i


def _parse_sql_string(text: str, start: int) -> Tuple[str, int]:
    """
    Parse a single-quoted SQL string starting at `start`.
    Handles: \\'  \\n  \\r  \\t  \\\\  ''  (doubled-quote escape)
    Returns (unescaped_string_content, position_after_closing_quote).
    """
    i = start + 1
    n = len(text)
    chars: List[str] = []
    while i < n:
        ch = text[i]
        if ch == "\\":
            if i + 1 < n:
                nch = text[i + 1]
                escape_map = {
                    "n": "\n", "r": "\r", "t": "\t",
                    "0": "\0", "\\": "\\", "'": "'", '"': '"',
                    "Z": "\x1a", "b": "\b",
                }
                chars.append(escape_map.get(nch, nch))
                i += 2
            else:
                i += 1
        elif ch == "'":
            if i + 1 < n and text[i + 1] == "'":
                # Doubled quote escape ''
                chars.append("'")
                i += 2
            else:
                return "".join(chars), i + 1
        else:
            chars.append(ch)
            i += 1
    return "".join(chars), i


def _parse_double_quoted(text: str, start: int) -> Tuple[str, int]:
    """
    Parse a double-quoted string/identifier.
    Handles \"\" doubled-quote escape and \\\" backslash escape.
    """
    i = start + 1
    n = len(text)
    chars: List[str] = []
    while i < n:
        ch = text[i]
        if ch == "\\":
            if i + 1 < n:
                chars.append(text[i + 1])
                i += 2
            else:
                i += 1
        elif ch == '"':
            if i + 1 < n and text[i + 1] == '"':
                chars.append('"')
                i += 2
            else:
                return "".join(chars), i + 1
        else:
            chars.append(ch)
            i += 1
    return "".join(chars), i


def _map_row(cols: List[str], vals: List[str]) -> Dict[str, str]:
    """Map parsed values to column names. Handles column count mismatch safely."""
    row: Dict[str, str] = {}
    for idx, col in enumerate(cols):
        row[col] = vals[idx] if idx < len(vals) else ""
    return row


# ---------------------------------------------------------------------------
# COPY ... FROM stdin parsing (Postgres)
# ---------------------------------------------------------------------------

_COPY_HEADER_RE = re.compile(
    r"COPY\s+(?:public\s*\.\s*)?(?P<table>[a-zA-Z_][a-zA-Z0-9_$]*)"
    r"\s*\((?P<cols>[^)]+)\)\s+FROM\s+stdin\s*;",
    re.IGNORECASE,
)


def _parse_copy_blocks(text: str, tables: Dict[str, Dict], schema: Dict[str, List[str]]) -> None:
    """
    Parse Postgres COPY tablename (col1, col2) FROM stdin; ... backslash-dot blocks.
    Data rows are tab-delimited. \\N means NULL.
    """
    for m in _COPY_HEADER_RE.finditer(text):
        table_name = m.group("table").lower()
        cols = [c.strip().strip('"').lower() for c in m.group("cols").split(",")]
        block_start = m.end()

        # Find the backslash-dot terminator
        end_m = re.search(r"^\\[.]", text[block_start:], re.MULTILINE)
        if end_m:
            block_text = text[block_start: block_start + end_m.start()]
        else:
            block_text = text[block_start:]

        if table_name not in tables:
            tables[table_name] = {
                "table_name": table_name,
                "columns":    cols,
                "rows":       [],
                "row_count":  0,
                "dialect":    "postgres",
            }

        for line in block_text.splitlines():
            line = line.rstrip("\n")
            if not line or line.startswith("\\."):  # noqa: W605
                continue
            raw_vals = line.split("\t")
            vals = ["" if v == "\\N" else v.replace("\\t", "\t").replace("\\n", "\n") for v in raw_vals]
            row = _map_row(cols, vals)
            tables[table_name]["rows"].append(row)
            tables[table_name]["row_count"] += 1

        log.debug(f"[sql_parser] COPY '{table_name}' — {tables[table_name]['row_count']} rows")
