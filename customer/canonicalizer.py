"""
customer/canonicalizer.py
==========================
Maps raw Excel column headers to factory canonical field names.

Resolution order per column:
  1. Exact match against dialect column_hints (term → canonical, case-insensitive)
  2. Normalised snake_case match (spaces/dashes → underscores)
  3. Atom field list fuzzy match (loaded from atoms.json for the matched atom)
  4. Falls back to plain snake_case (_clean_col) — never fails

The factory endpoint POST /api/factory/canonicalize delegates here.
The customer-side function canonicalize_headers() can also call the factory
endpoint directly when running in the runtime process; if that call fails it
falls back to the local pure-Python resolver without any user-visible error.

Public API
----------
  canonicalize_headers(columns, vertical, atom_canonical_id=None)
      → Dict[str, str]   {original_col: canonical_col}

  resolve_column(raw_name, col_hints_index)
      → str              canonical name (or normalised snake_case fallback)

  build_col_hints_index(vertical)
      → Dict[str, str]   {normalised_term: canonical}   built from dialect
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_col(name: str) -> str:
    """Normalise a column name to lowercase snake_case."""
    return re.sub(r"[^a-z0-9_]", "_", name.lower().strip()).strip("_") or "col"


def _normalise_term(term: str) -> str:
    """
    Normalise a vocabulary term for index lookup.
    Collapses whitespace, converts separators to underscores, lowercases.
    """
    s = term.strip().lower()
    s = re.sub(r"[\s\-/\\]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


# ── Dialect index builder ─────────────────────────────────────────────────────

def build_col_hints_index(vertical: str) -> Dict[str, str]:
    """
    Load the dialect file for `vertical` and build a lookup dict from
    the `column_hints` vocabulary section.

    Returns: {normalised_term: canonical_field_name}

    Falls back to {} if the dialect file is missing or malformed.
    """
    try:
        from core.paths import DIALECTS_DIR
        dialect_path = DIALECTS_DIR / f"{vertical}_dialect.json"
        if not dialect_path.exists():
            log.debug(f"[canonicalizer] No dialect file for vertical '{vertical}'")
            return {}

        import json
        data = json.loads(dialect_path.read_text(encoding="utf-8"))
        vocab = data.get("vocabulary", {})
        hints = vocab.get("column_hints", [])

        index: Dict[str, str] = {}
        for entry in hints:
            term      = entry.get("term", "")
            canonical = entry.get("canonical", "")
            if term and canonical:
                index[_normalise_term(term)] = canonical

        log.debug(f"[canonicalizer] Loaded {len(index)} column_hints for '{vertical}'")
        return index

    except Exception as exc:
        log.warning(f"[canonicalizer] Could not build col_hints_index for '{vertical}': {exc}")
        return {}


# ── Per-column resolver ───────────────────────────────────────────────────────

def resolve_column(raw_name: str, col_hints_index: Dict[str, str]) -> str:
    """
    Resolve one raw column name to its canonical field name.

    Resolution order:
      1. Exact normalised match in col_hints_index
      2. Normalised snake_case (plain _clean_col fallback)

    Never raises.
    """
    if not raw_name or not raw_name.strip():
        return "col"

    normalised = _normalise_term(raw_name)

    # 1. Direct hit in dialect index
    if normalised in col_hints_index:
        canonical = col_hints_index[normalised]
        log.debug(f"[canonicalizer] '{raw_name}' → '{canonical}' (dialect hit)")
        return canonical

    # 2. snake_case fallback
    snake = _clean_col(raw_name)
    # Check if snake_case version itself hits the index
    if snake in col_hints_index:
        canonical = col_hints_index[snake]
        log.debug(f"[canonicalizer] '{raw_name}' → '{canonical}' (snake hit)")
        return canonical

    log.debug(f"[canonicalizer] '{raw_name}' → '{snake}' (fallback)")
    return snake


# ── Atom field overlay ────────────────────────────────────────────────────────

def _atom_fields_index(atom_canonical_id: str) -> Dict[str, str]:
    """
    Load the field list for a specific atom and return
    {normalised_field_name: canonical_field_name}.

    Used as a secondary layer on top of the dialect column_hints.
    """
    if not atom_canonical_id:
        return {}
    try:
        import json
        from core.paths import ATOMS_PATH
        raw   = json.loads(ATOMS_PATH.read_text(encoding="utf-8"))
        atoms = raw.get("_default", raw) if isinstance(raw, dict) else {}
        for atom in atoms.values():
            if atom.get("canonical_id") != atom_canonical_id:
                continue
            index: Dict[str, str] = {}
            for f in atom.get("fields", []):
                fname = f.get("name", "").strip()
                if fname:
                    index[_normalise_term(fname)] = fname
                    index[_clean_col(fname)]       = fname
            return index
    except Exception as exc:
        log.debug(f"[canonicalizer] Could not load atom fields for '{atom_canonical_id}': {exc}")
    return {}


# ── Main public function ──────────────────────────────────────────────────────

def canonicalize_headers(
    columns:           List[str],
    vertical:          str,
    atom_canonical_id: Optional[str] = None,
) -> Dict[str, str]:
    """
    Map a list of raw column headers to factory canonical field names.

    Args:
        columns:           Original column names from the Excel sheet
        vertical:          Detected vertical (e.g. "logistics", "supply_chain")
        atom_canonical_id: Best-matched atom (optional; improves precision)

    Returns:
        {original_col: canonical_col}  — every input column is present.
        Columns with no dialect/atom match get plain snake_case names.
        Duplicate canonical names get a _N suffix so no key collides.
    """
    col_hints = build_col_hints_index(vertical)
    atom_idx  = _atom_fields_index(atom_canonical_id) if atom_canonical_id else {}

    result: Dict[str, str] = {}
    seen:   Dict[str, int] = {}   # tracks how many times a canonical name has appeared

    for col in columns:
        if not col or not str(col).strip():
            # Blank header — generate a placeholder
            placeholder = f"col_{len(result)}"
            result[col] = placeholder
            continue

        normalised = _normalise_term(col)
        snake      = _clean_col(col)

        # Resolution priority: dialect → atom → snake_case
        canonical = (
            col_hints.get(normalised)
            or col_hints.get(snake)
            or atom_idx.get(normalised)
            or atom_idx.get(snake)
            or snake
        )

        # De-duplicate: if the canonical name already appeared, suffix with _N
        if canonical in seen:
            seen[canonical] += 1
            canonical = f"{canonical}_{seen[canonical]}"
        else:
            seen[canonical] = 0

        result[col] = canonical

    mapped = sum(1 for o, c in result.items() if c != _clean_col(o))
    if mapped:
        log.info(
            f"[canonicalizer] {mapped}/{len(columns)} columns remapped "
            f"(vertical='{vertical}', atom='{atom_canonical_id}')"
        )

    return result


# ── Factory-endpoint caller (runtime → factory) ───────────────────────────────

def canonicalize_via_factory(
    columns:           List[str],
    vertical:          str,
    atom_canonical_id: Optional[str] = None,
    factory_url:       str = "http://localhost:8080",
    timeout:           float = 5.0,
) -> Dict[str, str]:
    """
    Call the factory's POST /api/factory/canonicalize endpoint.
    Falls back to local canonicalize_headers() on any error — the upload
    never fails because of a canonicalization issue.

    Returns: {original_col: canonical_col}
    """
    try:
        import httpx
        payload = {
            "columns":           columns,
            "vertical":          vertical,
            "atom_canonical_id": atom_canonical_id,
        }
        resp = httpx.post(
            f"{factory_url}/api/factory/canonicalize",
            json=payload,
            timeout=timeout,
        )
        if resp.status_code == 200:
            data = resp.json()
            canon_map = data.get("canonical_map", {})
            if canon_map and isinstance(canon_map, dict):
                log.info(
                    f"[canonicalizer] Factory endpoint returned {len(canon_map)} mappings "
                    f"for vertical='{vertical}'"
                )
                return canon_map
            log.warning("[canonicalizer] Factory returned empty canonical_map — falling back")
    except Exception as exc:
        log.warning(f"[canonicalizer] Factory canonicalize call failed (non-fatal): {exc}")

    # Local fallback
    return canonicalize_headers(columns, vertical, atom_canonical_id)


def detect_vertical_via_factory(
    table_name:         str,
    columns:            List[str],
    candidate_domains:  List[str],
    factory_url:        str   = "http://localhost:8080",
    timeout:            float = 5.0,
) -> Optional[str]:
    """
    Call the factory's POST /api/factory/detect-vertical endpoint.

    Used by sot_ingestion.detect_vertical() as the LLM fallback when keyword
    voting can't decide (zero signal, weak signal, or a tie).

    Sends only table name + normalised column names — no customer row data.
    candidate_domains is pre-scoped by the caller (tied domains or all domains).

    Returns the chosen domain string, or None if the call fails.
    On failure the caller falls back to a keyword best-guess with low_confidence=True.
    """
    try:
        import httpx
        payload = {
            "table_name":        table_name,
            "columns":           columns,
            "candidate_domains": candidate_domains,
        }
        resp = httpx.post(
            f"{factory_url}/api/factory/detect-vertical",
            json    = payload,
            timeout = timeout,
        )
        if resp.status_code == 200:
            chosen = resp.json().get("vertical", "").strip().lower()
            if chosen and chosen in candidate_domains:
                log.info(
                    f"[canonicalizer] detect_vertical_via_factory: "
                    f"'{table_name}' → '{chosen}'"
                )
                return chosen
            log.warning(
                f"[canonicalizer] detect_vertical_via_factory: "
                f"unexpected response {resp.json()!r} — falling back"
            )
        else:
            log.warning(
                f"[canonicalizer] detect_vertical_via_factory: "
                f"HTTP {resp.status_code} — falling back"
            )
    except Exception as exc:
        log.warning(
            f"[canonicalizer] detect_vertical_via_factory failed (non-fatal): {exc}"
        )
    return None
