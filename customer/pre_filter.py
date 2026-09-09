"""
customer/pre_filter.py
=======================
Pre-filter fast-track for repeated file uploads.

When a customer re-uploads a file whose column fingerprint was seen before,
this module short-circuits detect_vertical() and returns the cached detection
result — including canonical_map and grain_keys — with zero LLM calls.

If the cache misses, returns None and the caller falls through to the full
detect_vertical() + atom matching pipeline as normal.

Usage (in customer_router.py upload handlers):

    from customer.pre_filter import check_pre_filter, build_cache_payload

    cached = check_pre_filter(customer_data_dir, normalised_columns)
    if cached:
        detection    = cached["detection"]
        canonical_map = cached["canonical_map"]
    else:
        detection    = detect_vertical(table_name, columns)
        canonical_map = build_canonical_map(columns, detection.get("atom_canonical_id"))
        # ... after ingest_csv, store in cache:
        record_upload(..., sheets=[build_cache_payload(sheet_info)])
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


def check_pre_filter(
    customer_data_dir:  Path,
    normalised_columns: List[str],
) -> Optional[Dict[str, Any]]:
    """
    Look up the column cache for this normalised column set.

    Returns a dict on cache HIT:
        {
            "detection":      { vertical, atom_canonical_id, confidence,
                                method, low_confidence, scores, atom_overlap },
            "canonical_map":  { original_col: canonical_col, ... },
            "grain_keys":     [ ... ],
            "hit_count":      int,
        }

    Returns None on cache MISS — caller should run detect_vertical().
    """
    from customer.upload_registry import lookup_column_cache

    cached = lookup_column_cache(customer_data_dir, normalised_columns)
    if cached is None:
        return None

    # Reconstruct a detection dict from the cache entry
    detection = {
        "vertical":          cached["vertical"],
        "atom_canonical_id": cached.get("atom_canonical_id"),
        "confidence":        cached.get("confidence", 0.0),
        "method":            "column_cache",
        "low_confidence":    cached.get("confidence", 0.0) < 0.60,
        "scores":            cached.get("scores", {}),
        "atom_overlap":      cached.get("confidence", 0.0),
    }

    canonical_map = cached.get("canonical_map", {})
    grain_keys    = cached.get("grain_keys", [])

    log.info(
        f"[pre_filter] Cache HIT → vertical={detection['vertical']} "
        f"atom={detection['atom_canonical_id']} "
        f"canonical_map_entries={len(canonical_map)} "
        f"grain_keys={grain_keys}"
    )

    return {
        "detection":     detection,
        "canonical_map": canonical_map,
        "grain_keys":    grain_keys,
    }


def build_cache_payload(
    sheet_name:          str,
    original_sheet_name: str,
    original_columns:    List[str],
    normalised_columns:  List[str],
    detection:           Dict[str, Any],
    canonical_map:       Dict[str, str],
    grain_keys:          List[str],
    row_count:           int,
    content_hash:        str = "",
) -> Dict[str, Any]:
    """
    Build the sheet payload dict that record_upload() expects,
    including the canonical_map, grain_keys, and content_hash for future
    cache hits and duplicate detection.
    """
    return {
        "sheet_name":          sheet_name,
        "original_sheet_name": original_sheet_name,
        "columns":             original_columns,
        "normalised_columns":  normalised_columns,
        "vertical":            detection.get("vertical", "unknown"),
        "atom_canonical_id":   detection.get("atom_canonical_id"),
        "confidence":          detection.get("confidence", 0.0),
        "canonical_map":       canonical_map,
        "grain_keys":          grain_keys,
        "row_count":           row_count,
        "content_hash":        content_hash,
    }


def resolve_grain_keys(atom_canonical_id: Optional[str]) -> List[str]:
    """
    Look up grain_keys for an atom from atoms.json.
    Returns [] if atom not found or atoms.json unreadable.
    Used when building the initial cache entry after a full detect_vertical() run.
    """
    if not atom_canonical_id:
        return []
    try:
        import json
        from core.paths import ATOMS_PATH
        raw   = json.loads(ATOMS_PATH.read_text(encoding="utf-8"))
        atoms = raw.get("_default", raw) if isinstance(raw, dict) else {}
        for atom in atoms.values():
            if atom.get("canonical_id") == atom_canonical_id:
                return atom.get("grain_keys", [])
    except Exception as exc:
        log.debug(f"[pre_filter] Could not resolve grain_keys for '{atom_canonical_id}': {exc}")
    return []
