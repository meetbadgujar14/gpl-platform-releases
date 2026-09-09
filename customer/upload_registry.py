"""
customer/upload_registry.py
============================
Per-customer upload audit log — drives two features:

  1. DUPLICATE DETECTION
     Before processing a file, check if the exact same filename + sheet names
     + normalised column structure was already ingested. If yes, stop immediately
     and return a duplicate response instead of re-ingesting.

     Note: re-uploads of the SAME file with NEW/CHANGED ROWS are NOT treated
     as duplicates — they fall through to ingest_csv() → upsert_rows() so the
     row-level diff is applied. Only identical structure + identical filename
     stops the pipeline.

  2. COLUMN CACHE (fast-track)
     If a sheet's sorted normalised column fingerprint was seen before, skip
     detect_vertical() and reuse the stored vertical + atom_canonical_id.
     Saves the atoms.json scan on every re-upload.

One JSON file per customer:
  customer_runtime/customers/{customer_id}/data/upload_registry.json

Structure:
  {
    "uploads": [
      {
        "file_id":     "a1b2c3d4",
        "filename":    "supply_chain.xlsx",
        "uploaded_at": "2026-08-14T10:00:00Z",
        "sheet_count": 3,
        "sheets": [
          {
            "sheet_name":          "supply_chain_shipments",
            "original_sheet_name": "Supply Chain Shipments",
            "col_fingerprint":     "3f8a...",   ← sorted-normalised column hash
            "columns":             [...],        ← original column names
            "normalised_columns":  [...],        ← snake_case column names
            "vertical":            "supply_chain",
            "atom_canonical_id":   "supply_chain_shipments_wms_record",
            "confidence":          0.83,
            "row_count":           40
          }
        ]
      }
    ],
    "column_cache": {
      "3f8a...": {
        "vertical":          "supply_chain",
        "atom_canonical_id": "supply_chain_shipments_wms_record",
        "confidence":        0.83,
        "first_seen":        "2026-08-14T10:00:00Z",
        "hit_count":         3
      }
    }
  }
"""

import hashlib
import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

_write_locks: Dict[str, threading.Lock] = {}
_lock_mutex  = threading.Lock()


def _get_lock(reg_path: Path) -> threading.Lock:
    key = str(reg_path.resolve())
    with _lock_mutex:
        if key not in _write_locks:
            _write_locks[key] = threading.Lock()
        return _write_locks[key]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _registry_path(customer_data_dir: Path) -> Path:
    return customer_data_dir / "upload_registry.json"


def _load(reg_path: Path) -> Dict[str, Any]:
    if reg_path.exists():
        try:
            return json.loads(reg_path.read_text(encoding="utf-8"))
        except Exception:
            log.warning(f"[upload_registry] Could not parse {reg_path} — starting fresh")
    return {"uploads": [], "column_cache": {}}


def _save(reg_path: Path, data: Dict[str, Any]) -> None:
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    reg_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Column fingerprint ─────────────────────────────────────────────────────────

def column_fingerprint(normalised_columns: List[str]) -> str:
    """
    Stable 12-char hash from sorted normalised column names.
    Same columns in any order → same fingerprint.
    Used as the key in column_cache.
    """
    sig = "|".join(sorted(c.strip().lower() for c in normalised_columns if c))
    return hashlib.md5(sig.encode()).hexdigest()[:12]


def content_hash(data: bytes, length: int = 16) -> str:
    """
    Short hash of raw file/sheet bytes.
    Used alongside col_fingerprint in duplicate detection so that
    same-columns-different-rows files are NOT treated as duplicates.
    """
    import hashlib
    return hashlib.sha256(data).hexdigest()[:length]


def normalise_col(name: str) -> str:
    """Mirror of sot_ingestion._clean_col() — keep in sync."""
    import re
    name = str(name).lower().strip()
    name = re.sub(r"[\s\-/\\]+", "_", name)
    name = re.sub(r"[^a-z0-9_]", "", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "col"


# ── Public API ─────────────────────────────────────────────────────────────────

def check_duplicate(
    customer_data_dir:      Path,
    filename:               str,
    sheet_name:             str,
    normalised_columns:     List[str],
    content_hash:           str = "",
) -> Dict[str, Any]:
    """
    Check if this exact sheet (same filename + columns + file content) was
    uploaded before.

    A re-upload of the same file with modified rows has the same column
    fingerprint but a different content_hash — NOT treated as a duplicate so
    upsert_rows() can compute the row-level diff and write data_history.json.

    content_hash: short hash of the raw file/sheet bytes. When provided,
                  BOTH col_fingerprint AND content_hash must match for a
                  duplicate hit. When omitted (empty string), falls back to
                  column-only matching (legacy behaviour).

    Returns:
        {
            "is_duplicate": bool,
            "file_id":      str | None,
            "uploaded_at":  str | None,
        }
    """
    reg_path = _registry_path(customer_data_dir)
    reg      = _load(reg_path)
    uploads  = reg.get("uploads", [])

    if not uploads:
        return {"is_duplicate": False, "file_id": None, "uploaded_at": None}

    fp = column_fingerprint(normalised_columns)

    for record in uploads:
        if record.get("filename") != filename:
            continue
        for sheet in record.get("sheets", []):
            if sheet.get("sheet_name") != sheet_name:
                continue
            if sheet.get("col_fingerprint") != fp:
                continue
            # Column structure matches — now check content hash
            # If content_hash provided: must also match (same rows = true dup)
            # If stored entry has no content_hash (legacy): treat as modified to allow re-upload
            stored_hash = sheet.get("content_hash", "")
            if content_hash and content_hash != stored_hash:
                # Either different content, or legacy entry with no stored hash —
                # either way allow through so row diff can be computed
                log.info(
                    f"[upload_registry] MODIFIED re-upload: '{filename}' / '{sheet_name}' "
                    f"same columns but different content — processing for row diff"
                )
                return {"is_duplicate": False, "file_id": None, "uploaded_at": None}
            log.info(
                f"[upload_registry] DUPLICATE: '{filename}' / '{sheet_name}' "
                f"matches file_id={record['file_id']} uploaded at {record['uploaded_at']}"
            )
            return {
                "is_duplicate": True,
                "file_id":      record["file_id"],
                "uploaded_at":  record["uploaded_at"],
            }

    return {"is_duplicate": False, "file_id": None, "uploaded_at": None}


def lookup_column_cache(
    customer_data_dir:  Path,
    normalised_columns: List[str],
) -> Optional[Dict[str, Any]]:
    """
    Check column cache for a matching normalised column fingerprint.

    Returns cached detection dict if found:
        {
            "vertical":          str,
            "atom_canonical_id": str | None,
            "confidence":        float,
        }
    Or None if no cache hit.
    """
    reg_path = _registry_path(customer_data_dir)
    reg      = _load(reg_path)
    cache    = reg.get("column_cache", {})

    fp = column_fingerprint(normalised_columns)

    if fp in cache:
        entry = cache[fp]
        log.info(
            f"[upload_registry] Column cache HIT (fp={fp}) → "
            f"vertical={entry['vertical']} atom={entry.get('atom_canonical_id')} "
            f"hits={entry.get('hit_count', 0)+1}"
        )
        # Increment hit counter (best-effort, no lock needed for a counter)
        try:
            reg_w = _load(reg_path)
            if fp in reg_w.get("column_cache", {}):
                reg_w["column_cache"][fp]["hit_count"] = entry.get("hit_count", 0) + 1
                _save(reg_path, reg_w)
        except Exception:
            pass

        return {
            "vertical":          entry["vertical"],
            "atom_canonical_id": entry.get("atom_canonical_id"),
            "confidence":        entry.get("confidence", 0.0),
            "canonical_map":     entry.get("canonical_map", {}),
            "grain_keys":        entry.get("grain_keys", []),
            "method":            "column_cache",
            "low_confidence":    entry.get("confidence", 0.0) < 0.60,
            "scores":            {},
            "atom_overlap":      entry.get("confidence", 0.0),
        }

    log.info(f"[upload_registry] Column cache MISS (fp={fp})")
    return None


def record_upload(
    customer_data_dir: Path,
    filename:          str,
    sheets:            List[Dict[str, Any]],
) -> str:
    """
    Record a completed upload in the registry and update the column cache.

    Args:
        customer_data_dir: get_customer_paths(cid)["DATA_DIR"]
        filename:          original filename
        sheets:            list of per-sheet dicts, each with:
                             sheet_name, original_sheet_name, columns,
                             normalised_columns, vertical, atom_canonical_id,
                             confidence, row_count

    Returns:
        file_id (8-char string)
    """
    reg_path = _registry_path(customer_data_dir)

    with _get_lock(reg_path):
        reg      = _load(reg_path)
        file_id  = str(uuid.uuid4())[:8]
        ts       = _now()

        # Build sheet records + update column cache
        sheet_records = []
        for s in sheets:
            norm_cols = s.get("normalised_columns") or [
                normalise_col(c) for c in s.get("columns", [])
            ]
            fp = column_fingerprint(norm_cols)

            sheet_records.append({
                "sheet_name":          s["sheet_name"],
                "original_sheet_name": s.get("original_sheet_name", s["sheet_name"]),
                "col_fingerprint":     fp,
                "content_hash":        s.get("content_hash", ""),
                "columns":             s.get("columns", []),
                "normalised_columns":  norm_cols,
                "vertical":            s.get("vertical", "unknown"),
                "atom_canonical_id":   s.get("atom_canonical_id"),
                "confidence":          s.get("confidence", 0.0),
                "row_count":           s.get("row_count", 0),
            })

            # Update column cache (overwrite with latest detection)
            # Bug C5 fix: also store canonical_map and grain_keys
            reg.setdefault("column_cache", {})[fp] = {
                "vertical":          s.get("vertical", "unknown"),
                "atom_canonical_id": s.get("atom_canonical_id"),
                "confidence":        s.get("confidence", 0.0),
                "canonical_map":     s.get("canonical_map", {}),
                "grain_keys":        s.get("grain_keys", []),
                "content_hash":      s.get("content_hash", ""),
                "first_seen":        reg.get("column_cache", {}).get(fp, {}).get("first_seen", ts),
                "hit_count":         reg.get("column_cache", {}).get(fp, {}).get("hit_count", 0),
            }

        reg.setdefault("uploads", []).append({
            "file_id":     file_id,
            "filename":    filename,
            "uploaded_at": ts,
            "sheet_count": len(sheet_records),
            "sheets":      sheet_records,
        })

        _save(reg_path, reg)

    log.info(
        f"[upload_registry] Recorded upload '{filename}' "
        f"file_id={file_id} sheets={len(sheet_records)}"
    )
    return file_id


def get_all_uploads(customer_data_dir: Path) -> List[Dict[str, Any]]:
    """Return all upload records, newest first."""
    reg = _load(_registry_path(customer_data_dir))
    uploads = reg.get("uploads", [])
    return sorted(uploads, key=lambda x: x.get("uploaded_at", ""), reverse=True)


def get_column_cache(customer_data_dir: Path) -> Dict[str, Any]:
    """Return the full column cache for inspection."""
    return _load(_registry_path(customer_data_dir)).get("column_cache", {})
