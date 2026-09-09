"""
services/relationship_validator.py
====================================
Validates atom_relationships.json against atoms.json.

Checks:
  1. Both from_atom and to_atom exist in atoms
  2. from_field exists in from_atom's fields
  3. to_field exists in to_atom's fields
  4. No self-references
  5. No duplicate relationships
  6. _id fields with no FK declared (suspicious missing relationships)

Returns a structured report with errors, warnings, and suggestions.
No AI involved — pure structural validation.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple

import core.paths as _paths


def validate_relationships(vertical: str | None = None) -> Dict:
    """
    Run all relationship validation checks.

    Args:
        vertical: if given, only check atoms/rels for that vertical.
                  If None, checks all.

    Returns dict with keys:
        status       : "ok" | "errors" | "warnings"
        errors       : List[str]   — must fix before mock data
        warnings     : List[str]   — suspicious, review recommended
        suggestions  : List[dict]  — _id fields with no FK (AI can create these)
        stats        : Dict
    """
    # Load atoms
    if not _paths.ATOMS_PATH.exists():
        return _error_response("atoms.json not found — run VerticalSchemaAgent first.")

    raw   = json.loads(_paths.ATOMS_PATH.read_text(encoding="utf-8"))
    atoms = list(raw.get("_default", {}).values())
    if vertical:
        atoms = [a for a in atoms if a.get("domain") == vertical]
    if not atoms:
        return _error_response(f"No atoms found for vertical '{vertical}'.")

    atom_map: Dict[str, Dict] = {a["canonical_id"]: a for a in atoms}
    atom_fields: Dict[str, set] = {
        cid: {f["name"] for f in a.get("fields", [])}
        for cid, a in atom_map.items()
    }
    atom_ids = set(atom_map.keys())

    # Load relationships
    rels_path = _paths.PROJECT_ROOT / "data" / "atom_relationships.json"
    if not rels_path.exists():
        return _error_response("atom_relationships.json not found.")

    rels: List[Dict] = json.loads(rels_path.read_text(encoding="utf-8"))
    if vertical:
        rels = [r for r in rels
                if r.get("from_atom","") in atom_ids or r.get("to_atom","") in atom_ids]

    errors:      List[str] = []
    warnings:    List[str] = []
    suggestions: List[Dict] = []
    seen:        set = set()

    for r in rels:
        fa = r.get("from_atom", "")
        ff = r.get("from_field", "")
        ta = r.get("to_atom", "")
        tf = r.get("to_field", "")

        # 1. Atoms exist
        if fa not in atom_ids:
            errors.append(f"from_atom '{fa}' not found in atoms.")
            continue
        if ta not in atom_ids:
            errors.append(f"to_atom '{ta}' not found in atoms.")
            continue

        # 2. Fields exist
        if ff not in atom_fields[fa]:
            errors.append(
                f"'{_short(fa)}.{ff}' — field '{ff}' does not exist in atom. "
                f"Available: {sorted(atom_fields[fa])}"
            )
        if tf not in atom_fields[ta]:
            errors.append(
                f"'{_short(ta)}.{tf}' — field '{tf}' does not exist in atom. "
                f"Available: {sorted(atom_fields[ta])}"
            )

        # 3. Self-reference
        if fa == ta:
            warnings.append(f"Self-reference on '{_short(fa)}' — likely wrong.")

        # 4. Duplicate
        key = (fa, ff, ta, tf)
        if key in seen:
            warnings.append(
                f"Duplicate relationship: {_short(fa)}.{ff} → {_short(ta)}.{tf}"
            )
        seen.add(key)

    # 5. _id fields with no FK declared (suggestions for AI to create)
    declared_fks: set = {(r["from_atom"], r["from_field"]) for r in rels}
    for a in atoms:
        cid = a["canonical_id"]
        for f in a.get("fields", []):
            fname = f.get("name", "")
            role  = f.get("role", "")
            if (
                fname.endswith("_id")
                and role != "primary_key"
                and (cid, fname) not in declared_fks
            ):
                # Guess the target atom from the field name
                guess = _guess_target(fname, atom_ids, cid)
                suggestions.append({
                    "from_atom":    cid,
                    "from_field":   fname,
                    "guessed_target_atom":  guess,
                    "reason": f"Field '{fname}' looks like a FK but has no relationship declared.",
                })

    status = "ok"
    if errors:
        status = "errors"
    elif warnings:
        status = "warnings"

    return {
        "status":      status,
        "vertical":    vertical or "all",
        "errors":      errors,
        "warnings":    warnings,
        "suggestions": suggestions,
        "stats": {
            "atoms_checked":        len(atoms),
            "relationships_checked": len(rels),
            "errors":               len(errors),
            "warnings":             len(warnings),
            "suggestions":          len(suggestions),
        }
    }


def _short(cid: str) -> str:
    """Strip vertical prefix for readability."""
    parts = cid.split("_")
    return "_".join(parts[2:]) if len(parts) > 2 else cid


def _guess_target(field_name: str, atom_ids: set, exclude: str) -> str | None:
    """Guess which atom a FK field references based on name matching."""
    # e.g. supplier_id → look for atom with 'supplier' in name
    stem = field_name[:-3]  # strip '_id'
    for cid in atom_ids:
        if cid == exclude:
            continue
        if stem in cid or stem.rstrip("s") in cid:
            return cid
    return None


def _error_response(msg: str) -> Dict:
    return {
        "status": "errors",
        "errors": [msg],
        "warnings": [],
        "suggestions": [],
        "stats": {"atoms_checked": 0, "relationships_checked": 0,
                  "errors": 1, "warnings": 0, "suggestions": 0}
    }
