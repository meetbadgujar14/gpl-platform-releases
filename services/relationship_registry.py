"""
relationship_registry.py
========================
Stores and retrieves explicit FK relationships between atoms.

Relationships are saved to data/atom_relationships.json.

Each relationship records:
  - from_atom:        canonical_id of the fact/child table (has the FK column)
  - from_field:       the FK column name in from_atom
  - to_atom:          canonical_id of the dimension/parent table (owns the PK)
  - to_field:         the PK column name in to_atom
  - created_at:       ISO timestamp

When a relationship is saved, both atom definitions in atoms.json are
updated so the compiler can read join paths directly from the atom schema:
  - from_atom field gets:  fk_target_atom, fk_target_field
  - to_atom field gets:    referenced_by list updated
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

import logging; get_logger = logging.getLogger
from core.paths import ATOMS_PATH

log = get_logger(__name__)

_REL_PATH = Path(__file__).parent.parent / "data" / "atom_relationships.json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


# ── Load / Save relationships file ────────────────────────────────────────────

def _load_relationships() -> List[Dict]:
    if not _REL_PATH.exists():
        return []
    try:
        text = _REL_PATH.read_text(encoding="utf-8").strip()
        if not text:
            return []
        return json.loads(text)
    except Exception as e:
        log.error(f"relationship_registry: failed to load: {e}")
        return []


def _save_relationships(rels: List[Dict]) -> None:
    _REL_PATH.write_text(json.dumps(rels, indent=2), encoding="utf-8")


# ── Load / Save atoms.json ─────────────────────────────────────────────────────

def _load_atoms_db() -> Dict:
    if not ATOMS_PATH.exists():
        return {"_default": {}}
    try:
        text = ATOMS_PATH.read_text(encoding="utf-8").strip()
        if not text:
            return {"_default": {}}
        return json.loads(text)
    except Exception:
        return {"_default": {}}


def _save_atoms_db(db: Dict) -> None:
    ATOMS_PATH.write_text(json.dumps(db, indent=2), encoding="utf-8")


def _get_atom(db: Dict, canonical_id: str) -> Optional[Dict]:
    default = db.get("_default", {})
    for v in default.values():
        if isinstance(v, dict) and v.get("canonical_id") == canonical_id:
            return v
    return None


def _find_atom_key(db: Dict, canonical_id: str) -> Optional[str]:
    default = db.get("_default", {})
    for k, v in default.items():
        if isinstance(v, dict) and v.get("canonical_id") == canonical_id:
            return k
    return None


# ── Update atom fields with FK info ───────────────────────────────────────────

def _patch_atom_fields(db: Dict, canonical_id: str,
                        field_name: str, patch: Dict) -> bool:
    """Apply patch dict to a specific field in an atom. Returns True if saved."""
    key = _find_atom_key(db, canonical_id)
    if key is None:
        log.warning(f"relationship_registry: atom '{canonical_id}' not found in atoms.json")
        return False

    atom = db["_default"][key]
    fields = atom.get("fields", [])
    found = False
    for f in fields:
        if f.get("name") == field_name:
            f.update(patch)
            found = True
            break

    if not found:
        log.warning(f"relationship_registry: field '{field_name}' not found in atom '{canonical_id}'")
        return False

    atom["updated_at"] = _now()
    db["_default"][key] = atom
    return True


# ── Public API ─────────────────────────────────────────────────────────────────

def list_relationships() -> List[Dict]:
    """Return all declared relationships."""
    return _load_relationships()


def add_relationship(
    from_atom: str,
    from_field: str,
    to_atom: str,
    to_field: str,
    use_for_goals: bool = True,
) -> Dict[str, Any]:
    """
    Save a new FK relationship and update both atom definitions.

    Returns:
        {"status": "created"|"duplicate", "relationship": {...}}
    """
    rels = _load_relationships()

    # Check duplicate
    for r in rels:
        if (r["from_atom"] == from_atom and r["from_field"] == from_field
                and r["to_atom"] == to_atom and r["to_field"] == to_field):
            return {"status": "duplicate", "relationship": r}

    rel = {
        "from_atom":     from_atom,
        "from_field":    from_field,
        "to_atom":       to_atom,
        "to_field":      to_field,
        "use_for_goals": use_for_goals,
        "created_at":    _now(),
    }
    rels.append(rel)
    _save_relationships(rels)

    # ── Update atoms.json ──────────────────────────────────────────────────────
    db = _load_atoms_db()

    # 1. from_atom field: add FK pointer
    _patch_atom_fields(db, from_atom, from_field, {
        "fk_target_atom":  to_atom,
        "fk_target_field": to_field,
    })

    # 2. to_atom field: add referenced_by
    key = _find_atom_key(db, to_atom)
    if key is not None:
        atom = db["_default"][key]
        fields = atom.get("fields", [])
        for f in fields:
            if f.get("name") == to_field:
                refs = f.get("referenced_by", [])
                if from_atom not in refs:
                    refs.append(from_atom)
                f["referenced_by"] = refs
                break
        atom["updated_at"] = _now()
        db["_default"][key] = atom

    _save_atoms_db(db)

    log.info(f"relationship_registry: created {from_atom}.{from_field} → {to_atom}.{to_field}")
    return {"status": "created", "relationship": rel}


def delete_relationship(from_atom: str, from_field: str,
                         to_atom: str, to_field: str) -> bool:
    """
    Remove a relationship and clean up atom FK annotations.
    Returns True if found and deleted.
    """
    rels = _load_relationships()
    new_rels = [r for r in rels if not (
        r["from_atom"] == from_atom and r["from_field"] == from_field
        and r["to_atom"] == to_atom and r["to_field"] == to_field
    )]

    if len(new_rels) == len(rels):
        return False  # not found

    _save_relationships(new_rels)

    # ── Clean up atoms.json ────────────────────────────────────────────────────
    db = _load_atoms_db()

    # Remove FK from from_atom field (only if no other rel uses same from_field)
    still_used = any(
        r["from_atom"] == from_atom and r["from_field"] == from_field
        for r in new_rels
    )
    if not still_used:
        key = _find_atom_key(db, from_atom)
        if key is not None:
            for f in db["_default"][key].get("fields", []):
                if f.get("name") == from_field:
                    f.pop("fk_target_atom", None)
                    f.pop("fk_target_field", None)
            db["_default"][key]["updated_at"] = _now()

    # Remove from referenced_by on to_atom field
    key = _find_atom_key(db, to_atom)
    if key is not None:
        for f in db["_default"][key].get("fields", []):
            if f.get("name") == to_field:
                refs = f.get("referenced_by", [])
                if from_atom in refs:
                    refs.remove(from_atom)
                f["referenced_by"] = refs
        db["_default"][key]["updated_at"] = _now()

    _save_atoms_db(db)

    log.info(f"relationship_registry: deleted {from_atom}.{from_field} → {to_atom}.{to_field}")
    return True


def get_relationships_for_atom(canonical_id: str) -> List[Dict]:
    """Return all relationships where this atom is on either side."""
    rels = _load_relationships()
    return [r for r in rels
            if r["from_atom"] == canonical_id or r["to_atom"] == canonical_id]
