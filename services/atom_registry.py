"""
atom_registry.py
================
Central registry for atom definitions - ONE atom per canonical_id.

Structure in atoms.json:
{
  "_default": {
    "finance_invoice_system_record": {
      "canonical_id": "finance_invoice_system_record",
      "name": "Sales Invoice",
      "domain": "finance",
      "record_type": "record",
      "system": "SYSTEM",
      "grain_keys": ["invoice_number"],
      "grain_description": "one row per invoice",
      "description": "...",
      "created_at": "2024-01-01T10:00:00Z",
      "updated_at": "2024-02-01T15:00:00Z",
      "fields": [
        {
          "name": "invoice_number",
          "type": "string",
          "role": "primary_key",
          "additivity": "non_additive",
          "description": "...",
          "added_at": "2024-01-01T10:00:00Z"
        }
      ]
    }
  }
}

Operations:
- get_atom(canonical_id) \u2192 Returns atom definition or None
- upsert_atom(atom_dict) \u2192 Creates or updates atom, merges new fields
- list_atoms() \u2192 Returns all canonical_ids
"""

from datetime import datetime, timezone
from tinydb import TinyDB
from typing import Dict, List, Any

import logging; get_logger = logging.getLogger
import core.paths as _core_paths

log = get_logger(__name__)

# Module-level path — overridden by _customer_path_context_for_customer
# during customer onboarding. _get_db() reads this variable directly so
# that the path swap takes effect without needing a module reload.
ATOMS_PATH = _core_paths.ATOMS_PATH


def _get_db():
    """
    Get TinyDB instance for atoms.json.

    Reads ATOMS_PATH from this module's namespace (not from the import-time
    binding) so that _customer_path_context_for_customer's path swap is
    respected during customer onboarding pipelines.

    encoding='utf-8' is required — without it, TinyDB opens the file with
    Python's platform-default encoding. On Windows that's cp1252, which
    crashes ('charmap' codec can't decode byte ...) the moment atoms.json
    contains any Unicode character outside that codepage (em dashes, curly
    quotes, accented names — all things Claude-generated text commonly has).
    """
    import services.atom_registry as _self
    return TinyDB(_self.ATOMS_PATH, encoding="utf-8")


def _get_timestamp() -> str:
    """Get ISO timestamp with timezone"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def get_atom(canonical_id: str) -> Dict[str, Any] | None:
    """
    Get atom definition by canonical_id.
    
    Args:
        canonical_id: e.g. "finance_invoice_system_record"
    
    Returns:
        Atom dict or None if not found
    """
    db = _get_db()
    # Query by canonical_id field, not doc_id
    from tinydb import Query
    Atom = Query()
    atom = db.get(Atom.canonical_id == canonical_id)
    db.close()
    return atom


def upsert_atom(atom_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create or update an atom definition.
    
    If atom exists:
        - Merge new fields (add any fields not in existing)
        - Keep existing fields unchanged
        - Update updated_at timestamp
        - Log field changes
    
    If atom doesn't exist:
        - Create new atom with all fields
        - Set created_at and updated_at
    
    Args:
        atom_dict: Complete atom definition with canonical_id and fields
    
    Returns:
        {
            "status": "created" | "updated" | "unchanged",
            "canonical_id": str,
            "fields_added": [str],
            "fields_total": int
        }
    """
    canonical_id = atom_dict.get("canonical_id")
    if not canonical_id:
        raise ValueError("atom_dict must have canonical_id")
    
    db = _get_db()
    from tinydb import Query
    Atom = Query()
    existing_atom = db.get(Atom.canonical_id == canonical_id)
    timestamp = _get_timestamp()
    
    if existing_atom:
        # Atom exists - check for new fields
        existing_fields = {f["name"]: f for f in existing_atom.get("fields", [])}
        new_fields_list = atom_dict.get("fields", [])
        
        fields_added = []
        merged_fields = list(existing_fields.values())  # Start with existing
        
        for new_field in new_fields_list:
            field_name = new_field.get("name")
            if field_name not in existing_fields:
                # New field - add it
                new_field["added_at"] = timestamp
                merged_fields.append(new_field)
                fields_added.append(field_name)
        
        if fields_added:
            # Update atom with merged fields
            existing_atom["fields"] = merged_fields
            existing_atom["updated_at"] = timestamp
            db.update(existing_atom, Atom.canonical_id == canonical_id)
            db.close()
            
            log.info(
                f"Atom '{canonical_id}' UPDATED: "
                f"{len(fields_added)} field(s) added: {fields_added}"
            )
            
            return {
                "status": "updated",
                "canonical_id": canonical_id,
                "fields_added": fields_added,
                "fields_total": len(merged_fields)
            }
        else:
            # No new fields - unchanged
            db.close()
            
            log.info(f"Atom '{canonical_id}' exists with same fields - no changes")
            
            return {
                "status": "unchanged",
                "canonical_id": canonical_id,
                "fields_added": [],
                "fields_total": len(existing_fields)
            }
    
    else:
        # New atom - create it
        atom_dict["created_at"] = timestamp
        atom_dict["updated_at"] = timestamp
        
        # Add added_at to all fields
        for field in atom_dict.get("fields", []):
            field["added_at"] = timestamp
        
        db.insert(atom_dict)
        db.close()
        
        log.info(
            f"Atom '{canonical_id}' CREATED: "
            f"{len(atom_dict.get('fields', []))} field(s)"
        )
        
        return {
            "status": "created",
            "canonical_id": canonical_id,
            "fields_added": [f["name"] for f in atom_dict.get("fields", [])],
            "fields_total": len(atom_dict.get("fields", []))
        }


def list_atoms() -> List[str]:
    """
    Get list of all canonical_ids.
    
    Returns:
        List of canonical_id strings
    """
    db = _get_db()
    all_atoms = db.all()
    db.close()
    
    return [atom.get("canonical_id") for atom in all_atoms if atom.get("canonical_id")]


def get_primary_key_field(canonical_id: str) -> str | None:
    """
    Get the primary key field name for an atom.
    
    Args:
        canonical_id: e.g. "finance_invoice_system_record"
    
    Returns:
        Primary key field name or None
    """
    atom = get_atom(canonical_id)
    if not atom:
        return None
    
    # Check grain_keys first (preferred)
    grain_keys = atom.get("grain_keys", [])
    if grain_keys:
        return grain_keys[0]
    
    # Fallback: find field with role="primary_key"
    for field in atom.get("fields", []):
        if field.get("role") == "primary_key":
            return field.get("name")
    
    return None


def get_atom_fields(canonical_id: str) -> List[str]:
    """
    Get list of field names for an atom.
    
    Args:
        canonical_id: e.g. "finance_invoice_system_record"
    
    Returns:
        List of field names
    """
    atom = get_atom(canonical_id)
    if not atom:
        return []
    
    return [f["name"] for f in atom.get("fields", [])]


def get_all_atoms() -> List[Dict[str, Any]]:
    """Return all atom dicts."""
    db    = _get_db()
    atoms = db.all()
    db.close()
    return atoms
