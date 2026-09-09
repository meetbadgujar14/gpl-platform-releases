"""
services/impact.py
====================
Computes which atoms are "affected" by a set of newly-added or changed
tables, by walking the real FK relationship graph outward from them.

Example: tables A, B, C, D already exist with a relationship A<->C. You add
table E, which relates to A. The affected set becomes {E, A, C} — B and D,
having no path to E, are left out.

Used to scope expensive regeneration (mock data, domain goals) to only the
atoms that actually need to move, instead of redoing the entire vertical
on every Generate call.
"""

import logging
from typing import Dict, Set

from services import atom_registry
from services.relationship_registry import list_relationships

log = logging.getLogger(__name__)


def table_names_to_atom_ids(table_names: Set[str], vertical: str) -> Set[str]:
    """
    Map raw table names (e.g. 'supply_chain_orders_erp_record', from a CSV
    filename) to the atom canonical_ids VerticalSchemaAgent reconciled them
    into. Matching is case-insensitive since atom canonical_ids may differ
    from table names only in casing (e.g. '..._ERP_record' vs '..._erp_record').
    """
    atoms = [a for a in atom_registry.get_all_atoms() if a.get("domain") == vertical]
    by_lower = {a["canonical_id"].lower(): a["canonical_id"] for a in atoms if a.get("canonical_id")}

    matched: Set[str] = set()
    for t in table_names:
        cid = by_lower.get(t.lower())
        if cid:
            matched.add(cid)
        else:
            log.warning(f"[impact] Could not map table '{t}' to an atom canonical_id")
    return matched


def compute_affected_atoms(vertical: str, changed_atom_ids: Set[str]) -> Set[str]:
    """
    BFS outward from changed_atom_ids across the FK relationship graph
    (both directions), restricted to atoms in this vertical. Returns the
    full connected component — every atom reachable from a changed atom,
    including the changed atoms themselves.

    If changed_atom_ids is empty, or none of them exist in this vertical's
    atom set, returns an empty set (caller should treat that as "nothing
    to scope, fall back to regenerating everything" for safety).
    """
    all_atoms = atom_registry.get_all_atoms()
    vertical_atom_ids = {a["canonical_id"] for a in all_atoms if a.get("domain") == vertical}

    seed = changed_atom_ids & vertical_atom_ids
    if not seed:
        return set()

    rels = [
        r for r in list_relationships()
        if r.get("from_atom") in vertical_atom_ids and r.get("to_atom") in vertical_atom_ids
    ]

    adjacency: Dict[str, Set[str]] = {}
    for r in rels:
        a, b = r.get("from_atom"), r.get("to_atom")
        if not a or not b:
            continue
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    affected: Set[str] = set()
    frontier = list(seed)
    while frontier:
        cid = frontier.pop()
        if cid in affected:
            continue
        affected.add(cid)
        for neighbor in adjacency.get(cid, ()):
            if neighbor not in affected:
                frontier.append(neighbor)

    log.info(f"[impact] vertical={vertical} changed={sorted(seed)} → "
             f"affected={sorted(affected)}")
    return affected
