"""
agents/relationship_declaration_agent.py
=========================================
RelationshipDeclarationAgent — Claude managed agent that declares FK
relationships between atoms that already exist for a vertical.

This is a deliberate split from VerticalSchemaAgent: schema creation and
relationship declaration are two separate concerns, run as two separate
cards/steps. VerticalSchemaAgent only gathers atoms — it no longer touches
relationships.

This agent does NOT create or modify atoms. It only reads existing atom
definitions for a vertical and declares FK relationships between them using
domain knowledge (e.g. "orders.customer_id -> customers.customer_id").

This is distinct from the (separate, deterministic) Relationship Discovery
Phase 1 agent, which runs zero-AI-call algorithms against actual data
(column matching, functional dependency, hidden dimensions, grain
verification, temporal patterns) to validate declared relationships and
surface ones this agent may have missed. The intended order is:

  1. VerticalSchemaAgent           — gather atoms only
  2. RelationshipDeclarationAgent  — AI declares FK relationships (this file)
  3. Relationship Discovery Phase 1 — algorithmic validation / gap-filling
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from anthropic import Anthropic

from core.config import settings
from services import atom_registry
from services.relationship_registry import add_relationship, list_relationships


def _get_client():
    """Build a fresh client per call so a runtime API-key update takes effect immediately."""
    return Anthropic(api_key=settings.ANTHROPIC_API_KEY)


def _get_model():
    return settings.ANTHROPIC_MODEL


_SYSTEM = """
You are the RelationshipDeclarationAgent for the BizzBrain GPL system.

Your ONLY job is to declare FK relationships between atoms that already
exist for a vertical. You do not create, edit, or validate atoms — they
already exist. You just read them and connect them.

FK RELATIONSHIPS:
  from_atom/from_field = atom that HAS the FK column (the child / fact table)
  to_atom/to_field      = atom that OWNS the PK (the parent / dimension table)

  e.g. from_atom="supply_chain_orders_ERP_record", from_field="customer_id",
       to_atom="supply_chain_customers_CRM_record", to_field="customer_id"

RULES:
  - Only declare a relationship if both atoms exist (get_existing_atoms first).
  - Only declare a relationship if the from_field is plausibly a foreign key
    pointing at the to_field (matching or clearly-related grain key/primary key).
  - Do not declare duplicate relationships — call get_relationships first and
    skip any pair already declared.
  - Do not guess wildly. If you are not reasonably confident two atoms are
    related, skip the pair rather than declaring a low-confidence relationship.
  - Go through the full atom list systematically — for each fact/event/state
    atom, look for FK columns that point at dimension/reference atoms, and
    also consider fact-to-fact links (e.g. invoice_lines -> orders).

Do not stop until you have gone through every atom in the provided list.
"""

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "get_existing_atoms",
        "description": "Read existing atoms for the vertical (or all domains if empty).",
        "input_schema": {
            "type": "object",
            "properties": {"domain": {"type": "string", "description": "Filter by domain. Empty = all."}},
            "required": ["domain"]
        }
    },
    {
        "name": "get_relationships",
        "description": "Read all existing FK relationships already declared.",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "write_relationship",
        "description": "Declare FK relationship between two existing atoms. Both must already exist.",
        "input_schema": {
            "type": "object",
            "properties": {
                "from_atom":  {"type": "string"},
                "from_field": {"type": "string"},
                "to_atom":    {"type": "string"},
                "to_field":   {"type": "string"}
            },
            "required": ["from_atom", "from_field", "to_atom", "to_field"]
        }
    }
]


def _tool_get_existing_atoms(domain: str) -> Dict:
    result = []
    for cid in atom_registry.list_atoms():
        a = atom_registry.get_atom(cid)
        if a and (not domain or a.get("domain") == domain):
            result.append({
                "canonical_id":   a.get("canonical_id"),
                "domain":         a.get("domain"),
                "record_type":    a.get("record_type"),
                "grain_keys":     a.get("grain_keys", []),
                "fields":         [f["name"] for f in a.get("fields", [])],
            })
    return {"atoms": result, "count": len(result)}


def _tool_get_relationships() -> Dict:
    rels = list_relationships()
    return {"relationships": rels, "count": len(rels)}


def _tool_write_relationship(from_atom: str, from_field: str, to_atom: str, to_field: str) -> Dict:
    if not atom_registry.get_atom(from_atom):
        return {"status": "error", "error": f"from_atom '{from_atom}' does not exist"}
    if not atom_registry.get_atom(to_atom):
        return {"status": "error", "error": f"to_atom '{to_atom}' does not exist"}
    try:
        return add_relationship(from_atom, from_field, to_atom, to_field)
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _dispatch(name: str, inp: Dict) -> str:
    try:
        if   name == "get_existing_atoms":  r = _tool_get_existing_atoms(inp.get("domain", ""))
        elif name == "get_relationships":   r = _tool_get_relationships()
        elif name == "write_relationship":  r = _tool_write_relationship(inp["from_atom"], inp["from_field"], inp["to_atom"], inp["to_field"])
        else:                               r = {"error": f"Unknown tool: {name}"}
    except Exception as e:
        r = {"error": f"Tool error: {e}"}
    return json.dumps(r)


def run(vertical: str, max_iterations: int = 200) -> Dict:
    """
    Run the RelationshipDeclarationAgent for a vertical.

    Reads all atoms already gathered for `vertical` and declares FK
    relationships between them. Does not create or modify atoms.

    Returns:
        {status, vertical, atoms_considered, relationships_created, iterations, summary}
    """
    existing = _tool_get_existing_atoms(vertical)
    atom_ids = [a["canonical_id"] for a in existing["atoms"]]

    if not atom_ids:
        return {
            "status": "no_changes",
            "vertical": vertical,
            "atoms_considered": 0,
            "relationships_created": 0,
            "iterations": 0,
            "summary": f"No atoms exist yet for '{vertical}'. Run the Vertical Schema Agent first.",
        }

    atom_list_str = "\n".join(f"  - {cid}" for cid in atom_ids)
    task = f"""
Declare FK relationships between the atoms that already exist for the
'{vertical}' vertical.

ATOMS TO CONSIDER ({len(atom_ids)} total):
{atom_list_str}

STEPS:
1. Call get_existing_atoms(domain='{vertical}') to see full field lists.
2. Call get_relationships() to see what's already declared — skip duplicates.
3. Go through the atom list and declare every plausible FK relationship
   you find using write_relationship.

Do not declare done until every atom on the list has been considered.
"""

    messages = [{"role": "user", "content": task}]
    rels_created = 0
    iterations   = 0
    summary      = ""

    while iterations < max_iterations:
        iterations += 1

        response = _get_client().messages.create(
            model=_get_model(),
            max_tokens=4096,
            system=_SYSTEM,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    summary = block.text
            break

        if response.stop_reason == "tool_use":
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result_str = _dispatch(block.name, block.input)
                result_obj = json.loads(result_str)
                if block.name == "write_relationship":
                    if result_obj.get("status") == "created":
                        rels_created += 1
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": result_str})
            messages.append({"role": "user", "content": results})
        else:
            break

    status = "success" if rels_created else "no_changes"
    return {
        "status":                 status,
        "vertical":               vertical,
        "atoms_considered":       len(atom_ids),
        "relationships_created":  rels_created,
        "iterations":             iterations,
        "summary":                summary,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run RelationshipDeclarationAgent")
    parser.add_argument("vertical")
    args = parser.parse_args()

    result = run(vertical=args.vertical)
    print(json.dumps(result, indent=2))
