"""
customer/context_detector.py
=============================
Detects which business context a newly-uploaded table belongs to.

For each upload, this module:
1. Reads the existing contexts.json for the customer
2. Analyses the new table's schema + sample values
3. Uses an AI call to decide: join existing context OR create new context
4. Updates contexts.json accordingly

contexts.json structure (per customer, lives in DATA_DIR):
{
  "contexts": {
    "logistics": {
      "id": "logistics",
      "name": "Logistics Operations",
      "tables": ["shipments", "carriers", "delivery_events"],
      "detected_at": "2026-01-01T00:00:00Z"
    },
    "hr": {
      "id": "hr",
      "name": "HR Analytics",
      "tables": ["employees"],
      "detected_at": "2026-01-02T00:00:00Z"
    }
  },
  "default_context": "logistics"
}
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_contexts(data_dir: Path) -> Dict[str, Any]:
    path = data_dir / "contexts.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"contexts": {}, "default_context": None}


def _save_contexts(data_dir: Path, ctx: Dict[str, Any]) -> None:
    path = data_dir / "contexts.json"
    path.write_text(json.dumps(ctx, indent=2), encoding="utf-8")


def _safe_json(text: str) -> Optional[Dict]:
    """Extract and parse the first JSON object from a string."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except Exception:
        return None


# ── Core logic ────────────────────────────────────────────────────────────────

def detect_context(
    data_dir: Path,
    table_name: str,
    columns: List[Dict[str, str]],
    sample_values: Optional[Dict[str, List[str]]] = None,
    vertical: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Determine which business context the table belongs to.

    Resolution order:
    1. Table name prefix (LAW — always wins, no scoring/AI can override)
    2. Vertical hint fast-path (if prefix unknown but vertical hint matches existing context)
    3. Column overlap scoring (>= 2 points)
    4. AI classification
    5. Fallback to vertical hint or slug
    """
    ctx_store = _load_contexts(data_dir)
    existing  = ctx_store.get("contexts", {})
    col_names = [c["name"] for c in columns]

    # -- PREFIX IS LAW: resolve context from table name prefix first.
    # Ensures logistics_* tables are never mis-assigned to supply_chain
    # just because supply_chain has a larger column cache score.
    _PREFIX_MAP = [
        ("supply_chain_",    "supply_chain",  "Supply Chain"),
        ("logistics_",       "logistics",     "Logistics Operations"),
        ("retail_shopify_",  "retail_shopify","Retail / Shopify"),
        ("retail_",          "retail_shopify","Retail / Shopify"),
        ("hr_",              "hr",            "HR Analytics"),
        ("human_resources_", "hr",            "HR Analytics"),
        ("finance_",         "finance",       "Finance Analytics"),
        ("sales_",           "sales",         "Sales Analytics"),
    ]
    tk = table_name.lower()
    for prefix, ctx_id, ctx_name in _PREFIX_MAP:
        if tk.startswith(prefix):
            if ctx_id in existing:
                _add_table_to_context(ctx_store, ctx_id, table_name, data_dir)
                return {
                    "context_id":   ctx_id,
                    "context_name": existing[ctx_id]["name"],
                    "action":       "joined",
                }
            else:
                _create_context(ctx_store, ctx_id, ctx_name, table_name, data_dir)
                return {
                    "context_id":   ctx_id,
                    "context_name": ctx_name,
                    "action":       "created",
                }

    # -- Unknown prefix: fall through to scoring + AI for genuinely novel tables.

    # Fast path: vertical hint matches existing context id exactly
    if vertical and vertical in existing:
        _add_table_to_context(ctx_store, vertical, table_name, data_dir)
        return {
            "context_id":   vertical,
            "context_name": existing[vertical]["name"],
            "action":       "joined",
        }

    # Column overlap scoring
    best_match, best_score = _score_existing_contexts(existing, col_names, sample_values or {})
    if best_score >= 2:
        _add_table_to_context(ctx_store, best_match, table_name, data_dir)
        return {
            "context_id":   best_match,
            "context_name": existing[best_match]["name"],
            "action":       "joined",
        }

    # AI classification
    result = _ai_classify_context(existing, table_name, col_names, sample_values or {}, vertical)

    if result is None:
        context_id = vertical or _slugify(table_name)
        if context_id in existing:
            _add_table_to_context(ctx_store, context_id, table_name, data_dir)
            return {"context_id": context_id, "context_name": existing[context_id]["name"], "action": "joined"}
        else:
            context_name = _guess_name(context_id)
            _create_context(ctx_store, context_id, context_name, table_name, data_dir)
            return {"context_id": context_id, "context_name": context_name, "action": "created"}

    action     = result.get("action", "create")
    context_id = result.get("context_id", "").strip().lower().replace(" ", "_")

    if action == "join" and context_id in existing:
        _add_table_to_context(ctx_store, context_id, table_name, data_dir)
        return {
            "context_id":   context_id,
            "context_name": existing[context_id]["name"],
            "action":       "joined",
        }
    else:
        # Create new context
        context_name = result.get("context_name", _guess_name(context_id or table_name))
        if not context_id:
            context_id = _slugify(context_name or table_name)
        _create_context(ctx_store, context_id, context_name, table_name, data_dir)
        return {
            "context_id":   context_id,
            "context_name": context_name,
            "action":       "created",
        }


def _score_existing_contexts(
    existing: Dict[str, Any],
    col_names: List[str],
    sample_values: Dict[str, List[str]],
) -> tuple:
    """
    Score how well each existing context matches the new table.
    Returns (best_context_id, score).
    Score = shared column keywords (2pts each) + value overlap (1pt each).
    """
    best_id, best_score = None, 0

    for ctx_id, ctx in existing.items():
        score = 0
        ctx_cols = set(ctx.get("column_names", []))

        for col in col_names:
            col_l = col.lower()
            # Check shared column names
            for cc in ctx_cols:
                if col_l == cc.lower() or (len(col_l) > 4 and col_l in cc.lower()):
                    score += 2
                    break

        # Value overlap: check if any sample values appear in stored enum samples
        ctx_sample_vals = set(v.lower() for v in ctx.get("sample_values", []))
        for vals in sample_values.values():
            for v in (vals or []):
                if str(v).lower() in ctx_sample_vals:
                    score += 1

        if score > best_score:
            best_id, best_score = ctx_id, score

    return best_id, best_score


def _ai_classify_context(
    existing: Dict[str, Any],
    table_name: str,
    col_names: List[str],
    sample_values: Dict[str, List[str]],
    vertical: Optional[str],
) -> Optional[Dict]:
    """Call LLM to classify the table into an existing or new context."""
    try:
        from anthropic import Anthropic
        from core.config import settings
        client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    except Exception as e:
        log.warning(f"[context_detector] Cannot create Anthropic client: {e}")
        return None

    # Summarise existing contexts for the prompt
    existing_summary = []
    for ctx_id, ctx in existing.items():
        existing_summary.append(
            f"- context_id={ctx_id!r}, name={ctx['name']!r}, "
            f"tables={ctx['tables']}"
        )
    existing_text = "\n".join(existing_summary) if existing_summary else "None yet."

    # Safe sample preview (no raw customer values sent — only col names and value types)
    sample_preview = {}
    for col, vals in list(sample_values.items())[:8]:
        sample_preview[col] = [str(v)[:30] for v in (vals or [])[:3]]

    prompt = f"""You are a business data context classifier.

A customer has uploaded a new dataset. Decide whether it belongs to an existing 
business context or represents a new, distinct business domain.

NEW TABLE
=========
Table name  : {table_name}
Columns     : {col_names}
Sample values (first 3 per column): {json.dumps(sample_preview)}
Vertical hint: {vertical or 'unknown'}

EXISTING CONTEXTS
=================
{existing_text}

INSTRUCTIONS
============
Return ONLY a JSON object, no explanation, no markdown:

If this table clearly belongs to an existing context (shared keys, related domain):
{{"action": "join", "context_id": "<existing_context_id>", "reason": "<brief>"}}

If this table represents a NEW business domain:
{{"action": "create", "context_id": "<new_snake_case_id>", "context_name": "<Human Readable Name>", "reason": "<brief>"}}

Rules:
- "join" only when the domain is clearly related (logistics + carriers = join, logistics + employees = create)
- context_id must be short snake_case (e.g. "logistics", "hr", "finance", "supply_chain")
- context_name should be human-readable (e.g. "Logistics Operations", "HR Analytics")
- When in doubt, create a new context
"""

    try:
        resp = client.messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text if resp.content else ""
        result = _safe_json(text)
        log.info(f"[context_detector] AI result for '{table_name}': {result}")
        return result
    except Exception as e:
        log.warning(f"[context_detector] AI call failed: {e}")
        return None


def _add_table_to_context(
    ctx_store: Dict[str, Any],
    context_id: str,
    table_name: str,
    data_dir: Path,
) -> None:
    ctx = ctx_store["contexts"][context_id]
    if table_name not in ctx["tables"]:
        ctx["tables"].append(table_name)
    ctx["updated_at"] = _now()
    _save_contexts(data_dir, ctx_store)


def _create_context(
    ctx_store: Dict[str, Any],
    context_id: str,
    context_name: str,
    table_name: str,
    data_dir: Path,
) -> None:
    ctx_store["contexts"][context_id] = {
        "id":           context_id,
        "name":         context_name,
        "tables":       [table_name],
        "column_names": [],
        "sample_values": [],
        "detected_at":  _now(),
        "updated_at":   _now(),
    }
    # Set default context if this is the first
    if not ctx_store.get("default_context"):
        ctx_store["default_context"] = context_id
    _save_contexts(data_dir, ctx_store)
    # Create the context directory under customers/{cid}/contexts/{context_id}/
    ctx_dir = data_dir.parent / "context" / context_id
    ctx_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"[context_detector] Created context '{context_id}' at {ctx_dir}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "context"


def _guess_name(context_id: str) -> str:
    """Convert snake_case id to a human-readable name."""
    words = context_id.replace("_", " ").title().split()
    suffix_map = {
        "Logistics": "Logistics Operations",
        "Hr": "HR Analytics",
        "Finance": "Finance Analytics",
        "Supply": "Supply Chain",
        "Inventory": "Inventory Management",
        "Sales": "Sales Analytics",
        "Marketing": "Marketing Analytics",
    }
    for word in words:
        if word in suffix_map:
            return suffix_map[word]
    return " ".join(words) + " Analytics"


def update_context_schema_cache(
    data_dir: Path,
    context_id: str,
    col_names: List[str],
    sample_values: Dict[str, List[str]],
) -> None:
    """
    Keep a rolling cache of column names and sample values per context.
    Used by _score_existing_contexts for fast (no-AI) matching on future uploads.
    """
    ctx_store = _load_contexts(data_dir)
    ctx = ctx_store.get("contexts", {}).get(context_id)
    if not ctx:
        return

    existing_cols = set(ctx.get("column_names", []))
    existing_cols.update(col_names)
    ctx["column_names"] = list(existing_cols)[:200]  # cap

    existing_samples = set(ctx.get("sample_values", []))
    for vals in sample_values.values():
        for v in (vals or [])[:5]:
            existing_samples.add(str(v)[:50])
    ctx["sample_values"] = list(existing_samples)[:500]  # cap

    _save_contexts(data_dir, ctx_store)
