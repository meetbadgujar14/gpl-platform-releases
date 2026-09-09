"""
customer/sot_ingestion.py
==========================
SOT Ingestion — first step of customer onboarding.

What this does:
  1. Reads uploaded CSV files
  2. Normalises column names (lowercase, underscores)
  3. Applies factory canonical column names when an atom match exists
  4. Writes canonical CSVs to customer_runtime/sot_csv/{vertical}/
  5. Extracts schema: table names, column names, types — NO data values
  6. Infers FK relationships from column naming patterns
  7. Returns customer_schema dict ready for VerticalSchemaAgent CUSTOMER mode

Customer data never leaves sot_csv/. Only schema (structure) is passed forward.
"""

import csv
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.paths import CUSTOMER_SOT_DIR, CUSTOMER_FINGERPRINTS_PATH, CUSTOMER_ENUMS_PATH

ENUM_THRESHOLD = 20   # mirrors services/discover_enums.py

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clean_col(name: str) -> str:
    """Normalise a column name to lowercase snake_case."""
    return re.sub(r"[^a-z0-9_]", "_", name.lower().strip()).strip("_")


def _infer_type(samples: List[str]) -> str:
    """Infer column type from sample values."""
    non_empty = [v.strip() for v in samples if v.strip()]
    if not non_empty:
        return "string"

    def is_numeric(v):
        try:
            float(v.replace(",", "").replace("$", "").replace("%", ""))
            return True
        except ValueError:
            return False

    def is_date(v):
        return bool(re.match(r"\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4}", v))

    sample = non_empty[:30]
    if sum(1 for v in sample if is_date(v)) / len(sample) > 0.7:
        return "date"
    if sum(1 for v in sample if is_numeric(v)) / len(sample) > 0.8:
        return "number"
    return "string"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _extract_enums(rows: List[Dict], col_map: Dict[str, str], type_map: Dict[str, str]) -> Dict[str, List[str]]:
    """
    For every string column with ≤ ENUM_THRESHOLD distinct values,
    return the sorted list of unique values (case-preserved).
    Threshold matches discover_enums.py so factory and customer runtime
    use the same definition of 'categorical'.
    col_map: original_col → final_col_name (after canonical mapping)
    """
    enums: Dict[str, List[str]] = {}
    for orig_col, final_col in col_map.items():
        if type_map.get(final_col) != "string":
            continue
        values = sorted({str(r.get(orig_col, "")).strip() for r in rows if r.get(orig_col, "").strip()})
        if 0 < len(values) <= ENUM_THRESHOLD:
            enums[final_col] = values
    return enums


def _update_field_values(
    atom_key:          str,
    enums:             Dict[str, List[str]],
    customer_data_dir: Path,
) -> None:
    """
    Merge newly extracted enums into per-customer field_values.json.

    field_values.json is the single source of truth for enum values —
    used by all factory agents (vertical_schema_agent, seed_agent,
    vocabulary_agent, branch_a/b, wizard_steps).

    Format: { "atom_canonical_id.column_name": ["value1", "value2", ...] }

    customer_enums.json is no longer written — field_values.json is the
    only enum store.
    """
    fv_path = customer_data_dir / "field_values.json"
    fv: Dict = {}
    if fv_path.exists():
        try:
            fv = json.loads(fv_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    for col, values in enums.items():
        fv[f"{atom_key}.{col}"] = values
    fv_path.parent.mkdir(parents=True, exist_ok=True)
    fv_path.write_text(json.dumps(fv, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"[sot_ingestion] field_values.json updated — {len(enums)} enums from '{atom_key}'")


def _guess_primary_key(columns: List[str], atom_canonical_id: Optional[str] = None) -> str:
    """
    Resolve the primary key column for upsert_rows().

    Priority:
      1. Factory grain_keys from atoms.json (most reliable — LLM-validated)
      2. Any column ending with '_id'
      3. Any column named 'id'
      4. First column as fallback
    """
    # ── Tier 1: look up factory grain_keys ────────────────────────────────────
    if atom_canonical_id:
        try:
            from core.paths import ATOMS_PATH
            import json as _json
            raw   = _json.loads(ATOMS_PATH.read_text(encoding="utf-8"))
            atoms = raw.get("_default", raw) if isinstance(raw, dict) else {}
            for atom in atoms.values():
                if atom.get("canonical_id") == atom_canonical_id:
                    grain_keys = atom.get("grain_keys", [])
                    if grain_keys:
                        pk = grain_keys[0]
                        if pk in columns:
                            log.debug(f"[_guess_primary_key] Using factory grain_key '{pk}' for '{atom_canonical_id}'")
                            return pk
                        log.warning(
                            f"[_guess_primary_key] Factory grain_key '{pk}' not found in columns "
                            f"for '{atom_canonical_id}' — falling back to heuristics"
                        )
        except Exception as exc:
            log.debug(f"[_guess_primary_key] Could not read atoms.json: {exc}")

    # ── Tier 2: heuristics ────────────────────────────────────────────────────
    for col in columns:
        if col.endswith("_id"):
            return col
    if "id" in columns:
        return "id"
    return columns[0] if columns else "id"


def build_canonical_map(
    original_cols:     List[str],
    atom_canonical_id: Optional[str],
) -> Dict[str, str]:
    """
    BUG N1 fix: build the mapping original_col → factory_canonical_name.

    For each original column, look up the matching field in the atom's
    field list (by fuzzy snake_case comparison). If a match is found,
    use the factory's canonical field name. Otherwise fall back to
    _clean_col() so the column is still included.

    Returns: {original_col: final_col_name}
    """
    # Base mapping: original → snake_case
    base: Dict[str, str] = {}
    seen: Dict[str, int] = {}
    for col in original_cols:
        clean = _clean_col(col)
        if not clean:
            clean = f"col_{len(base)}"
        if clean in seen:
            seen[clean] += 1
            clean = f"{clean}_{seen[clean]}"
        else:
            seen[clean] = 0
        base[col] = clean

    if not atom_canonical_id:
        return base

    # Load atom fields for this canonical_id
    try:
        from core.paths import ATOMS_PATH
        raw   = json.loads(ATOMS_PATH.read_text(encoding="utf-8"))
        atoms = raw.get("_default", raw) if isinstance(raw, dict) else {}
        atom_fields: Dict[str, str] = {}   # snake_case_name → canonical_name
        for atom in atoms.values():
            if atom.get("canonical_id") != atom_canonical_id:
                continue
            for f in atom.get("fields", []):
                fname = f.get("name", "").strip()
                if fname:
                    atom_fields[_clean_col(fname)] = fname
        if not atom_fields:
            return base
    except Exception as exc:
        log.debug(f"[build_canonical_map] Could not load atom fields: {exc}")
        return base

    # Overlay factory canonical names where they match
    result: Dict[str, str] = {}
    for orig_col, clean_col in base.items():
        canonical = atom_fields.get(clean_col)
        result[orig_col] = canonical if canonical else clean_col

    mapped = sum(1 for o, c in result.items() if c != base[o])
    if mapped:
        log.info(
            f"[build_canonical_map] Applied {mapped} factory canonical names "
            f"for atom '{atom_canonical_id}'"
        )
    return result


def ingest_csv(
    file_path:         Path,
    table_name:        str,
    vertical:          str,
    atom_canonical_id: Optional[str] = None,
    upload_id:         Optional[str] = None,
    customer_data_dir: Optional[Path] = None,
    canonical_map:     Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Read one customer CSV, normalise column names, write to sot_csv/.
    Returns schema profile — column names + types only, no data values.

    BUG N1 fix: accepts canonical_map param. When provided (original→canonical),
    those mappings override _clean_col() so the SOT CSV uses factory field names
    (e.g. 'compensation' instead of 'salary').

    BUG G4+G5 fix: enum storage now uses per-customer customer_data_dir and
    atom_canonical_id as the key.

    The output CSV is saved as:
      - {atom_canonical_id}.csv  if atom_canonical_id is provided
      - {table_name}.csv         as fallback when no atom match was found
    """
    with open(file_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if any(v.strip() for v in r.values())]
        original_cols = list(reader.fieldnames or [])

    if not rows:
        return {"status": "error", "reason": f"Empty file: {file_path.name}"}

    # ── BUG N1: Build column name mapping ────────────────────────────────────
    # Use provided canonical_map if available, otherwise build it fresh.
    # canonical_map: original_col → final_col_name (factory canonical or snake_case)
    if canonical_map:
        col_map = canonical_map
        # Ensure every column in the file is covered (defensive)
        seen: Dict[str, int] = {}
        for col in original_cols:
            if col not in col_map:
                clean = _clean_col(col)
                if not clean:
                    clean = f"col_{len(col_map)}"
                if clean in seen:
                    seen[clean] += 1
                    clean = f"{clean}_{seen[clean]}"
                else:
                    seen[clean] = 0
                col_map[col] = clean
    else:
        col_map = build_canonical_map(original_cols, atom_canonical_id)

    # Infer types from samples
    type_map: Dict[str, str] = {}
    for orig_col, final_col in col_map.items():
        samples = [r.get(orig_col, "") for r in rows[:50]]
        type_map[final_col] = _infer_type(samples)

    # ── Write normalised CSV to sot_csv/ ─────────────────────────────────────
    out_dir   = CUSTOMER_SOT_DIR / vertical
    out_dir.mkdir(parents=True, exist_ok=True)
    save_name = atom_canonical_id if atom_canonical_id else table_name
    out_path  = out_dir / f"{save_name}.csv"
    if atom_canonical_id and atom_canonical_id != table_name:
        log.info(
            f"[sot_ingestion] Saving as '{save_name}.csv' "
            f"(matched atom '{atom_canonical_id}' for uploaded '{table_name}')"
        )

    final_cols = list(col_map.values())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=final_cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({col_map[k]: v for k, v in row.items() if k in col_map})

    fingerprint = _sha256(out_path)
    log.info(f"[sot_ingestion] {table_name} → {save_name}: {len(rows)} rows → {out_path}")

    # ── BUG G4+G5: Extract enums and store per-customer under atom key ────────
    enums = _extract_enums(rows, col_map, type_map)
    # Use customer_data_dir if provided, else fall back to global path (legacy)
    if customer_data_dir:
        _update_field_values(save_name, enums, customer_data_dir)
    else:
        # Fallback: write to global field_values path (should not happen in normal flow)
        _update_field_values_global(save_name, enums)
    log.info(f"[sot_ingestion] {table_name}: {len(enums)} enum columns → {list(enums.keys())}")

    # ── Upsert rows into data_store.json ─────────────────────────────────────
    upsert_result: Dict[str, Any] = {}
    if upload_id and customer_data_dir:
        try:
            from customer.data_store import upsert_rows, compute_column_hash
            clean_rows = [
                {col_map[k]: str(v) if v is not None else "" for k, v in row.items() if k in col_map}
                for row in rows
            ]
            col_hash = compute_column_hash(list(col_map.values()))
            pk_field = _guess_primary_key(list(col_map.values()), atom_canonical_id)
            upsert_result = upsert_rows(
                customer_data_dir = customer_data_dir,
                atom_id           = save_name,
                primary_key_field = pk_field,
                new_rows          = clean_rows,
                upload_id         = upload_id,
                column_hash       = col_hash,
            )
            log.info(
                f"[sot_ingestion] upsert '{save_name}': "
                f"+{upsert_result['rows_added']} added, "
                f"~{upsert_result['rows_updated']} updated, "
                f"={upsert_result['rows_unchanged']} unchanged, "
                f"-{upsert_result['rows_deleted']} deleted"
            )
        except Exception as exc:
            log.warning(f"[sot_ingestion] data_store upsert failed (non-fatal): {exc}")

    return {
        "status":            "ok",
        "table_name":        table_name,
        "saved_as":          save_name,
        "atom_canonical_id": atom_canonical_id,
        "row_count":         len(rows),
        "sot_path":          str(out_path),
        "fingerprint":       fingerprint,
        "enums":             enums,
        "col_map":           col_map,        # expose for cache storage
        "upsert":            upsert_result,
        "schema": {
            "table_name": save_name,
            "row_count":  len(rows),
            "columns": [
                {
                    "name":          final_col,
                    "original_name": orig_col,
                    "type":          type_map[final_col],
                }
                for orig_col, final_col in col_map.items()
            ],
            "enums": enums,
        },
    }


def _update_field_values_global(atom_key: str, enums: Dict[str, List[str]]) -> None:
    """Legacy fallback — writes to global field_values.json. Should not happen in normal flow."""
    from core.paths import FIELD_VALUES_PATH
    fv: Dict = {}
    if FIELD_VALUES_PATH.exists():
        try:
            fv = json.loads(FIELD_VALUES_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    for col, values in enums.items():
        fv[f"{atom_key}.{col}"] = values
    FIELD_VALUES_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIELD_VALUES_PATH.write_text(
        json.dumps(fv, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def infer_relationships(table_schemas: List[Dict]) -> List[Dict]:
    """
    Infer FK relationships between tables by column naming patterns.
    e.g. orders.customer_id → customers.id
    """
    relationships = []
    tables = {
        s["table_name"]: {c["name"] for c in s["columns"]}
        for s in table_schemas
    }

    seen = set()
    for from_table, from_cols in tables.items():
        for col in from_cols:
            if not col.endswith("_id"):
                continue
            ref_hint = col[:-3]
            for to_table, to_cols in tables.items():
                if to_table == from_table:
                    continue
                key = (from_table, col, to_table)
                if key in seen:
                    continue
                if ref_hint in to_table or to_table.startswith(ref_hint):
                    to_col = col if col in to_cols else ("id" if "id" in to_cols else None)
                    if to_col:
                        relationships.append({
                            "from_table": from_table,
                            "from_col":   col,
                            "to_table":   to_table,
                            "to_col":     to_col,
                            "confidence": "inferred",
                        })
                        seen.add(key)
                        break

    return relationships


def build_customer_schema(
    table_schemas:     List[Dict],
    relationships:     List[Dict],
    customer_data_dir: Optional[Path] = None,
) -> Dict:
    """
    Build the customer_schema payload for VerticalSchemaAgent CUSTOMER mode.
    Schema only — no data values.

    BUG G4 fix: reads enums from per-customer customer_data_dir when provided.
    """
    all_enums: Dict[str, Dict] = {}

    # Read from field_values.json (single source of truth for enums)
    # field_values format: {"atom_key.col": [values]} — convert to nested {atom_key: {col: [values]}}
    if customer_data_dir:
        fv_path = customer_data_dir / "field_values.json"
    else:
        from core.paths import FIELD_VALUES_PATH
        fv_path = FIELD_VALUES_PATH

    if fv_path.exists():
        try:
            flat = json.loads(fv_path.read_text(encoding="utf-8"))
            for key, values in flat.items():
                if "." in key:
                    atom_key, col = key.split(".", 1)
                    all_enums.setdefault(atom_key, {})[col] = values
        except Exception:
            pass

    return {
        "tables": [
            {
                "name": s["table_name"],
                "columns": [
                    {"name": c["name"], "type": c["type"]}
                    for c in s["columns"]
                ],
                "enums": all_enums.get(s["table_name"], s.get("enums", {})),
            }
            for s in table_schemas
        ],
        "relationships": [
            {
                "from_table": r["from_table"],
                "from_col":   r["from_col"],
                "to_table":   r["to_table"],
                "to_col":     r["to_col"],
            }
            for r in relationships
        ],
        "enums": all_enums,
    }


def update_fingerprints(table_results: List[Dict], vertical: str) -> None:
    """
    Write/update data_fingerprints.json in customer_runtime with SHA-256
    hashes of the ingested SOT CSVs.
    """
    existing: Dict = {}
    if CUSTOMER_FINGERPRINTS_PATH.exists():
        try:
            existing = json.loads(CUSTOMER_FINGERPRINTS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    for result in table_results:
        if result.get("status") == "ok":
            key = f"{vertical}/{result['table_name']}"
            existing[key] = {
                "sha256":     result["fingerprint"],
                "table":      result["table_name"],
                "vertical":   vertical,
                "updated_at": _now(),
            }

    CUSTOMER_FINGERPRINTS_PATH.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info(f"[sot_ingestion] Updated fingerprints: {len(table_results)} tables")


# ── Vertical auto-detection ────────────────────────────────────────────────────

# ── Domain keyword vocabulary ──────────────────────────────────────────────────
# Substring-matched against normalised column names. Keep this large — the
# bigger and more specific these lists are, the less the LLM fallback is needed.
# When a file is misclassified: look at its columns, find the obvious domain
# word that didn't trigger, add it here. One word fixes all future files.
_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "supply_chain": [
        "supplier", "vendor", "purchase_order", "procurement", "sku",
        "bom", "lead_time", "reorder", "warehouse", "stock", "inventory",
        "replenishment", "sourcing", "rfq", "goods_receipt", "grn",
        "raw_material", "finished_good", "safety_stock", "moq", "incoterm",
        "duty", "tariff", "origin_country", "commodity", "category_manager",
    ],
    "logistics": [
        "shipment", "carrier", "delivery", "tracking", "fleet", "route",
        "dispatch", "freight", "consignment", "manifest", "waybill", "pod",
        "last_mile", "transporter", "lorry", "driver", "vehicle", "trip",
        "load", "unload", "dock", "hub", "lane", "tms", "docket",
        "eta", "ata", "origin_hub", "destination_hub", "weight_kg",
    ],
    "retail_shopify": [
        "order", "customer", "product", "cart", "checkout", "refund",
        "fulfillment", "variant", "storefront", "discount", "collection",
        "listing", "shopify", "pos", "coupon", "sku", "basket", "session",
        "abandoned", "conversion", "gmv", "aov", "return", "exchange",
        "gift_card", "loyalty", "reward", "store_credit", "wishlist",
    ],
    "hr": [
        "employee", "salary", "payroll", "department", "hire", "termination",
        "headcount", "leave", "appraisal", "designation", "attendance",
        "recruiter", "onboarding", "offboarding", "band", "grade", "ctc",
        "bonus", "pf", "esi", "lop", "shift", "roster", "manager",
        "reporting_to", "doj", "dol", "probation", "confirmation",
    ],
    "finance": [
        "invoice", "payment", "revenue", "expense", "ledger", "budget",
        "account", "fiscal", "receivable", "payable", "journal", "credit",
        "debit", "tax", "reconciliation", "gst", "tds", "vat", "p&l",
        "balance_sheet", "cashflow", "bank", "transaction", "settlement",
        "write_off", "provision", "accrual", "depreciation", "capex", "opex",
    ],
    "sales": [
        "opportunity", "pipeline", "deal", "prospect", "quota", "commission",
        "lead", "crm", "conversion", "forecast", "territory", "rep",
        "closing", "stage", "win_rate", "account_manager", "demo",
        "proposal", "negotiation", "renewal", "upsell", "cross_sell",
        "churn", "nrr", "arr", "mrr", "bookings", "revenue_target",
    ],
}

# Prefix map: filename/table_name starting with one of these → vertical decided
# immediately, no scoring. Longer prefixes first — "retail_shopify_" before "retail_".
_PREFIX_MAP: List[tuple] = [
    ("supply_chain_",    "supply_chain"),
    ("logistics_",       "logistics"),
    ("retail_shopify_",  "retail_shopify"),
    ("retail_",          "retail_shopify"),
    ("hr_",              "hr"),
    ("human_resources_", "hr"),
    ("finance_",         "finance"),
    ("sales_",           "sales"),
]

# Keyword vote count must exceed this for the dictionary to return a result.
# Anything at or below goes to the LLM fallback.
_CONFIDENCE_THRESHOLD = 2


def _llm_detect_vertical(
    table_name:        str,
    cols_lower:        List[str],
    candidate_domains: List[str],
) -> Optional[str]:
    """
    LLM fallback for vertical detection — routes through the factory server.

    The factory holds the Anthropic API key so no direct LLM call is made here.
    Mirrors the same pattern as canonicalize_via_factory() in canonicalizer.py.

    candidate_domains is pre-scoped by detect_vertical():
      - Tie:            only the tied domains
      - Low/no signal:  all known domains

    Returns the chosen domain string, or None if the factory call fails.
    On None the caller returns its keyword best-guess with low_confidence=True.
    """
    try:
        from core.config import settings
        from customer.canonicalizer import detect_vertical_via_factory
        return detect_vertical_via_factory(
            table_name        = table_name,
            columns           = cols_lower,
            candidate_domains = candidate_domains,
            factory_url       = settings.FACTORY_URL,
        )
    except Exception as exc:
        log.warning(f"[detect_vertical] Factory call failed (non-fatal): {exc}")
        return None


def detect_vertical(
    table_name:        str,
    columns:           List[str],
    atoms_path:        Optional[Path] = None,  # kept for call-site compatibility, unused
    original_filename: Optional[str]  = None,
) -> Dict[str, Any]:
    """
    Infer the vertical for a customer table. Three steps in order:

      Step 1 — Prefix hard-rule (instant, no scoring)
               Filename stem or table_name starts with a known domain prefix
               → return immediately, no further work.

      Step 2 — Keyword voting
               Count column names that contain each domain's vocabulary terms
               (substring match, one hit per column per domain).
               Clear winner (score >= 2, no tie) → return result.

      Step 3 — LLM fallback (only when step 2 can't decide)
               Triggered by: score == 0 (no signal), score < 2 (weak signal),
               or a tie at the top. Sends table name + columns only — no row
               data. Candidate list is tightened to tied domains when possible.
               If LLM call fails → returns best keyword guess with low_confidence.

    Return shape (identical to previous implementation — all call sites safe):
      {
        "vertical":          str,
        "confidence":        float,   # 0-1, for display only
        "method":            str,     # "prefix" | "keywords" | "llm" | "llm_failed"
        "low_confidence":    bool,
        "scores":            dict,    # raw keyword vote counts
        "atom_canonical_id": None,
        "atom_overlap":      0.0,
      }
    """
    import os as _os

    all_domains    = list(_DOMAIN_KEYWORDS.keys())
    cols_lower     = [c.lower().strip() for c in columns if c.strip()]

    # ── Build filename candidates (stem only, normalised) ─────────────────────
    prefix_candidates: List[str] = []
    if original_filename:
        stem = _os.path.splitext(original_filename)[0].lower().strip()
        stem = re.sub(r"[\s\-]+", "_", stem)
        prefix_candidates.append(stem)
    prefix_candidates.append(table_name.lower())

    # ── Step 1: Prefix hard-rule ──────────────────────────────────────────────
    for prefix, domain in _PREFIX_MAP:
        if any(cand == prefix.rstrip("_") or cand.startswith(prefix)
               for cand in prefix_candidates):
            log.info(
                f"[detect_vertical] {table_name!r} → {domain!r} "
                f"(method=prefix, matched={prefix!r})"
            )
            return {
                "vertical":          domain,
                "confidence":        1.0,
                "method":            "prefix",
                "low_confidence":    False,
                "scores":            {domain: 99},
                "atom_canonical_id": None,
                "atom_overlap":      0.0,
            }

    # ── Step 2: Keyword voting ────────────────────────────────────────────────
    scores: Dict[str, int] = {domain: 0 for domain in _DOMAIN_KEYWORDS}

    for domain, keywords in _DOMAIN_KEYWORDS.items():
        for col in cols_lower:
            for kw in keywords:
                if kw in col:
                    scores[domain] += 1
                    break  # one hit per column per domain — no double counting

    best_score  = max(scores.values())
    top_domains = [d for d in scores if scores[d] == best_score]

    # Conditions that route to LLM:
    #   a) best_score == 0  → no signal at all
    #   b) best_score <  2  → signal too weak to trust
    #   c) len(top_domains) > 1  → genuine tie
    needs_llm = best_score == 0 or best_score < _CONFIDENCE_THRESHOLD or len(top_domains) > 1

    if not needs_llm:
        # Clean keyword win — return directly
        best_domain = top_domains[0]
        confidence  = min(best_score / 10.0, 1.0)
        log.info(
            f"[detect_vertical] {table_name!r} → {best_domain!r} "
            f"(method=keywords, score={best_score}, confidence={confidence:.2f}) "
            f"scores={scores}"
        )
        return {
            "vertical":          best_domain,
            "confidence":        confidence,
            "method":            "keywords",
            "low_confidence":    False,
            "scores":            scores,
            "atom_canonical_id": None,
            "atom_overlap":      0.0,
        }

    # ── Step 3: LLM fallback ──────────────────────────────────────────────────
    # Tighten candidate list: tied domains when there's a tie, all domains
    # when signal is absent or too weak.
    if len(top_domains) > 1 and best_score > 0:
        # Genuine tie — restrict LLM to only the tied domains
        candidates = top_domains
        trigger    = f"tie({best_score}) between {top_domains}"
    else:
        # No signal or weak signal — open decision across all domains
        candidates = all_domains
        trigger    = f"low_signal(best_score={best_score})"

    log.info(
        f"[detect_vertical] {table_name!r} → LLM fallback "
        f"(trigger={trigger}, candidates={candidates}) scores={scores}"
    )

    chosen = _llm_detect_vertical(table_name, cols_lower, candidates)

    if chosen:
        log.info(
            f"[detect_vertical] {table_name!r} → {chosen!r} (method=llm)"
        )
        return {
            "vertical":          chosen,
            "confidence":        0.7,   # LLM result: reasonably confident but not certain
            "method":            "llm",
            "low_confidence":    False,
            "scores":            scores,
            "atom_canonical_id": None,
            "atom_overlap":      0.0,
        }

    # LLM call failed — return best keyword guess with low_confidence flag.
    # top_domains[0] is deterministic (dict insertion order = _DOMAIN_KEYWORDS order).
    fallback = top_domains[0] if best_score > 0 else all_domains[0]
    log.warning(
        f"[detect_vertical] {table_name!r} → {fallback!r} "
        f"(method=llm_failed, using keyword fallback)"
    )
    return {
        "vertical":          fallback,
        "confidence":        min(best_score / 10.0, 0.3),
        "method":            "llm_failed",
        "low_confidence":    True,
        "scores":            scores,
        "atom_canonical_id": None,
        "atom_overlap":      0.0,
    }


# ── Excel ingestion ────────────────────────────────────────────────────────────

def ingest_excel(
    file_bytes:        bytes,
    filename:          str,
    sot_dir:           "Path",
    customer_data_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Ingest an Excel workbook (.xlsx / .xls) — each sheet becomes one virtual table.
    Applies canonical column mapping (N1 fix) per sheet.
    customer_data_dir: per-customer data dir for enum storage (field_values.json).
    """
    import csv as _csv
    import tempfile

    from customer.excel_parser import parse_excel
    from customer.sot_ingestion import ingest_csv, detect_vertical
    from customer.data_cleaner import clean_sheet, write_transformation_log

    try:
        sheets = parse_excel(file_bytes, filename)
    except ValueError as exc:
        log.error(f"[ingest_excel] Parse failed for '{filename}': {exc}")
        return [{"status": "error", "reason": str(exc), "file": filename}]

    if not sheets:
        return [{
            "status": "error",
            "reason": f"'{filename}' contains no usable sheets (all sheets were empty or header-only).",
            "file":   filename,
        }]

    results: List[Dict[str, Any]] = []
    clean_reports: List[Dict] = []

    for sheet in sheets:
        sheet_name          = sheet["sheet_name"]
        original_sheet_name = sheet["original_sheet_name"]
        original_cols       = sheet["columns"]

        log.info(
            f"[ingest_excel] Processing sheet '{original_sheet_name}' "
            f"→ table '{sheet_name}' ({sheet['row_count']} rows)"
        )

        tmp_path: Optional[Path] = None
        try:
            # ── Detect vertical first so clean_sheet gets the correct date convention ──
            detection = detect_vertical(sheet_name, original_cols, original_filename=filename)
            vertical  = detection["vertical"]

            # ── Data cleaning pass (vertical now known) ───────────────────────────────
            sheet, _report = clean_sheet(sheet, options={"vertical": vertical})
            clean_reports.append(_report)
            rows = sheet["rows"]

            with tempfile.NamedTemporaryFile(
                suffix=".csv", delete=False, mode="w",
                newline="", encoding="utf-8"
            ) as tmp:
                tmp_path = Path(tmp.name)
                writer   = _csv.DictWriter(tmp, fieldnames=original_cols, extrasaction="ignore")
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: ("" if v is None else str(v)) for k, v in row.items()})

            log.info(
                f"[ingest_excel] '{original_sheet_name}' → vertical='{vertical}' "
                f"confidence={detection['confidence']:.2f} method={detection['method']}"
            )

            import customer.sot_ingestion as _sot_mod
            _orig_sot            = _sot_mod.CUSTOMER_SOT_DIR
            _sot_mod.CUSTOMER_SOT_DIR = sot_dir
            try:
                result = ingest_csv(
                    tmp_path,
                    table_name        = sheet_name,
                    vertical          = vertical,
                    atom_canonical_id = detection.get("atom_canonical_id"),
                    customer_data_dir = customer_data_dir,
                )
            finally:
                _sot_mod.CUSTOMER_SOT_DIR = _orig_sot

            result["original_sheet_name"] = original_sheet_name
            result["detection"]           = detection
            results.append(result)

        except Exception as exc:
            log.error(f"[ingest_excel] Failed on sheet '{original_sheet_name}': {exc}")
            results.append({
                "status":              "error",
                "reason":              str(exc),
                "table_name":          sheet_name,
                "original_sheet_name": original_sheet_name,
                "detection":           {},
            })
        finally:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    if customer_data_dir and clean_reports:
        try:
            write_transformation_log(customer_data_dir, filename, clean_reports)
        except Exception as _log_exc:
            log.warning(f"[ingest_excel] transformation_log write failed (non-fatal): {_log_exc}")

    log.info(
        f"[ingest_excel] '{filename}' complete — "
        f"{sum(1 for r in results if r.get('status') == 'ok')} ok, "
        f"{sum(1 for r in results if r.get('status') == 'error')} errors"
    )
    return results


def ingest_sql(
    file_bytes:        bytes,
    filename:          str,
    sot_dir:           "Path",
    customer_data_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Ingest a MySQL or PostgreSQL .sql dump file.
    Each table in the dump becomes one virtual table, mirroring ingest_excel().
    Uses sql_parser.parse_sql() to extract tables -> feeds each into ingest_csv().
    customer_data_dir: per-customer data dir for enum storage (field_values.json).
    """
    import csv as _csv
    import tempfile

    from customer.sql_parser import parse_sql
    from customer.sot_ingestion import ingest_csv, detect_vertical
    from customer.data_cleaner import clean_sheet, write_transformation_log

    try:
        tables = parse_sql(file_bytes, filename)
    except Exception as exc:
        log.error(f"[ingest_sql] Parse failed for '{filename}': {exc}")
        return [{"status": "error", "reason": str(exc), "file": filename}]

    if not tables:
        return [{
            "status": "error",
            "reason": (
                f"'{filename}' contains no usable tables "
                f"(no INSERT/COPY data found, or all tables were empty)."
            ),
            "file": filename,
        }]

    results: List[Dict[str, Any]] = []
    clean_reports: List[Dict] = []

    for table_info in tables:
        table_name = table_info["table_name"]
        columns    = table_info["columns"]
        rows       = table_info["rows"]
        dialect    = table_info.get("dialect", "unknown")

        log.info(
            f"[ingest_sql] Processing table '{table_name}' "
            f"({table_info['row_count']} rows, dialect={dialect})"
        )

        tmp_path: Optional[Path] = None
        try:
            # Build a sheet-like dict so clean_sheet() can be reused
            sheet = {
                "sheet_name":          table_name,
                "original_sheet_name": table_name,
                "columns":             columns,
                "rows":                rows,
                "row_count":           len(rows),
            }

            detection = detect_vertical(table_name, columns, original_filename=filename)
            vertical  = detection["vertical"]

            # Data cleaning pass (vertical now known)
            sheet, _report = clean_sheet(sheet, options={"vertical": vertical})
            clean_reports.append(_report)
            cleaned_rows = sheet["rows"]

            # Write to temp CSV for ingest_csv()
            with tempfile.NamedTemporaryFile(
                suffix=".csv", delete=False, mode="w",
                newline="", encoding="utf-8"
            ) as tmp:
                tmp_path = Path(tmp.name)
                writer   = _csv.DictWriter(tmp, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                for row in cleaned_rows:
                    writer.writerow({k: ("" if v is None else str(v)) for k, v in row.items()})

            log.info(
                f"[ingest_sql] '{table_name}' -> vertical='{vertical}' "
                f"confidence={detection['confidence']:.2f} method={detection['method']}"
            )

            import customer.sot_ingestion as _sot_mod
            _orig_sot             = _sot_mod.CUSTOMER_SOT_DIR
            _sot_mod.CUSTOMER_SOT_DIR = sot_dir
            try:
                result = ingest_csv(
                    tmp_path,
                    table_name        = table_name,
                    vertical          = vertical,
                    atom_canonical_id = detection.get("atom_canonical_id"),
                    customer_data_dir = customer_data_dir,
                )
            finally:
                _sot_mod.CUSTOMER_SOT_DIR = _orig_sot

            result["original_table_name"] = table_name
            result["detection"]           = detection
            result["sql_dialect"]         = dialect
            results.append(result)

        except Exception as exc:
            log.error(f"[ingest_sql] Failed on table '{table_name}': {exc}")
            results.append({
                "status":              "error",
                "reason":              str(exc),
                "table_name":          table_name,
                "original_table_name": table_name,
                "detection":           {},
                "sql_dialect":         dialect,
            })
        finally:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    if customer_data_dir and clean_reports:
        try:
            write_transformation_log(customer_data_dir, filename, clean_reports)
        except Exception as _log_exc:
            log.warning(f"[ingest_sql] transformation_log write failed (non-fatal): {_log_exc}")

    log.info(
        f"[ingest_sql] '{filename}' complete — "
        f"{sum(1 for r in results if r.get('status') == 'ok')} ok, "
        f"{sum(1 for r in results if r.get('status') == 'error')} errors"
    )
    return results
