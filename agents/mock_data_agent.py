"""
agents/mock_data_agent.py
==========================
MockDataAgent — generates realistic synthetic CSV mock data for a vertical
using Claude as the primary data generator.

DESIGN:
  - Claude generates ALL data — names, IDs, amounts, dates, statuses.
    Not hardcoded patterns. Real-looking business data.
  - Generates in batches of 50 rows per API call.
  - Each batch receives context from the previous batch (last 5 rows +
    ID ranges used) so the data is coherent across batches.
  - Claude returns JSON arrays — we convert to CSV (safe, no escaping issues).
  - FK consistency enforced: dimension tables first (topological sort),
    child tables receive parent ID pools so JOINs produce real results.

ROW COUNTS:
  state     → 500 rows (100 unique entity IDs × 5 state changes each)
  record    → 250 rows
  event     → 250 rows
  snapshot  → 250 rows
  dimension → realistic small counts (warehouses=5, suppliers=15, customers=30...)

OUTPUTS:
  data/mock_data/{vertical}/{canonical_id}.csv  — one CSV per atom
  data/data_fingerprints.json — SHA-256 hash per generated CSV (per canonical_id)
    No consumer yet — Step 7 (Compilation) will use this to detect when
    mock data has changed and invalidate/recompile stale cached aterms.
  discover_enums runs after all CSVs written → field_values.json updated
"""

import csv
import hashlib
import json
import logging
import random
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from anthropic import Anthropic, APIStatusError, RateLimitError, APIConnectionError

from core.config import settings
import core.paths as _paths
from core.paths import LOGS_DIR
from services import atom_registry
from services.relationship_registry import list_relationships

log = logging.getLogger(__name__)

def _get_client():
    """Build a fresh client per call so a runtime API-key update takes effect immediately."""
    return Anthropic(api_key=settings.ANTHROPIC_API_KEY)


def _get_model():
    return settings.ANTHROPIC_MODEL


# ── Row count caps — read from env vars (set via Settings UI) ──────────────────
import os as _os
_ROW_COUNTS = {
    "state":    int(_os.getenv("MOCK_ROWS_STATE",    "75")),
    "record":   int(_os.getenv("MOCK_ROWS_RECORD",   "50")),
    "event":    int(_os.getenv("MOCK_ROWS_EVENT",    "50")),
    "snapshot": int(_os.getenv("MOCK_ROWS_SNAPSHOT", "50")),
}

_DIMENSION_COUNTS = {
    "warehouse": 5,  "warehouses": 5,
    "region":    8,  "regions":    8,
    "country":   12, "countries":  12,
    "currency":  10, "currencies": 10,
    "supplier":  15, "suppliers":  15,
    "vendor":    12, "vendors":    12,
    "employee":  20, "employees":  20,
    "staff":     20,
    "department": 8, "departments": 8,
    "category":  15, "categories": 15,
    "product":   20, "products":   20,
    "item":      20, "items":      20,
    "sku":       20, "skus":       20,
    "customer":  20, "customers":  20,
    "account":   20, "accounts":   20,
    "location":  15, "locations":  15,
    "store":     10, "stores":     10,
    "channel":    8, "channels":    8,
    "segment":    6, "segments":    6,
    "cost_center":10,"cost_centers":10,
    "carrier":   10, "carriers":   10,
    "driver":    15, "drivers":    15,
    "vehicle":   15, "vehicles":   15,
    "asset":     20, "assets":     20,
    "position":  15, "positions":  15,
    "court":      8, "courts":      8,
    "attorney":  15, "attorneys":  15,
    "policy":    20, "policies":   20,
}
_DEFAULT_DIMENSION_COUNT = int(_os.getenv("MOCK_ROWS_DIMENSION", "15"))


def _field_aware_row_cap(atom: Dict, base_rows: int) -> int:
    """
    Cap row count based on number of fields to stay within 8192 output tokens.
    Rough estimate: each row with N fields uses ~N*40 chars.
    8192 tokens ≈ 16000 chars safely → max_rows = 16000 / (fields * 40)
    """
    n_fields = len(atom.get("fields", []))
    if n_fields == 0:
        return base_rows
    safe_chars  = 14000  # conservative limit
    chars_per_row = n_fields * 45
    max_rows = max(10, safe_chars // chars_per_row)
    return min(base_rows, max_rows)


def _row_count_for(atom: Dict, row_counts: Optional[Dict[str, int]] = None) -> int:
    rt  = atom.get("record_type", "record")
    cid = atom.get("canonical_id", "").lower()
    if rt == "dimension":
        for keyword, count in _DIMENSION_COUNTS.items():
            if keyword in cid:
                return count
        return _DEFAULT_DIMENSION_COUNT
    counts = row_counts if row_counts is not None else _ROW_COUNTS
    base = counts.get(rt, 50)
    return _field_aware_row_cap(atom, base)


# ── Topological sort — dimensions first ───────────────────────────────────────

def _topological_sort(atoms: List[Dict], rels: List[Dict]) -> List[Dict]:
    """Sort atoms so parent/dimension tables come before child tables."""
    atom_map = {a["canonical_id"]: a for a in atoms}
    deps: Dict[str, set] = {a["canonical_id"]: set() for a in atoms}
    for r in rels:
        fa, ta = r.get("from_atom", ""), r.get("to_atom", "")
        if fa in deps and ta in deps and fa != ta:
            deps[fa].add(ta)

    in_degree = {cid: len(parents) for cid, parents in deps.items()}
    queue     = [cid for cid, deg in in_degree.items() if deg == 0]
    order     = []

    while queue:
        queue.sort(key=lambda c: (
            0 if atom_map[c].get("record_type") == "dimension" else 1, c
        ))
        node = queue.pop(0)
        order.append(node)
        for cid, parents in deps.items():
            if node in parents:
                parents.discard(node)
                if not parents and cid not in order:
                    queue.append(cid)

    remaining = [a["canonical_id"] for a in atoms if a["canonical_id"] not in order]
    order.extend(remaining)
    return [atom_map[cid] for cid in order if cid in atom_map]


# ── Build system prompt for the agent ─────────────────────────────────────────

def _build_system_prompt(atom: Dict, vertical: str, fk_context: Dict[str, List]) -> str:
    cid    = atom["canonical_id"]
    rt     = atom.get("record_type", "record")
    fields = atom.get("fields", [])

    field_descriptions = "\n".join(
        f"  - {f['name']} ({f['type']}, role={f['role']}, additivity={f['additivity']})"
        for f in fields
    )

    fk_info = ""
    if fk_context:
        fk_lines = "\n".join(
            f"  - {field} must be drawn from: {ids[:10]}{'...' if len(ids)>10 else ''}"
            for field, ids in fk_context.items()
        )
        fk_info = f"\nFK CONSTRAINTS (use ONLY these values for these fields):\n{fk_lines}"

    state_note = ""
    if rt == "state":
        state_note = """
IMPORTANT — STATE TABLE RULES:
  This is a state table. Each entity has MULTIPLE rows showing state changes over time.
  - Use the same entity ID for 5 consecutive rows (showing state progression)
  - The dedup_sort_col (date field) must increase with each state change
  - Status should progress logically (e.g. pending → processing → shipped → delivered)
  - Do NOT create a new unique entity for every row"""

    return f"""You are a business data generator for the GPL factory system.

Your job is to generate realistic synthetic mock data for the '{vertical}' business vertical.
You are generating data for the atom: {cid}
Record type: {rt}

FIELDS:
{field_descriptions}
{fk_info}
{state_note}

RULES:
  1. ALL data must look realistic — real business names, proper IDs, realistic amounts
  2. IDs should look real: use formats like INV-2847, ORD-10492, CUST-3841 (random numbers, not sequential 0001)
  3. Company names: real-sounding names like "Pacific Rim Logistics", "Meridian Healthcare Group"
  4. Person names: real first/last names from diverse backgrounds
  5. Amounts: realistic for the entity type (invoices in thousands, not 1-50000 randomly)
  6. Dates: YYYY-MM-DD format, within last 2 years, logically consistent
  7. Status values: distribute realistically (more completed than cancelled)
  8. Measure fields (additive): positive numbers appropriate for business context
  9. Measure fields (semi_additive): rates/prices/percentages in realistic ranges
  10. FK fields: ONLY use values from the FK constraints above — do not invent new ones

Return ONLY a valid JSON array of row objects. No explanation, no markdown, no extra text.
Each object must have exactly these keys: {[f['name'] for f in fields]}"""


# ── Build batch prompt ─────────────────────────────────────────────────────────


def _call_claude(
    system_prompt:   str,
    user_prompt:     str,
    expected_fields: List[str],
) -> Tuple[List[Dict], str]:
    """
    Single Claude call — generates all rows in one shot.
    Retries once on JSON parse error with a stricter prompt.
    Stops immediately on credit exhaustion.
    Rate limit / overload → exponential backoff up to 3 retries.
    Returns (rows, error_message).
    """
    for attempt in range(4):  # 1 normal + 3 rate-limit retries
        try:
            prompt = user_prompt if attempt == 0 else (
                user_prompt + "\n\nCRITICAL: Return ONLY a valid JSON array. "
                "No markdown, no commentary, no trailing commas. "
                "Every string value must be properly quoted and closed."
            )
            response = _get_client().messages.create(
                model=_get_model(),
                max_tokens=8192,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()

            # Strip markdown fences
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            rows = json.loads(text)

            if not isinstance(rows, list):
                return [], f"Expected JSON array, got {type(rows).__name__}"

            # Clean rows — ensure all expected fields present
            cleaned = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                cleaned.append({f: row.get(f, "") for f in expected_fields})

            return cleaned, ""

        except json.JSONDecodeError as e:
            if attempt == 0:
                # Retry once with stricter prompt
                log.warning(f"[MockDataAgent] JSON parse error, retrying with stricter prompt: {e}")
                continue
            return [], f"JSON parse error after retry: {e}"

        except (RateLimitError, APIStatusError) as e:
            status  = getattr(e, "status_code", 0)
            err_msg = str(e)

            # Credit exhaustion — stop everything immediately
            if "credit balance" in err_msg.lower() or "too low" in err_msg.lower():
                log.error("[MockDataAgent] CREDIT EXHAUSTED — stopping immediately.")
                return [], f"CREDIT_EXHAUSTED: {e}"

            # Rate limit / overload — backoff and retry
            wait = (2 ** attempt) * 5
            log.warning(f"[MockDataAgent] API error {status}, waiting {wait}s (attempt {attempt+1})…")
            time.sleep(wait)
            continue

        except APIConnectionError as e:
            wait = (2 ** attempt) * 5
            log.warning(f"[MockDataAgent] Connection error, waiting {wait}s (attempt {attempt+1})…")
            time.sleep(wait)
            continue

        except Exception as e:
            err_str = str(e).lower()
            if any(x in err_str for x in ["529","overloaded","rate_limit","429","too_many_requests"]):
                wait = (2 ** attempt) * 5
                log.warning(f"[MockDataAgent] Overload, waiting {wait}s (attempt {attempt+1})…")
                time.sleep(wait)
                continue
            return [], f"API error: {e}"

    return [], "Failed after retries"


def _generate_atom(atom, vertical, fk_pools, row_counts: Optional[Dict[str, int]] = None):
    """Generate all rows for one atom in a single Claude call."""
    cid    = atom["canonical_id"]
    fields = atom.get("fields", [])
    gkeys  = atom.get("grain_keys", [])

    expected_fields = [f["name"] for f in fields]
    total_rows      = _row_count_for(atom, row_counts)

    pk_field = next((f["name"] for f in fields if f.get("role") == "primary_key"), None)
    if not pk_field and gkeys:
        pk_field = gkeys[0]

    # Build FK context from relationships
    rels   = list_relationships()
    fk_ctx = {}
    for r in rels:
        if r.get("from_atom") == cid:
            pool_key = "{}.{}".format(r["to_atom"], r["to_field"])
            if pool_key in fk_pools:
                fk_ctx[r["from_field"]] = fk_pools[pool_key]

    system_prompt = _build_system_prompt(atom, vertical, fk_ctx)

    # Load declared states from seed file for this atom
    # This ensures status/state fields use exact values the compiler expects
    seed_state_instructions = ""
    try:
        import core.paths as _cp
        seed_files = list((_cp.DATA_DIR / "seeds").glob(f"{vertical}_seed.json"))
        if seed_files:
            seed_data = json.loads(seed_files[0].read_text(encoding="utf-8"))
            entities  = seed_data.get("entities", {})
            # Find entity that maps to this atom
            for ent_name, ent_def in entities.items():
                if ent_def.get("atom") == cid:
                    states = ent_def.get("states", [])
                    if states:
                        # Find which field is the state/status field
                        state_field = next(
                            (f["name"] for f in fields
                             if any(kw in f["name"].lower()
                                    for kw in ["status","state","stage","type","phase","category"])
                             and f.get("role") not in ("primary_key","measure","time")),
                            None
                        )
                        if state_field:
                            seed_state_instructions = (
                                "\n\nSTATE FIELD CONSTRAINT - CRITICAL:"
                                "\nThe field '{}' MUST use ONLY these exact values (case-sensitive): {}"
                                "\nDo NOT use any other values for this field."
                            ).format(state_field, states)
                    break
    except Exception:
        pass

    # FK constraint instructions — very explicit to prevent Claude making up IDs
    fk_instructions = ""
    if fk_ctx:
        fk_lines = [
            "- {} MUST be one of these EXACT values (copy them verbatim, do not invent new ones): {}".format(
                f, list(v)[:30]
            )
            for f, v in fk_ctx.items()
        ]
        fk_instructions = (
            "\n\nCRITICAL FK CONSTRAINTS — you MUST follow these exactly:\n" +
            "\n".join(fk_lines) +
            "\nDo NOT invent new IDs for these fields. Use ONLY the values listed above."
        )

    # Build the uniqueness constraint text for the prompt
    if len(gkeys) > 1:
        uniqueness_rule = (
            "3. The COMBINATION of ({}) must be UNIQUE across all {} rows — "
            "no two rows may have the same values for ALL of these fields together. "
            "Each individual field CAN repeat, but the composite combination cannot."
        ).format(", ".join(gkeys), total_rows)
    else:
        uniqueness_rule = "3. Every {} must be UNIQUE across all {} rows".format(
            pk_field or "primary key", total_rows
        )

    from datetime import datetime as _dt
    _today        = _dt.now()
    _current_month= _today.strftime("%Y-%m")
    _current_year = _today.strftime("%Y")
    _month_start  = _today.replace(day=1).strftime("%Y-%m-%d")
    _year_start   = _today.replace(month=1, day=1).strftime("%Y-%m-%d")
    _two_yrs_ago  = _today.replace(year=_today.year - 2).strftime("%Y-%m-%d")

    date_rule = (
        "5. Date fields MUST use dates relative to today ({today}). Distribution:\n"
        "   - At least 30% of rows: dates in current month ({month_start} to {today_str})\n"
        "   - At least 60% of rows: dates in current year ({year_start} to {today_str})\n"
        "   - Remaining rows: dates going back no further than {two_yrs_ago}\n"
        "   - NEVER use future dates. NEVER use dates older than 2 years.\n"
        "   - This ensures time-scoped queries (this_month, this_quarter, ytd) return data."
    ).format(
        today      = _today.strftime("%Y-%m-%d"),
        today_str  = _today.strftime("%Y-%m-%d"),
        month_start= _month_start,
        year_start = _year_start,
        two_yrs_ago= _two_yrs_ago,
    )

    user_prompt = (
        "Generate exactly {} rows of realistic mock data for the ".format(total_rows) +
        "'{}' table as a JSON array.\n\n".format(cid) +
        "Rules:\n" +
        "1. Return ONLY a valid JSON array - no markdown, no explanation\n" +
        "2. Every row must have ALL these fields: {}\n".format(expected_fields) +
        uniqueness_rule + "\n" +
        "4. Use realistic, varied business data - no placeholder text\n" +
        date_rule +
        fk_instructions +
        seed_state_instructions
    )

    log.info("[MockDataAgent] Generating {} — {} rows in single call".format(cid, total_rows))

    rows, error = _call_claude(
        system_prompt   = system_prompt,
        user_prompt     = user_prompt,
        expected_fields = expected_fields,
    )

    if error:
        if "CREDIT_EXHAUSTED" in error:
            raise RuntimeError("CREDIT_EXHAUSTED — stopping pipeline. Top up your Anthropic account.")
        log.error("[MockDataAgent] Failed to generate {}: {}".format(cid, error))
        return [], [error]

    if not rows:
        log.warning("[MockDataAgent] No rows returned for {}".format(cid))
        return [], ["empty response"]

    # Deduplicate on composite grain keys (or single PK if no composite)
    dedup_keys = gkeys if len(gkeys) > 1 else ([pk_field] if pk_field else [])
    if dedup_keys:
        seen, deduped, dropped = set(), [], 0
        for row in rows:
            # Build composite key tuple from all grain key values
            composite = tuple(str(row.get(k, "")) for k in dedup_keys)
            if composite in seen:
                dropped += 1
                continue
            seen.add(composite)
            deduped.append(row)
        if dropped:
            log.warning(
                "[MockDataAgent] {}: dropped {} duplicate composite grain ({}) rows".format(
                    cid, dropped, "+".join(dedup_keys)
                )
            )
        rows = deduped

    # Redistribute FK columns for realistic distribution.
    # IMPORTANT: Skip any FK field that is part of the composite grain key —
    # redistributing grain key fields after dedup destroys composite uniqueness
    # and re-introduces duplicate grain combinations.
    grain_key_set = set(gkeys)
    for field_name, pool in fk_ctx.items():
        if field_name in grain_key_set:
            log.debug(
                "[MockDataAgent] {}: skipping FK redistribution for '{}' "
                "(grain key — must preserve composite uniqueness)".format(cid, field_name)
            )
            continue
        if not pool or not rows:
            continue
        weights  = [random.paretovariate(1.5) for _ in pool]
        new_vals = random.choices(pool, weights=weights, k=len(rows))
        random.shuffle(new_vals)
        for row, val in zip(rows, new_vals):
            row[field_name] = val

    log.info("[MockDataAgent] {}: {} rows generated successfully".format(cid, len(rows)))
    return rows, []

def _write_csv(vertical: str, canonical_id: str, rows: List[Dict]) -> Path:
    """Write rows to data/mock_data/{vertical}/{canonical_id}.csv"""
    vertical_dir = _paths.MOCK_DATA_DIR / vertical
    vertical_dir.mkdir(parents=True, exist_ok=True)
    csv_path = vertical_dir / f"{canonical_id}.csv"

    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return csv_path

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def _sha256_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _update_fingerprints(
    vertical:  str,
    csv_paths: Dict[str, Path],
    row_counts: Dict[str, int],
) -> None:
    """
    Update data/data_fingerprints.json with a SHA-256 hash per generated CSV.

    Only overwrites entries for the canonical_ids generated in this run —
    other verticals'/atoms' existing entries are preserved untouched.

    This file has no consumer yet (compilation/Step 7 doesn't exist). It is
    written now, while the CSV paths are already in hand, so that whenever
    a cache-invalidation layer is built later it has fingerprint history
    to compare against rather than only future-dated hashes.
    """
    fingerprints: Dict[str, Any] = {}
    if _paths.DATA_FINGERPRINTS_PATH.exists():
        try:
            fingerprints = json.loads(_paths.DATA_FINGERPRINTS_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning(f"[MockDataAgent] Could not read existing fingerprints file, "
                        f"starting fresh: {e}")
            fingerprints = {}

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for cid, path in csv_paths.items():
        fingerprints[cid] = {
            "sha256":       _sha256_file(path),
            "vertical":     vertical,
            "row_count":    row_counts.get(cid, 0),
            "generated_at": now,
        }

    _paths.DATA_FINGERPRINTS_PATH.write_text(
        json.dumps(fingerprints, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info(f"[MockDataAgent] data_fingerprints.json updated "
             f"({len(csv_paths)} atom(s) hashed)")


# ── Generate one atom ──────────────────────────────────────────────────────────




def run(
    vertical:      str,
    mode:          str = "GENERIC",
    only_atom_ids: Optional[Set[str]] = None,
    mock_row_count: Optional[int] = None,
) -> Dict:
    """
    Run MockDataAgent for a vertical.

    Args:
        vertical:       e.g. "supply_chain", "hr", "finance"
        mode:           "GENERIC" or "CUSTOMER"
        only_atom_ids:  If given, only these atoms get fresh (LLM-generated)
                        mock data. Every other atom in the vertical keeps its
                        existing CSV as-is — its rows are read back off disk
                        purely to populate FK pools for atoms that DO
                        regenerate and depend on it. If None (default),
                        every atom regenerates — unchanged from before.
        mock_row_count: Customer-chosen row count for record/event/snapshot/state
                        tables. When provided, overrides the env-var defaults
                        (_ROW_COUNTS) for this run only. Dimension tables are
                        unaffected — they always use their fixed keyword counts.
                        Clamped to [10, 500].

    Returns:
        {status, vertical, mode, atoms_processed, total_rows,
         csvs_created, fingerprints_path, skipped, errors}
    """
    # Apply customer-chosen row count for this run (does not mutate module-level defaults)
    effective_row_counts = dict(_ROW_COUNTS)
    if mock_row_count is not None:
        clamped = max(10, min(500, int(mock_row_count)))
        effective_row_counts = {k: clamped for k in effective_row_counts}
        log.info(f"[MockDataAgent] mock_row_count override → {clamped} rows "
                 f"(record/event/snapshot/state)")

    log.info(f"[MockDataAgent] Starting — vertical={vertical} mode={mode} "
             f"scoped={'no' if only_atom_ids is None else len(only_atom_ids)}")

    # Load atoms for this vertical
    all_atoms = atom_registry.get_all_atoms()
    atoms     = [a for a in all_atoms if a.get("domain") == vertical]

    if not atoms:
        return {
            "status":  "error",
            "error":   f"No atoms found for '{vertical}'. Run VerticalSchemaAgent first.",
            "vertical": vertical,
        }

    # Load all FK relationships for this vertical
    all_rels = list_relationships()
    atom_ids = {a["canonical_id"] for a in atoms}
    rels     = [r for r in all_rels
                if r.get("from_atom") in atom_ids and r.get("to_atom") in atom_ids]

    # Topological sort — dimensions/parents first
    sorted_atoms = _topological_sort(atoms, rels)

    log.info(f"[MockDataAgent] {len(atoms)} atoms, generation order: "
             f"{[a['canonical_id'] for a in sorted_atoms]}")

    # Generate
    fk_pools:        Dict[str, List[str]] = {}
    csvs_created     = []
    csv_paths_by_cid: Dict[str, Path] = {}
    row_counts_by_cid: Dict[str, int] = {}
    total_rows_count = 0
    skipped          = []
    reused           = []
    all_errors       = []
    atoms_processed  = 0

    for atom in sorted_atoms:
        cid    = atom["canonical_id"]
        fields = atom.get("fields", [])
        gkeys  = atom.get("grain_keys", [])

        pk_field = next(
            (f["name"] for f in fields if f.get("role") == "primary_key"), None
        )
        if not pk_field and gkeys:
            pk_field = gkeys[0]

        if not pk_field:
            skipped.append(f"{cid} (no primary_key or grain_key)")
            log.warning(f"[MockDataAgent] Skipping {cid} — no PK found")
            continue

        # ── Reuse path: CSV already exists on disk ───────────────────────────
        # For full runs (only_atom_ids is None): reuse existing CSV so we
        # don't regenerate atoms that already succeeded (resume behaviour).
        # For scoped runs (only_atom_ids set): reuse only atoms outside scope.
        existing_csv = _paths.MOCK_DATA_DIR / vertical / f"{cid}.csv"
        should_reuse = (
            existing_csv.exists() and (
                only_atom_ids is None               # full run — reuse all existing
                or cid not in only_atom_ids         # scoped run — reuse out-of-scope
            )
        )
        if should_reuse:
            try:
                with open(existing_csv, newline="", encoding="utf-8") as f:
                    existing_rows = list(csv.DictReader(f))
                if existing_rows:
                    pk_values = [str(r[pk_field]) for r in existing_rows if r.get(pk_field)]
                    fk_pools[f"{cid}.{pk_field}"] = pk_values
                    csv_paths_by_cid[cid]  = existing_csv
                    row_counts_by_cid[cid] = len(existing_rows)
                    total_rows_count      += len(existing_rows)
                    atoms_processed       += 1
                    reused.append(cid)
                    log.info(f"[MockDataAgent] Reused existing CSV for {cid} "
                             f"({len(existing_rows)} rows, untouched)")
                    continue
            except Exception as e:
                log.warning(f"[MockDataAgent] Could not reuse existing CSV for "
                            f"{cid}, regenerating instead: {e}")
                # falls through to normal generation below

        # Pause between atoms — gives API breathing room between calls
        time.sleep(3)

        # Generate rows
        rows, errors = _generate_atom(atom, vertical, fk_pools, effective_row_counts)
        all_errors.extend(errors)

        if not rows:
            skipped.append(f"{cid} (no rows generated)")
            log.warning(f"[MockDataAgent] No rows for {cid} — errors: {errors}")
            continue

        # Write CSV
        try:
            csv_path = _write_csv(vertical, cid, rows)
            csvs_created.append(str(csv_path))
            csv_paths_by_cid[cid]  = csv_path
            row_counts_by_cid[cid] = len(rows)
            log.info(f"[MockDataAgent] CSV: {csv_path.name} ({len(rows)} rows)")
        except Exception as e:
            all_errors.append(f"{cid}: CSV write failed — {e}")
            log.error(f"[MockDataAgent] CSV write failed for {cid}: {e}")
            continue

        # Populate FK pool for child atoms
        if pk_field:
            pk_values = [str(r[pk_field]) for r in rows if r.get(pk_field)]
            pool_key  = f"{cid}.{pk_field}"
            fk_pools[pool_key] = pk_values
            log.info(f"[MockDataAgent] FK pool: {pool_key} → {len(pk_values)} IDs")

        total_rows_count += len(rows)
        atoms_processed  += 1

    # Update data_fingerprints.json for every CSV written this run
    try:
        _update_fingerprints(vertical, csv_paths_by_cid, row_counts_by_cid)
    except Exception as e:
        all_errors.append(f"data_fingerprints update failed: {e}")
        log.error(f"[MockDataAgent] data_fingerprints update failed: {e}")

    # Run discover_enums after all CSVs written
    try:
        from services.discover_enums import discover_enums
        discover_enums(vertical=vertical)
        log.info("[MockDataAgent] discover_enums completed → field_values.json updated")
    except Exception as e:
        all_errors.append(f"discover_enums failed: {e}")
        log.error(f"[MockDataAgent] discover_enums failed: {e}")

    status = "success" if atoms_processed > 0 else "error"

    # ── Completeness check ────────────────────────────────────────────────────
    all_atom_ids   = {a["canonical_id"] for a in atoms}
    covered_ids    = set(reused) | {Path(p).stem for p in csvs_created}
    missing_ids    = all_atom_ids - covered_ids
    missing_atoms  = [
        {
            "canonical_id": cid,
            "record_type":  next(
                (a.get("record_type", "unknown") for a in atoms if a["canonical_id"] == cid),
                "unknown"
            ),
        }
        for cid in sorted(missing_ids)
    ]
    incomplete = len(missing_atoms) > 0

    if incomplete:
        status = "incomplete"
        log.warning(
            f"[MockDataAgent] INCOMPLETE — {len(missing_atoms)} of {len(atoms)} atoms "
            f"have no CSV: {[m['canonical_id'] for m in missing_atoms]}"
        )

    log.info(
        f"[MockDataAgent] Done — atoms={atoms_processed} "
        f"({len(reused)} reused, {atoms_processed - len(reused)} regenerated), "
        f"rows={total_rows_count}, csvs={len(csvs_created)}, errors={len(all_errors)}, "
        f"missing={len(missing_atoms)}"
    )

    return {
        "status":            status,
        "vertical":          vertical,
        "reused_atoms":      reused,
        "mode":              mode,
        "atoms_total":       len(atoms),
        "atoms_processed":   atoms_processed,
        "atoms_missing":     len(missing_atoms),
        "missing_atoms":     missing_atoms,
        "incomplete":        incomplete,
        "total_rows":        total_rows_count,
        "csvs_created":      csvs_created,
        "fingerprints_path": str(_paths.DATA_FINGERPRINTS_PATH),
        "skipped":           skipped,
        "errors":            all_errors,
    }


def run_single_atom(vertical: str, canonical_id: str) -> Dict:
    """
    Regenerate mock data for one specific atom only.
    Deletes the existing CSV if present and regenerates from scratch.
    Used by the Debug UI atom dropdown to regen a single atom's CSV.
    """
    import json as _j

    # Load atom definition
    raw   = _j.loads(_paths.ATOMS_PATH.read_text(encoding="utf-8"))
    atoms = list((raw.get("_default", raw) if "_default" in raw else raw).values())
    atom  = next((a for a in atoms if a.get("canonical_id") == canonical_id), None)

    if atom is None:
        return {"status": "error", "error": f"Atom '{canonical_id}' not found"}

    # Build FK pools from existing CSVs (same as full run)
    # Key format must match _generate_atom lookup: "{atom_cid}.{pk_field}"
    csv_dir   = _paths.MOCK_DATA_DIR / vertical
    fk_pools  = {}
    all_atoms = {a["canonical_id"]: a for a in atoms if a.get("domain") == vertical}
    for other_cid, other_atom in all_atoms.items():
        csv_path = csv_dir / f"{other_cid}.csv"
        if csv_path.exists() and other_cid != canonical_id:
            try:
                import csv as _csv
                with open(csv_path, encoding="utf-8") as f:
                    rows = list(_csv.DictReader(f))
                for field in other_atom.get("fields", []):
                    if field.get("role") == "primary_key":
                        pk_field = field["name"]
                        vals = [r[pk_field] for r in rows if pk_field in r]
                        if vals:
                            # Use the same key format as the full run:
                            # "{canonical_id}.{pk_field}"
                            pool_key = f"{other_cid}.{pk_field}"
                            fk_pools[pool_key] = vals
                            log.debug(
                                f"[MockDataAgent/single] FK pool: {pool_key} → {len(vals)} IDs"
                            )
            except Exception:
                pass

    # Delete existing CSV
    csv_path = csv_dir / f"{canonical_id}.csv"
    if csv_path.exists():
        csv_path.unlink()
        log.info(f"[MockDataAgent/single] Deleted existing CSV: {csv_path.name}")

    # Regenerate
    rows, errors = _generate_atom(atom, vertical, fk_pools)

    if errors:
        return {"status": "error", "canonical_id": canonical_id, "errors": errors}

    if rows:
        _write_csv(vertical, canonical_id, rows)
        log.info(f"[MockDataAgent/single] Regenerated {canonical_id}: {len(rows)} rows")
        return {
            "status":       "ok",
            "canonical_id": canonical_id,
            "rows":         len(rows),
        }
    else:
        return {"status": "error", "canonical_id": canonical_id, "error": "No rows generated"}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run MockDataAgent")
    parser.add_argument("vertical", help="e.g. supply_chain")
    parser.add_argument("--mode", default="GENERIC", choices=["GENERIC", "CUSTOMER"])
    args = parser.parse_args()

    result = run(vertical=args.vertical, mode=args.mode)
    print(json.dumps(result, indent=2, default=str))