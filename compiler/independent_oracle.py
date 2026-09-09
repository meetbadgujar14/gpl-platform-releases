"""
compiler/independent_oracle.py
================================
Phase 9.5 — Independent Oracle Verifier.

A completely separate LLM that has never seen the GPL formula is given:
  - The goal text (what business question we're answering)
  - The atom canonical_id and record_type
  - The full CSV content for that atom (all rows, not a sample)
  - Any FK-joined CSVs if the formula references multiple tables

It writes its own pandas code from scratch to answer the same question.
We execute that code in a restricted namespace and compare the result
to the stored oracle within a 2% drift threshold.

Two independent systems reaching the same answer via different paths
provides a fundamentally different level of confidence than any amount
of re-running the same GPL formula.

Verdicts:
  MATCH        — pct_diff ≤ 2%    → strongest confirmation
  NEAR_MATCH   — pct_diff ≤ 10%   → likely correct, minor discrepancy
  WARNING      — pct_diff ≤ 30%   → meaningful discrepancy, review
  MISMATCH     — pct_diff > 30%   → the two systems disagree significantly
  ZERO_MATCH   — both returned 0  → correctly empty
  SKIP         — cannot verify (composed formula, no CSV, etc.)
  ERROR        — execution failed

Runs only on:
  - Wizard-compiled aterms (lock_method = llm_wizard)
  - Domain wave goals (wave in A, B, C, D, E)

Cost: ~$0.005 per aterm (1 LLM call)
"""

import json
import logging
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from anthropic import Anthropic
from core.config import settings
import core.paths as _paths

log = logging.getLogger(__name__)

# ── Thresholds ─────────────────────────────────────────────────────────────────
_MATCH_THRESHOLD      = 0.02   # ≤ 2%  → MATCH
_NEAR_MATCH_THRESHOLD = 0.10   # ≤ 10% → NEAR_MATCH
_WARNING_THRESHOLD    = 0.30   # ≤ 30% → WARNING
                               # > 30% → MISMATCH

# ── Waves that qualify for independent verification ────────────────────────────
_QUALIFYING_WAVES    = {"A", "B", "C", "D", "E", "wave_A", "wave_B",
                        "wave_C", "wave_D", "wave_E"}
_QUALIFYING_METHODS  = {"llm_wizard"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _client() -> Anthropic:
    return Anthropic(api_key=settings.ANTHROPIC_API_KEY)


def _model() -> str:
    return settings.ANTHROPIC_MODEL


def _should_run(aterm: Dict) -> Tuple[bool, str]:
    """
    Decide whether this aterm qualifies for independent verification.
    Returns (should_run, skip_reason).
    """
    wave        = str(aterm.get("wave", ""))
    lock_method = aterm.get("lock_method", "")
    formula     = aterm.get("formula_line", "")

    # Composed formulas (Branch C) — no CSV to run against
    if formula.startswith("RATIO:") or " / " in formula and "MEASURE" not in formula:
        return False, "SKIP_composed_formula"

    # Branch C composition formulas
    if aterm.get("operator") == "branch_c":
        return False, "SKIP_branch_c"

    # Qualifies if wizard-compiled OR domain wave
    if lock_method in _QUALIFYING_METHODS:
        return True, ""
    if wave in _QUALIFYING_WAVES:
        return True, ""

    return False, f"SKIP_not_qualifying (wave={wave} method={lock_method})"


def _read_csv_as_text(csv_path: Path) -> str:
    """Read a CSV file and return its full content as text."""
    if not csv_path.exists():
        return ""
    return csv_path.read_text(encoding="utf-8").strip()


def _find_atom_csvs(aterm: Dict, vertical: str) -> Dict[str, str]:
    """
    Find all CSV files referenced by this aterm.
    Returns {atom_canonical_id: csv_content_string}.
    """
    csv_dir = _paths.MOCK_DATA_DIR / vertical
    formula = aterm.get("formula_line", "")
    slots   = aterm.get("slots", {})
    cid     = aterm.get("canonical_id", "")

    # Extract atom canonical IDs from formula using regex
    # Pattern: looks for vertical_entityname_SOURCE_type patterns
    atom_pattern = re.compile(
        rf"({re.escape(vertical)}_[a-z][a-z0-9_]*(?:ERP|WMS|MANUAL|SHOPIFY|SYSTEM)"
        rf"_(?:record|dimension|state|snapshot|event))"
    )
    found_atoms = set(atom_pattern.findall(formula))

    # Also try to infer from slots
    entity = slots.get("entity", "")
    if entity:
        # Search csv_dir for matching files
        for f in csv_dir.glob("*.csv"):
            if entity in f.stem:
                # Extract atom cid from filename
                atom_cid = f.stem  # filename without .csv
                found_atoms.add(atom_cid)

    # Fallback: if nothing found, try to match by CID prefix
    if not found_atoms:
        # CID like logistics_drivers_total_trips_completed_by_state_count
        # → look for logistics_drivers_* CSV
        parts = cid.split("_")
        if len(parts) >= 2:
            prefix = f"{parts[0]}_{parts[1]}"
            for f in csv_dir.glob(f"{prefix}*.csv"):
                found_atoms.add(f.stem)

    result = {}
    for atom_cid in found_atoms:
        csv_path = csv_dir / f"{atom_cid}.csv"
        content  = _read_csv_as_text(csv_path)
        if content:
            result[atom_cid] = content

    return result


def _build_prompt(aterm: Dict, csv_paths: Dict[str, str]) -> str:
    """
    Build the prompt for the independent LLM.
    Passes file paths — not inline CSV data — so the LLM uses
    pd.read_csv(path) which avoids triple-quoted string literal errors.
    csv_paths: {atom_canonical_id: temp_file_path}
    """
    goal_text  = aterm.get("source_goal", "")
    cid        = aterm.get("canonical_id", "")
    slots      = aterm.get("slots", {})
    unit       = slots.get("unit", "count")
    time_scope = slots.get("time", "all_time")
    state      = slots.get("state", "all")

    # Build CSV file listing for the prompt
    csv_listing = ""
    for atom_cid, path in csv_paths.items():
        short = "_".join(atom_cid.split("_")[2:])
        csv_listing += f"\n  {atom_cid}  →  {path}"

    # Show column headers only (no data rows) to guide the LLM
    col_hints = ""
    for atom_cid, path in csv_paths.items():
        try:
            with open(path, encoding="utf-8") as f:
                header = f.readline().strip()
            col_hints += f"\n  {atom_cid} columns: {header}"
        except Exception:
            pass

    # Time scope hint
    time_hint = ""
    if time_scope not in ("all_time", "alltime", ""):
        time_hint = (
            f"\nNOTE: The goal is time-scoped to '{time_scope}'. "
            "The mock data may not have current-period rows — if no rows match the time filter, print(0)."
        )

    prompt = f"""You are an independent data analyst verifying a business metric.

BUSINESS QUESTION: {goal_text}
METRIC ID: {cid}
UNIT: {unit}
STATE FILTER: {state if state not in ('all', 'base', '') else 'none'}
{time_hint}

CSV FILES (already on disk — use pd.read_csv with the exact paths below):
{csv_listing}

COLUMN HEADERS:
{col_hints}

Write Python code using pandas to answer the business question.
Use pd.read_csv(r"<exact_path>") with the paths listed above.

CRITICAL RULES — read carefully:
1. Use ONLY the column names shown in COLUMN HEADERS above
2. Always print() a single numeric value as the LAST line — never print strings, IDs, or multiple values
3. For COUNT goals: use df['col'].nunique() or len(df) — NOT len(df[df['col']==max_val])
4. For SUM goals: use df['col'].sum() — quantity, weight, value, amount columns always need SUM not COUNT
5. For AVG goals: use df['col'].mean()
6. For MAX goals: use df['col'].max() — return the MAX VALUE itself, not a row ID or row count
7. For MIN goals: use df['col'].min() — return the MIN VALUE itself, not a row ID or row count
8. For "vs" or "compared to" goals: compute each metric separately — do NOT sum or combine them; return only the primary metric value stated in the goal
9. If state filter needed: filter on exact string values visible in the data
10. If no rows match: print(0)
11. Round floats to 6 decimal places: print(round(result, 6))

DEDUPLICATION RULE — critical for state/snapshot atoms:
If the CSV filename contains '_WMS_state' or '_snapshot', it is a state table where
each entity appears multiple times as its state changes over time.

The correct order depends on whether a time scope is present:

CASE 1 — NO time scope (all_time goals):
  Dedup FIRST across all rows, then apply any state filters, then aggregate:
    df = df.sort_values('<date_col>').groupby('<grain_key>').last().reset_index()
    df = df[df['state_col'] == 'value']  # state filter after dedup
    result = df['col'].agg(...)

CASE 2 — WITH time scope (last_month, ytd, this_month, this_quarter etc.):
  Filter to the time window FIRST, then dedup within that window, then aggregate:
    df['date_col'] = pd.to_datetime(df['date_col'])
    df = df[(df['date_col'] >= start) & (df['date_col'] <= end)]  # time filter FIRST
    df = df.sort_values('<date_col>').groupby('<grain_key>').last().reset_index()  # then dedup
    result = df['col'].agg(...)

WHY: If you dedup first then filter by time, you collapse each entity to its LATEST
row across ALL time. That latest row may be outside your time window, giving 0 results
for last_month or fewer results for ytd. Always filter time window first, then dedup
within that window to get the correct snapshot for that period.

DATE BOUNDARY RULE — always use period-based filtering:
When filtering by time scope, ALWAYS use pandas period-based filtering to avoid
time-component issues. NEVER use datetime.today().replace(day=1) as a boundary
since it carries the current time and excludes rows stored at midnight.

Use these EXACT patterns:
  df['date_col'] = pd.to_datetime(df['date_col'])
  today_period = pd.Timestamp.today().to_period('M')

  # For last_month:
  df = df[df['date_col'].dt.to_period('M') == today_period - 1]

  # For this_month:
  df = df[df['date_col'].dt.to_period('M') == today_period]

  # For ytd:
  df = df[df['date_col'].dt.year == pd.Timestamp.today().year]

  # For this_quarter:
  today = pd.Timestamp.today()
  df = df[(df['date_col'].dt.year == today.year) & (df['date_col'].dt.quarter == today.quarter)]

  # For last_quarter:
  today = pd.Timestamp.today()
  current_q = today.quarter
  current_y = today.year
  last_q = current_q - 1 if current_q > 1 else 4
  last_q_year = current_y if current_q > 1 else current_y - 1
  df = df[(df['date_col'].dt.year == last_q_year) & (df['date_col'].dt.quarter == last_q)]

The grain_key is the entity identifier column (e.g. inventory_id, order_id).
The date_col is the timestamp column (e.g. snapshot_date, updated_at).

Respond with ONLY the Python code. No explanation, no markdown backticks."""

    return prompt


def _execute_pandas_code(code: str, csv_contents: Dict[str, str]) -> Tuple[Optional[float], str]:
    """
    Write CSVs to temp files, then execute the LLM-generated pandas code.
    Returns (result_value, error_message).
    """
    import io
    import tempfile
    import os
    import pandas as pd

    # Write each CSV to a real temp file so pd.read_csv(path) works
    tmp_files = {}
    tmp_dir   = tempfile.mkdtemp(prefix="gpl_oracle_")
    try:
        for atom_cid, content in csv_contents.items():
            tmp_path = os.path.join(tmp_dir, f"{atom_cid}.csv")
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(content)
            tmp_files[atom_cid] = tmp_path

        # Clean code — strip markdown if LLM added it
        cleaned = re.sub(r"^```python\s*", "", code.strip())
        cleaned = re.sub(r"^```\s*",       "", cleaned.strip())
        cleaned = re.sub(r"\s*```$",        "", cleaned.strip())

        # Capture print output
        output_lines = []

        def capture_print(*args, **kwargs):
            output_lines.append(" ".join(str(a) for a in args))

        namespace = {
            "pd":    pd,
            "io":    io,
            "print": capture_print,
            "open":  open,
        }

        try:
            exec(cleaned, namespace)
            if not output_lines:
                return None, "No output produced"
            raw = output_lines[-1].strip()

            # Try direct float conversion first
            try:
                return float(raw), ""
            except ValueError:
                pass

            # Multi-line output (e.g. value_counts() result) — try summing the values
            # Format is typically: "label    count\nlabel2   count2\n..."
            lines = raw.split("\n")
            nums = []
            for line in lines:
                parts = line.strip().split()
                if parts:
                    try:
                        nums.append(float(parts[-1]))
                    except ValueError:
                        pass
            if nums:
                total = sum(nums)
                log.debug(
                    f"[phase9.5] Multi-line output detected — summed {len(nums)} values → {total}"
                )
                return total, ""

            return None, f"Could not convert output to float: {raw[:100]}"

        except Exception as e:
            return None, str(e)[:300]

    finally:
        # Clean up temp files
        import shutil
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


def _compare(oracle: float, independent: float) -> Tuple[str, float]:
    """
    Compare oracle vs independent result. Returns (verdict, pct_diff).
    """
    if oracle == 0 and independent == 0:
        return "ZERO_MATCH", 0.0

    if oracle == 0:
        # GPL says 0, independent says non-zero
        return "MISMATCH", 100.0

    pct_diff = abs(oracle - independent) / abs(oracle)

    if pct_diff <= _MATCH_THRESHOLD:
        return "MATCH", round(pct_diff * 100, 4)
    elif pct_diff <= _NEAR_MATCH_THRESHOLD:
        return "NEAR_MATCH", round(pct_diff * 100, 4)
    elif pct_diff <= _WARNING_THRESHOLD:
        return "WARNING", round(pct_diff * 100, 4)
    else:
        return "MISMATCH", round(pct_diff * 100, 4)


# ── Main entry point ───────────────────────────────────────────────────────────

def verify_independent(aterm: Dict, vertical: str) -> Dict:
    """
    Run Phase 9.5 independent oracle verification for one aterm.

    Returns a result dict:
    {
        canonical_id:         str,
        verdict:              MATCH | NEAR_MATCH | WARNING | MISMATCH | ZERO_MATCH | SKIP | ERROR,
        gpl_oracle:           float,
        independent_oracle:   float | None,
        pct_diff:             float | None,
        independent_code:     str,    # the pandas code the LLM generated
        reasoning:            str,    # one line explanation
        csv_tables_used:      [str],  # which CSVs were loaded
    }
    """
    cid        = aterm.get("canonical_id", "")
    oracle_raw = aterm.get("oracle_value")

    base = {
        "canonical_id":       cid,
        "verdict":            "SKIP",
        "gpl_oracle":         oracle_raw,
        "independent_oracle": None,
        "pct_diff":           None,
        "independent_code":   "",
        "reasoning":          "",
        "csv_tables_used":    [],
    }

    # ── Eligibility check ──────────────────────────────────────────────────────
    should_run, skip_reason = _should_run(aterm)
    if not should_run:
        base["reasoning"] = skip_reason
        return base

    # ── Oracle must be numeric ─────────────────────────────────────────────────
    if oracle_raw is None or isinstance(oracle_raw, dict):
        base["reasoning"] = "SKIP_grouped_or_null_oracle"
        return base

    oracle = float(oracle_raw)

    # ── Load CSVs ──────────────────────────────────────────────────────────────
    csv_contents = _find_atom_csvs(aterm, vertical)
    if not csv_contents:
        base["verdict"]   = "SKIP"
        base["reasoning"] = "SKIP_no_csv_found"
        return base

    base["csv_tables_used"] = list(csv_contents.keys())

    # ── Write CSVs to temp files so LLM can use pd.read_csv(path) ─────────────
    import tempfile, os, shutil
    tmp_dir   = tempfile.mkdtemp(prefix="gpl_oracle_")
    csv_paths = {}
    for atom_cid, content in csv_contents.items():
        tmp_path = os.path.join(tmp_dir, f"{atom_cid}.csv")
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
        csv_paths[atom_cid] = tmp_path

    # ── LLM call — generate independent pandas code ────────────────────────────
    prompt = _build_prompt(aterm, csv_paths)

    try:
        response = _client().messages.create(
            model=_model(),
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        code = response.content[0].text.strip()
    except Exception as e:
        base["verdict"]   = "ERROR"
        base["reasoning"] = f"LLM call failed: {str(e)[:100]}"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return base

    base["independent_code"] = code

    # ── Execute the generated code ─────────────────────────────────────────────
    independent_val, exec_error = _execute_pandas_code(code, csv_contents)

    # Cleanup temp dir
    shutil.rmtree(tmp_dir, ignore_errors=True)

    if independent_val is None:
        base["verdict"]   = "ERROR"
        base["reasoning"] = f"Code execution failed: {exec_error}"
        return base

    base["independent_oracle"] = round(independent_val, 6)

    # ── Compare ────────────────────────────────────────────────────────────────
    verdict, pct_diff = _compare(oracle, independent_val)
    base["verdict"]  = verdict
    base["pct_diff"] = pct_diff

    # ── Reasoning line ─────────────────────────────────────────────────────────
    if verdict == "MATCH":
        base["reasoning"] = f"GPL ({oracle}) == Independent ({independent_val}) within {pct_diff}%"
    elif verdict == "ZERO_MATCH":
        base["reasoning"] = "Both GPL and independent returned 0 — no matching rows in mock data"
    elif verdict == "NEAR_MATCH":
        base["reasoning"] = f"GPL ({oracle}) ≈ Independent ({independent_val}), {pct_diff}% drift — likely correct"
    elif verdict == "WARNING":
        base["reasoning"] = f"GPL ({oracle}) vs Independent ({independent_val}), {pct_diff}% drift — review recommended"
    elif verdict == "MISMATCH":
        base["reasoning"] = f"GPL ({oracle}) vs Independent ({independent_val}), {pct_diff}% drift — DISAGREEMENT"

    log.info(
        f"[phase9.5] {cid}: {verdict} "
        f"gpl={oracle} independent={independent_val} diff={pct_diff}%"
    )

    return base


def run_deep_verify(vertical: str) -> Dict:
    """
    Run Phase 9.5 on all qualifying aterms for a vertical.
    Saves results to data/independent_oracle_{vertical}.json.
    Returns summary dict for the UI.
    """
    from datetime import datetime, timezone

    log.info(f"[phase9.5] Starting deep verification for vertical='{vertical}'")

    # Load all aterms for this vertical
    aterms_dir = _paths.ATERMS_DIR
    results    = []
    errors     = []

    aterm_files = sorted(aterms_dir.glob(f"aterm_{vertical}_*.json"))
    log.info(f"[phase9.5] Found {len(aterm_files)} aterm files for {vertical}")

    for aterm_path in aterm_files:
        try:
            aterm = json.loads(aterm_path.read_text(encoding="utf-8"))
        except Exception as e:
            errors.append({"file": aterm_path.name, "error": str(e)})
            continue

        should_run, skip_reason = _should_run(aterm)
        if not should_run:
            continue

        result = verify_independent(aterm, vertical)
        results.append(result)

        # Write independent_verdict back to the aterm file
        try:
            aterm["independent_verdict"] = result["verdict"]
            aterm["independent_oracle"]  = result["independent_oracle"]
            aterm["independent_pct_diff"]= result["pct_diff"]
            aterm["independent_code"]    = result["independent_code"]
            aterm["independent_run_at"]  = datetime.now(timezone.utc).isoformat()
            aterm_path.write_text(
                json.dumps(aterm, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
        except Exception as e:
            log.warning(f"[phase9.5] Could not write back to {aterm_path.name}: {e}")

    # ── Tally results ──────────────────────────────────────────────────────────
    tally = {
        "MATCH":      0,
        "NEAR_MATCH": 0,
        "ZERO_MATCH": 0,
        "WARNING":    0,
        "MISMATCH":   0,
        "SKIP":       0,
        "ERROR":      0,
    }
    for r in results:
        v = r.get("verdict", "SKIP")
        tally[v] = tally.get(v, 0) + 1

    mismatches = [r for r in results if r["verdict"] == "MISMATCH"]
    warnings   = [r for r in results if r["verdict"] == "WARNING"]

    summary = {
        "vertical":       vertical,
        "total_checked":  len(results),
        "tally":          tally,
        "mismatches":     mismatches,
        "warnings":       warnings,
        "errors":         errors,
        "run_at":         datetime.now(timezone.utc).isoformat(),
        "results":        results,
    }

    # Persist to disk
    out_path = _paths.DATA_DIR / f"independent_oracle_{vertical}.json"
    out_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    log.info(
        f"[phase9.5] Done: {tally['MATCH']} MATCH, {tally['NEAR_MATCH']} NEAR_MATCH, "
        f"{tally['MISMATCH']} MISMATCH, {tally['WARNING']} WARNING, "
        f"{tally['SKIP']} SKIP out of {len(results)} checked"
    )

    return summary


# ── AI Analysis — "Get AI Help" per mismatch ──────────────────────────────────

def analyse_mismatch(result: Dict, vertical: str) -> Dict:
    """
    Phase 9.5 AI Analysis — called when a human clicks "Get AI Analysis"
    on a MISMATCH or WARNING in the Oracle Review tab.

    Sends to a fresh LLM:
      - The goal text
      - The GPL formula
      - The GPL oracle value
      - The independent pandas code
      - The independent result
      - The drift %
      - All mock data CSVs referenced by the formula

    Returns:
      {
        verdict:     LIKELY_FALSE_POSITIVE | GENUINE_CONCERN | INCONCLUSIVE,
        explanation: str,   # plain English explanation of what happened
        recommendation: str # what to do about it
      }
    """
    cid              = result.get("canonical_id", "")
    gpl_oracle       = result.get("gpl_oracle")
    ind_oracle       = result.get("independent_oracle")
    pct_diff         = result.get("pct_diff")
    ind_code         = result.get("independent_code", "")
    csv_tables_used  = result.get("csv_tables_used", [])

    # Load aterm file for goal text and formula
    aterm_path = _paths.ATERMS_DIR / f"aterm_{cid}.json"
    if not aterm_path.exists():
        return {
            "verdict":        "INCONCLUSIVE",
            "explanation":    "Aterm file not found.",
            "recommendation": "Manually inspect the aterm.",
        }

    aterm       = json.loads(aterm_path.read_text(encoding="utf-8"))
    goal_text   = aterm.get("source_goal", "")
    formula     = aterm.get("formula_line", "")
    slots       = aterm.get("slots", {})

    # Load mock data CSVs
    csv_dir  = _paths.MOCK_DATA_DIR / vertical
    csv_data = ""
    for atom_cid in csv_tables_used:
        csv_path = csv_dir / f"{atom_cid}.csv"
        if csv_path.exists():
            content = csv_path.read_text(encoding="utf-8").strip()
            short   = "_".join(atom_cid.split("_")[2:])
            csv_data += f"\n### {short}\n{content}\n"

    prompt = f"""You are a senior data analyst reviewing a discrepancy between two independently computed results for the same business metric.

BUSINESS QUESTION: {goal_text}
SLOTS: {json.dumps(slots)}

GPL FORMULA (what our compiler produced):
{formula}
GPL RESULT: {gpl_oracle}

INDEPENDENT CODE (what a separate LLM produced from scratch):
{ind_code}
INDEPENDENT RESULT: {ind_oracle}

DRIFT: {pct_diff}%

MOCK DATA USED:
{csv_data if csv_data else "(not available)"}

Analyse this discrepancy carefully. Consider:
1. Did the independent LLM correctly interpret the business question?
2. Did the GPL formula correctly answer the business question?
3. Are they computing different but valid things, or is one of them wrong?
4. What does the mock data tell us about which result is correct?

You MUST respond with a JSON object with exactly these three fields:
{{
  "verdict": "LIKELY_FALSE_POSITIVE" | "GENUINE_CONCERN" | "INCONCLUSIVE",
  "explanation": "One clear paragraph explaining what each system computed and why they differ",
  "recommendation": "One clear sentence on what action to take"
}}

VERDICT DEFINITIONS:
- LIKELY_FALSE_POSITIVE: The independent LLM misinterpreted the goal or used wrong logic. GPL formula is correct.
- GENUINE_CONCERN: The independent LLM correctly interpreted the goal and GPL formula appears to compute something different. Review recommended.
- INCONCLUSIVE: Cannot determine which is correct from available information.

Respond with ONLY the JSON object. No markdown, no preamble."""

    try:
        response = _client().messages.create(
            model=_model(),
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"^```\s*",     "", raw)
        raw = re.sub(r"\s*```$",      "", raw)
        parsed = json.loads(raw)
        log.info(
            f"[phase9.5/analysis] {cid}: {parsed.get('verdict')}"
        )
        return parsed
    except Exception as e:
        return {
            "verdict":        "INCONCLUSIVE",
            "explanation":    f"AI analysis failed: {str(e)[:100]}",
            "recommendation": "Manually inspect the formula and independent code.",
        }
