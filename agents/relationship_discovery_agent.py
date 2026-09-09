"""
agents/relationship_discovery_agent.py
=======================================
GPL Relationship Discovery Agent — Phase 1 (no CSV data needed)

Runs 5 of 7 discovery algorithms using only atom definitions (atoms.json,
atom_relationships.json, field_values.json). Pure Python — zero AI calls.

Algorithms implemented here:
  1. Column name matching     — find likely FKs from field name patterns
  4. Functional dependency    — find denormalized fields (A.name depends on A.id)
  5. Hidden dimension         — find low-cardinality fields that should be atoms
  6. Grain verification       — validate grain_keys are PKs and exist in fields
  7. Temporal pattern         — flag atoms with no date field (blocks time goals)

Phase 2 (after mock data CSVs exist):
  2. Value inclusion analysis — confirm FKs via set intersection on CSV data
  3. Cardinality analysis     — verify field roles match actual unique value counts

Output:
  {
    status, vertical,
    confirmed_relationships: [...],   # existing rels that look correct
    new_relationships:       [...],   # discovered FKs not yet declared
    functional_dependencies: [...],   # denormalized fields
    hidden_dimensions:       [...],   # low-cardinality fields → own atom
    grain_issues:            [...],   # broken grain_key declarations
    temporal_gaps:           [...],   # atoms with no date field
    errors:                  [...],   # structural errors (broken rels)
    stats:                   {...}
  }
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import core.paths as _paths


# ── Constants ──────────────────────────────────────────────────────────────────

# Suffixes that strongly suggest a FK field
_FK_SUFFIXES = ("_id", "_key", "_code", "_ref", "_fk")

# Field name patterns that suggest dates/times
_DATE_PATTERNS = (
    "date", "time", "at", "on", "timestamp", "created", "updated",
    "start", "end", "due", "scheduled", "posted", "closed", "opened",
)

# Roles considered "low cardinality" (good hidden dimension candidates)
_LOW_CARD_ROLES = ("dimension", "flag", "status", "category")

# Minimum number of enum values to consider a field a hidden dimension candidate
_MIN_ENUM_VALUES = 2
_MAX_ENUM_VALUES = 50  # above this it's probably not a useful grouping dimension


# ── Entry point ────────────────────────────────────────────────────────────────

def run(vertical: str) -> Dict:
    """
    Run all Phase 1 discovery algorithms for a vertical.
    Returns structured discovery report.
    """
    # Load atoms
    if not _paths.ATOMS_PATH.exists():
        return _err("atoms.json not found. Run VerticalSchemaAgent first.")

    raw   = json.loads(_paths.ATOMS_PATH.read_text(encoding="utf-8"))
    atoms = [a for a in raw.get("_default", {}).values()
             if a.get("domain") == vertical]
    if not atoms:
        return _err(f"No atoms found for vertical '{vertical}'.")

    atom_map:    Dict[str, Dict] = {a["canonical_id"]: a for a in atoms}
    atom_ids:    Set[str]        = set(atom_map.keys())
    atom_fields: Dict[str, Dict[str, Dict]] = {
        cid: {f["name"]: f for f in a.get("fields", [])}
        for cid, a in atom_map.items()
    }

    # Load declared relationships
    rels_path = _paths.PROJECT_ROOT / "data" / "atom_relationships.json"
    declared_rels: List[Dict] = []
    if rels_path.exists():
        all_rels = json.loads(rels_path.read_text(encoding="utf-8"))
        declared_rels = [r for r in all_rels
                         if r.get("from_atom") in atom_ids
                         or r.get("to_atom")   in atom_ids]

    declared_fk_set: Set[Tuple] = {
        (r["from_atom"], r["from_field"]) for r in declared_rels
    }

    # Load field_values (enums) if available
    fv_path = _paths.DATA_DIR / "field_values.json"
    field_values: Dict = {}
    if fv_path.exists():
        try:
            field_values = json.loads(fv_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # ── Run algorithms ─────────────────────────────────────────────────────────
    errors                = _algo_structural_errors(declared_rels, atom_ids, atom_fields)
    new_relationships     = _algo1_column_name_matching(atoms, atom_map, atom_fields, declared_fk_set)
    functional_deps       = _algo4_functional_dependency(atoms, atom_fields)
    hidden_dimensions     = _algo5_hidden_dimensions(atoms, atom_fields, field_values)
    grain_issues          = _algo6_grain_verification(atoms, atom_fields)
    temporal_gaps         = _algo7_temporal_patterns(atoms, atom_fields)

    # Confirmed relationships = declared rels with no structural errors
    error_keys = {e.get("from_atom","") + e.get("from_field","") for e in errors}
    confirmed  = [r for r in declared_rels
                  if (r["from_atom"] + r.get("from_field","")) not in error_keys]

    status = "errors" if errors else (
        "warnings" if (new_relationships or grain_issues or temporal_gaps) else "ok"
    )

    return {
        "status":                 status,
        "vertical":               vertical,
        "phase":                  1,
        "confirmed_relationships": confirmed,
        "new_relationships":      new_relationships,
        "functional_dependencies": functional_deps,
        "hidden_dimensions":      hidden_dimensions,
        "grain_issues":           grain_issues,
        "temporal_gaps":          temporal_gaps,
        "errors":                 errors,
        "stats": {
            "atoms":                    len(atoms),
            "declared_relationships":   len(declared_rels),
            "confirmed_relationships":  len(confirmed),
            "new_relationships_found":  len(new_relationships),
            "functional_dependencies":  len(functional_deps),
            "hidden_dimensions":        len(hidden_dimensions),
            "grain_issues":             len(grain_issues),
            "temporal_gaps":            len(temporal_gaps),
            "errors":                   len(errors),
        }
    }


# ── Structural error check (pre-flight) ────────────────────────────────────────

def _algo_structural_errors(
    rels: List[Dict],
    atom_ids: Set[str],
    atom_fields: Dict[str, Dict[str, Dict]],
) -> List[Dict]:
    """
    Check declared relationships for structural errors.
    - from_atom / to_atom must exist
    - from_field / to_field must exist in their atoms
    """
    errors = []
    seen   = set()

    for r in rels:
        fa, ff = r.get("from_atom",""), r.get("from_field","")
        ta, tf = r.get("to_atom",""),   r.get("to_field","")

        if fa not in atom_ids:
            errors.append({"type":"ATOM_NOT_FOUND","from_atom":fa,"message":f"from_atom '{fa}' not found"})
            continue
        if ta not in atom_ids:
            errors.append({"type":"ATOM_NOT_FOUND","to_atom":ta,"message":f"to_atom '{ta}' not found"})
            continue
        if ff not in atom_fields[fa]:
            errors.append({
                "type":       "FIELD_NOT_FOUND",
                "from_atom":  fa,
                "from_field": ff,
                "message":    f"Field '{ff}' not found in '{_s(fa)}'. "
                              f"Available: {sorted(atom_fields[fa].keys())}",
                "fix":        f"Rename from_field to match actual field name in atom"
            })
        if tf not in atom_fields[ta]:
            errors.append({
                "type":      "FIELD_NOT_FOUND",
                "to_atom":   ta,
                "to_field":  tf,
                "message":   f"Field '{tf}' not found in '{_s(ta)}'. "
                             f"Available: {sorted(atom_fields[ta].keys())}",
                "fix":       f"Rename to_field to match actual field name in atom"
            })
        key = (fa, ff, ta, tf)
        if key in seen:
            errors.append({"type":"DUPLICATE","message":f"Duplicate: {_s(fa)}.{ff} → {_s(ta)}.{tf}"})
        seen.add(key)

        if fa == ta:
            errors.append({"type":"SELF_REF","atom":fa,"message":f"Self-reference on {_s(fa)}"})

    return errors


# ── Algorithm 1: Column name matching ─────────────────────────────────────────

def _algo1_column_name_matching(
    atoms:          List[Dict],
    atom_map:       Dict[str, Dict],
    atom_fields:    Dict[str, Dict[str, Dict]],
    declared_fk_set: Set[Tuple],
) -> List[Dict]:
    """
    Find fields that look like FKs (end in _id/_key/_ref) but have no
    relationship declared. Try to match them to a likely target atom by
    stripping the suffix and comparing to atom names.

    e.g. purchase_orders.supplier_id → suppliers_ERP_dimension (name match)
    """
    found = []

    # Build a map: entity_stem → atom canonical_id
    # e.g. "supplier" → "supply_chain_suppliers_ERP_dimension"
    stem_to_atom: Dict[str, str] = {}
    for cid, a in atom_map.items():
        # Extract meaningful name parts from canonical_id
        # supply_chain_suppliers_ERP_dimension → ["suppliers", "supplier"]
        parts = cid.lower().split("_")
        for part in parts:
            if len(part) > 3 and part not in ("erp","wms","crm","dim","rec","snap"):
                stem_to_atom[part]            = cid
                stem_to_atom[part.rstrip("s")] = cid  # singular

    for a in atoms:
        cid    = a["canonical_id"]
        fields = atom_fields[cid]

        for fname, fdef in fields.items():
            role = fdef.get("role","")

            # Skip PKs and already-declared FKs
            if role == "primary_key":
                continue
            if (cid, fname) in declared_fk_set:
                continue

            # Check if field name ends in a FK suffix
            matched_suffix = next((s for s in _FK_SUFFIXES if fname.endswith(s)), None)
            if not matched_suffix:
                continue

            # Strip suffix to get stem: supplier_id → supplier
            stem = fname[: -len(matched_suffix)].lower().rstrip("_")

            # Try to find a target atom
            target_cid = stem_to_atom.get(stem) or stem_to_atom.get(stem.rstrip("s"))

            # Don't suggest self-reference
            if target_cid == cid:
                continue

            # Find the PK field in target atom (if we found one)
            target_pk = None
            if target_cid:
                target_fields = atom_fields[target_cid]
                target_pk = next(
                    (fn for fn, fd in target_fields.items() if fd.get("role") == "primary_key"),
                    None
                )

            confidence = "high" if target_cid and target_pk else "medium"

            found.append({
                "algorithm":       "column_name_matching",
                "from_atom":       cid,
                "from_field":      fname,
                "to_atom":         target_cid,
                "to_field":        target_pk,
                "confidence":      confidence,
                "reason":          f"Field '{fname}' ends in '{matched_suffix}' "
                                   f"and stem '{stem}' matches atom '{_s(target_cid or '')}'",
                "action":          "create_relationship" if target_cid else "review_manually",
            })

    return found


# ── Algorithm 4: Functional dependency ────────────────────────────────────────

def _algo4_functional_dependency(
    atoms:       List[Dict],
    atom_fields: Dict[str, Dict[str, Dict]],
) -> List[Dict]:
    """
    Detect fields that are likely denormalized from a dimension.

    Pattern: if atom has field_X_id AND field_X_name (or field_X_code),
    the name/code field is functionally dependent on the id field —
    it was probably copied from a dimension table.

    e.g. orders has supplier_id + supplier_name → supplier_name is denormalized
    """
    found = []

    for a in atoms:
        cid    = a["canonical_id"]
        fields = atom_fields[cid]
        fnames = set(fields.keys())

        for fname in list(fnames):
            if not any(fname.endswith(s) for s in _FK_SUFFIXES):
                continue
            if fields[fname].get("role") == "primary_key":
                continue

            # Strip suffix to get stem
            stem = re.sub(r"(_id|_key|_code|_ref|_fk)$", "", fname)
            if len(stem) < 2:
                continue

            # Look for dependent fields: stem_name, stem_code, stem_type, stem_status
            dependent_suffixes = ("_name","_code","_type","_status","_category",
                                  "_label","_description","_title","_class")
            deps_found = [
                fn for fn in fnames
                if fn.startswith(stem) and any(fn.endswith(ds) for ds in dependent_suffixes)
                and fn != fname
            ]

            if deps_found:
                found.append({
                    "algorithm":         "functional_dependency",
                    "atom":              cid,
                    "driver_field":      fname,       # e.g. supplier_id
                    "dependent_fields":  deps_found,  # e.g. [supplier_name, supplier_code]
                    "reason":            f"'{fname}' determines {deps_found} — "
                                        f"these are likely denormalized from a dimension atom.",
                    "recommendation":    "Verify these fields are intentionally denormalized. "
                                        "If a suppliers atom exists, cross-table join is preferred."
                })

    return found


# ── Algorithm 5: Hidden dimension extraction ───────────────────────────────────

def _algo5_hidden_dimensions(
    atoms:        List[Dict],
    atom_fields:  Dict[str, Dict[str, Dict]],
    field_values: Dict,
) -> List[Dict]:
    """
    Find fields that contain a small set of fixed values and appear
    in multiple atoms — they may be a hidden dimension that deserves
    its own atom (e.g. status, region, category).

    Uses field_values.json enum lists if available; falls back to
    role-based heuristics.
    """
    found = []

    # Collect all non-PK dimension/flag fields with their enum values
    field_enum_map: Dict[str, List] = {}  # fname → enum values from field_values
    if field_values:
        # field_values.json is typically {canonical_id: {field_name: [values]}}
        for cid_fv, fv_fields in field_values.items():
            if isinstance(fv_fields, dict):
                for fn, vals in fv_fields.items():
                    if isinstance(vals, list) and _MIN_ENUM_VALUES <= len(vals) <= _MAX_ENUM_VALUES:
                        field_enum_map[fn] = vals

    # Find fields with same name in multiple atoms (cross-atom pattern)
    field_atom_count: Dict[str, List[str]] = {}
    for a in atoms:
        cid    = a["canonical_id"]
        fields = atom_fields[cid]
        for fname, fdef in fields.items():
            role = fdef.get("role", "")
            if role not in _LOW_CARD_ROLES:
                continue
            if fdef.get("fk_target_atom"):
                continue  # already a FK, not hidden
            if any(fname.endswith(s) for s in _FK_SUFFIXES):
                continue  # already flagged as FK candidate
            field_atom_count.setdefault(fname, []).append(cid)

    for fname, atom_list in field_atom_count.items():
        if len(atom_list) < 2:
            continue  # only interesting if it appears in 2+ atoms

        enum_vals = field_enum_map.get(fname, [])
        found.append({
            "algorithm":    "hidden_dimension",
            "field_name":   fname,
            "appears_in":   [_s(cid) for cid in atom_list],
            "enum_values":  enum_vals,
            "reason":       f"Field '{fname}' appears in {len(atom_list)} atoms with "
                            f"role=dimension/flag. "
                            + (f"Has {len(enum_vals)} known values: {enum_vals}." if enum_vals
                               else "Consider extracting as a dimension atom."),
            "recommendation": "If this field has a fixed set of values shared across atoms, "
                              "consider creating a dimension atom for it to enable "
                              "cross-table grouping goals.",
        })

    return found


# ── Algorithm 6: Grain verification ───────────────────────────────────────────

def _algo6_grain_verification(
    atoms:       List[Dict],
    atom_fields: Dict[str, Dict[str, Dict]],
) -> List[Dict]:
    """
    Verify that grain_keys declared on each atom:
    1. Actually exist in the atom's fields
    2. Are marked as primary_key or at least dimension role
    3. State atoms have exactly the right grain pattern
       (entity_id + dedup_sort_col)
    """
    issues = []

    for a in atoms:
        cid        = a["canonical_id"]
        grain_keys = a.get("grain_keys", [])
        rt         = a.get("record_type", "record")
        fields     = atom_fields[cid]
        fnames     = set(fields.keys())

        # Check grain_keys exist
        for gk in grain_keys:
            if gk not in fnames:
                issues.append({
                    "algorithm":    "grain_verification",
                    "atom":         cid,
                    "grain_key":    gk,
                    "issue":        "GRAIN_KEY_MISSING",
                    "message":      f"grain_key '{gk}' declared but field does not exist in atom.",
                    "severity":     "CRITICAL",
                    "fix":          f"Add field '{gk}' to atom or correct grain_keys declaration."
                })
                continue

            fdef = fields[gk]
            role = fdef.get("role","")
            if role not in ("primary_key", "dimension", "time"):
                issues.append({
                    "algorithm":  "grain_verification",
                    "atom":       cid,
                    "grain_key":  gk,
                    "issue":      "GRAIN_KEY_WRONG_ROLE",
                    "message":    f"grain_key '{gk}' has role='{role}'. "
                                  f"Expected primary_key, dimension, or time.",
                    "severity":   "HIGH",
                    "fix":        f"Set role of '{gk}' to 'primary_key'."
                })

        # State atoms must have a dedup_sort_col (time field for ordering state changes)
        if rt == "state":
            dedup = a.get("dedup_sort_col")
            if not dedup:
                issues.append({
                    "algorithm": "grain_verification",
                    "atom":      cid,
                    "issue":     "STATE_NO_DEDUP_SORT",
                    "message":   f"State atom '{_s(cid)}' has no dedup_sort_col declared. "
                                 f"State tables need a time field to order state changes.",
                    "severity":  "HIGH",
                    "fix":       "Add dedup_sort_col field pointing to the date/timestamp field."
                })
            elif dedup not in fnames:
                issues.append({
                    "algorithm":     "grain_verification",
                    "atom":          cid,
                    "dedup_sort_col": dedup,
                    "issue":         "DEDUP_FIELD_MISSING",
                    "message":       f"dedup_sort_col '{dedup}' not found in atom fields.",
                    "severity":      "CRITICAL",
                    "fix":           f"Add field '{dedup}' to atom or correct dedup_sort_col."
                })

        # No grain_keys at all is suspicious
        if not grain_keys:
            issues.append({
                "algorithm": "grain_verification",
                "atom":      cid,
                "issue":     "NO_GRAIN_KEYS",
                "message":   f"Atom '{_s(cid)}' has no grain_keys declared.",
                "severity":  "MEDIUM",
                "fix":       "Add grain_keys (usually the primary key field)."
            })

    return issues


# ── Algorithm 7: Temporal pattern analysis ────────────────────────────────────

def _algo7_temporal_patterns(
    atoms:       List[Dict],
    atom_fields: Dict[str, Dict[str, Dict]],
) -> List[Dict]:
    """
    Identify atoms with no date/time field.
    These atoms cannot support time-scoped goals (e.g. 'orders this month').

    Also flag atoms that have date fields but no dedup_sort_col declared
    (state atoms especially need this).
    """
    gaps = []

    for a in atoms:
        cid    = a["canonical_id"]
        rt     = a.get("record_type","record")
        fields = atom_fields[cid]

        # Find all time-role or date-named fields
        time_fields = [
            fname for fname, fdef in fields.items()
            if fdef.get("role") == "time"
            or any(pat in fname.lower() for pat in _DATE_PATTERNS)
        ]

        if not time_fields:
            gaps.append({
                "algorithm":      "temporal_pattern",
                "atom":           cid,
                "record_type":    rt,
                "issue":          "NO_DATE_FIELD",
                "message":        f"'{_s(cid)}' has no date/time field. "
                                  f"Time-scoped goals (this week, this month, YTD) "
                                  f"will not be possible for this atom.",
                "severity":       "MEDIUM",
                "recommendation": "Add a date field (e.g. created_date, transaction_date) "
                                  "if time-based analysis is needed."
            })
        else:
            # Check if time fields have correct role=time
            wrong_role = [
                fname for fname in time_fields
                if fields[fname].get("role") != "time"
            ]
            if wrong_role:
                gaps.append({
                    "algorithm":      "temporal_pattern",
                    "atom":           cid,
                    "issue":          "DATE_FIELD_WRONG_ROLE",
                    "time_fields":    wrong_role,
                    "message":        f"Fields {wrong_role} look like dates but have role != 'time'. "
                                      f"This may prevent time-scoped goal generation.",
                    "severity":       "LOW",
                    "fix":            f"Set role='time' on {wrong_role}."
                })

    return gaps


# ── Helpers ────────────────────────────────────────────────────────────────────

def _s(cid: str) -> str:
    """Short name — strip vertical prefix."""
    parts = cid.split("_")
    return "_".join(parts[2:]) if len(parts) > 2 else cid


def _err(msg: str) -> Dict:
    return {
        "status": "errors", "phase": 1,
        "errors": [{"type":"SETUP_ERROR","message":msg}],
        "confirmed_relationships": [],
        "new_relationships": [],
        "functional_dependencies": [],
        "hidden_dimensions": [],
        "grain_issues": [],
        "temporal_gaps": [],
        "stats": {"errors":1}
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — CSV-based algorithms (run after mock data is complete)
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase2(vertical: str) -> Dict:
    """
    Run Phase 2 discovery algorithms using actual CSV data.

    Algorithm 2 — Value inclusion analysis:
        For each declared relationship from_atom.from_field → to_atom.to_field,
        compute: |values(from_field) ∩ values(to_field)| / |values(from_field)|
        If ratio < 0.8 → warning (FK values not found in target)
        Also scan undeclared _id fields for high-inclusion matches → new FK candidates

    Algorithm 3 — Cardinality analysis:
        For each field, count unique values vs total rows.
        Compare against declared role:
          - primary_key should have uniqueness ~1.0
          - dimension/flag should have low cardinality
          - measure should be numeric
        Flag mismatches.

    Returns same structure as run() but with phase=2 and two extra keys:
        inclusion_results  — per-relationship inclusion ratios
        cardinality_results — per-field cardinality vs declared role
    """
    from pathlib import Path
    import csv as csv_mod

    # Load atoms
    if not _paths.ATOMS_PATH.exists():
        return _err("atoms.json not found.")

    raw   = json.loads(_paths.ATOMS_PATH.read_text(encoding="utf-8"))
    atoms = [a for a in raw.get("_default", {}).values()
             if a.get("domain") == vertical]
    if not atoms:
        return _err(f"No atoms found for vertical '{vertical}'.")

    atom_map    = {a["canonical_id"]: a for a in atoms}
    atom_ids    = set(atom_map.keys())
    atom_fields = {
        cid: {f["name"]: f for f in a.get("fields", [])}
        for cid, a in atom_map.items()
    }

    # Load relationships
    rels_path = _paths.PROJECT_ROOT / "data" / "atom_relationships.json"
    declared_rels = []
    if rels_path.exists():
        all_rels = json.loads(rels_path.read_text(encoding="utf-8"))
        declared_rels = [r for r in all_rels if r.get("from_atom") in atom_ids]

    declared_fk_set = {(r["from_atom"], r["from_field"]) for r in declared_rels}

    # Load all CSVs into memory as value sets
    mock_dir = _paths.MOCK_DATA_DIR / vertical
    csv_values: Dict[str, Dict[str, set]] = {}  # cid → field → set of values

    for cid in atom_ids:
        csv_path = mock_dir / f"{cid}.csv"
        if not csv_path.exists():
            continue
        csv_values[cid] = {}
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv_mod.DictReader(f)
                for row in reader:
                    for field, val in row.items():
                        if field not in csv_values[cid]:
                            csv_values[cid][field] = set()
                        if val and val.strip():
                            csv_values[cid][field].add(val.strip())
        except Exception as e:
            pass

    missing_csvs = [cid for cid in atom_ids if cid not in csv_values]

    # ── Algorithm 2: Value inclusion analysis ─────────────────────────────────
    inclusion_results = []
    new_fk_candidates = []

    # 2a. Check declared relationships
    for r in declared_rels:
        fa, ff = r["from_atom"], r["from_field"]
        ta, tf = r["to_atom"],   r["to_field"]

        if fa not in csv_values or ta not in csv_values:
            inclusion_results.append({
                "from_atom": fa, "from_field": ff,
                "to_atom":   ta, "to_field":   tf,
                "status":    "SKIPPED",
                "reason":    "CSV not available for one or both atoms",
            })
            continue

        from_vals = csv_values[fa].get(ff, set())
        to_vals   = csv_values[ta].get(tf, set())

        if not from_vals:
            inclusion_results.append({
                "from_atom": fa, "from_field": ff,
                "to_atom":   ta, "to_field":   tf,
                "status":    "SKIPPED", "reason": f"No values in {_s(fa)}.{ff}",
            })
            continue

        overlap = len(from_vals & to_vals)
        ratio   = round(overlap / len(from_vals), 3)
        status  = "OK" if ratio >= 0.8 else ("WARN" if ratio >= 0.5 else "FAIL")

        inclusion_results.append({
            "from_atom":     fa, "from_field": ff,
            "to_atom":       ta, "to_field":   tf,
            "from_values":   len(from_vals),
            "to_values":     len(to_vals),
            "overlap":       overlap,
            "inclusion_ratio": ratio,
            "status":        status,
            "message": (
                f"{_s(fa)}.{ff} → {_s(ta)}.{tf}: "
                f"{overlap}/{len(from_vals)} values found in target ({ratio*100:.0f}%)"
            ),
        })

    # 2b. Scan undeclared _id fields for high-inclusion matches
    for cid, fields_vals in csv_values.items():
        for fname, fvals in fields_vals.items():
            if not fname.endswith("_id"):
                continue
            if atom_fields.get(cid, {}).get(fname, {}).get("role") == "primary_key":
                continue
            if (cid, fname) in declared_fk_set:
                continue
            if not fvals:
                continue

            # Check against all other atoms' PK fields
            best_match = None
            best_ratio = 0.0
            for other_cid, other_fields in csv_values.items():
                if other_cid == cid:
                    continue
                other_atom_fields = atom_fields.get(other_cid, {})
                for other_fname, other_fdef in other_atom_fields.items():
                    if other_fdef.get("role") != "primary_key":
                        continue
                    other_vals = other_fields.get(other_fname, set())
                    if not other_vals:
                        continue
                    overlap = len(fvals & other_vals)
                    ratio   = overlap / len(fvals)
                    if ratio > best_ratio:
                        best_ratio  = ratio
                        best_match  = (other_cid, other_fname)

            if best_ratio >= 0.7 and best_match:
                confidence = "high" if best_ratio >= 0.9 else "medium"
                new_fk_candidates.append({
                    "algorithm":       "value_inclusion",
                    "from_atom":       cid,
                    "from_field":      fname,
                    "to_atom":         best_match[0],
                    "to_field":        best_match[1],
                    "inclusion_ratio": round(best_ratio, 3),
                    "confidence":      confidence,
                    "reason": (
                        f"{int(best_ratio*100)}% of {_s(cid)}.{fname} values "
                        f"found in {_s(best_match[0])}.{best_match[1]}"
                    ),
                    "action": "create_relationship",
                })

    # ── Algorithm 3: Cardinality analysis ─────────────────────────────────────
    cardinality_results = []

    for cid, fields_vals in csv_values.items():
        a          = atom_map[cid]
        total_rows = sum(len(v) for v in fields_vals.values()) // max(len(fields_vals), 1)

        # Better total row count from CSV
        csv_path = mock_dir / f"{cid}.csv"
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                total_rows = sum(1 for _ in f) - 1
        except Exception:
            pass

        for fname, fvals in fields_vals.items():
            fdef      = atom_fields.get(cid, {}).get(fname, {})
            role      = fdef.get("role", "unknown")
            data_type = fdef.get("data_type") or fdef.get("type", "")
            n_unique  = len(fvals)
            uniqueness = round(n_unique / total_rows, 3) if total_rows > 0 else 0

            issue = None

            # PK should be fully unique
            # BUT: composite grain keys (multiple grain_keys) don't need
            # individual uniqueness — only the combination does.
            grain_keys = a.get("grain_keys", [])
            is_composite_grain = len(grain_keys) > 1 and fname in grain_keys

            if role == "primary_key" and uniqueness < 0.95 and not is_composite_grain:
                issue = {
                    "issue":    "PK_NOT_UNIQUE",
                    "severity": "CRITICAL",
                    "message":  f"Primary key '{fname}' has only {uniqueness*100:.0f}% uniqueness ({n_unique} unique / {total_rows} rows). PKs must be 100% unique.",
                }
            elif role == "primary_key" and is_composite_grain and uniqueness < 0.95:
                # Composite grain — check composite uniqueness instead
                # Build composite key set from CSV
                csv_path2 = mock_dir / f"{cid}.csv"
                composite_unique = set()
                try:
                    import csv as csv_mod2
                    with open(csv_path2, newline="", encoding="utf-8") as f2:
                        for row2 in csv_mod2.DictReader(f2):
                            composite_key = tuple(row2.get(gk,"") for gk in grain_keys)
                            composite_unique.add(composite_key)
                    composite_uniqueness = len(composite_unique) / total_rows if total_rows > 0 else 0
                    if composite_uniqueness < 0.95:
                        issue = {
                            "issue":    "COMPOSITE_GRAIN_NOT_UNIQUE",
                            "severity": "CRITICAL",
                            "message":  f"Composite grain {grain_keys} has only {composite_uniqueness*100:.0f}% uniqueness ({len(composite_unique)} unique combinations / {total_rows} rows).",
                        }
                except Exception:
                    pass

            # Dimension/flag should be low cardinality
            elif role in ("dimension", "flag") and n_unique > total_rows * 0.8 and n_unique > 10:
                issue = {
                    "issue":    "HIGH_CARDINALITY_DIMENSION",
                    "severity": "LOW",
                    "message":  f"Field '{fname}' has role='{role}' but {n_unique} unique values ({uniqueness*100:.0f}% of rows). Consider role='measure' or 'key'.",
                }

            # Measure should ideally be numeric
            elif role == "measure" and data_type == "string":
                issue = {
                    "issue":    "MEASURE_IS_STRING",
                    "severity": "MEDIUM",
                    "message":  f"Field '{fname}' has role='measure' but data_type='string'. Measures should be numeric.",
                }

            if issue:
                cardinality_results.append({
                    "algorithm": "cardinality",
                    "atom":      cid,
                    "field":     fname,
                    "role":      role,
                    "unique_values": n_unique,
                    "total_rows":    total_rows,
                    "uniqueness":    uniqueness,
                    **issue,
                })

    # ── Build result ──────────────────────────────────────────────────────────
    errors   = [r for r in inclusion_results if r.get("status") == "FAIL"]
    warnings = [r for r in inclusion_results if r.get("status") == "WARN"]
    warnings += [r for r in cardinality_results if r.get("severity") in ("CRITICAL","HIGH","MEDIUM")]

    status = "errors" if errors else ("warnings" if warnings else "ok")

    return {
        "status":              status,
        "vertical":            vertical,
        "phase":               2,
        "missing_csvs":        missing_csvs,
        "inclusion_results":   inclusion_results,
        "new_relationships":   new_fk_candidates,
        "cardinality_results": cardinality_results,
        "errors":              [r["message"] for r in errors],
        "warnings":            [r.get("message","") for r in warnings],
        "stats": {
            "atoms_checked":          len(atoms),
            "csvs_available":         len(csv_values),
            "relationships_checked":  len(declared_rels),
            "inclusion_ok":           sum(1 for r in inclusion_results if r.get("status")=="OK"),
            "inclusion_warn":         sum(1 for r in inclusion_results if r.get("status")=="WARN"),
            "inclusion_fail":         sum(1 for r in inclusion_results if r.get("status")=="FAIL"),
            "new_fk_candidates":      len(new_fk_candidates),
            "cardinality_issues":     len(cardinality_results),
        }
    }


def get_inclusion_failures(vertical: str) -> List[Dict]:
    """
    Return list of atoms that have inclusion failures in Phase 2.
    Each entry has: canonical_id, failing_field, target_atom, target_field, ratio.
    """
    result = run_phase2(vertical)
    failures = []
    for ir in result.get("inclusion_results", []):
        if ir.get("status") == "FAIL":
            failures.append({
                "canonical_id":  ir["from_atom"],
                "failing_field": ir["from_field"],
                "target_atom":   ir["to_atom"],
                "target_field":  ir["to_field"],
                "inclusion_ratio": ir.get("inclusion_ratio", 0),
            })
    return failures
