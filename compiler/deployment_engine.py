"""
compiler/deployment_engine.py
================================
GPL Deployment Engine — packages compiled aterms into a deployment zip
ready for the customer runtime environment.

What this does:
  1. Dependency resolution  — every aterm's depends_on CID is in the package
  2. Schema validation      — formula columns exist in atom definitions
  3. required_columns.json  — tells customer which CSV columns they must provide
  4. composition_rules.json — dependency graph for the runtime to resolve composed goals
  5. Package assembly       — structured directory layout inside a zip
  6. MANIFEST.json          — package metadata

What this does NOT do yet (added when customer runtime is built):
  - HMAC-SHA256 signing of the manifest
  - AES-256-GCM encryption of library files
  - Transmission over the Knowledge Delivery Channel
  - Customer registry lookup / schema compatibility check against real customer data

Package layout:
  supply_chain_v1_YYYYMMDD_HHMMSS.zip
    MANIFEST.json
    library/
      canonical_index.json
      lock_registry.json
      composition_rules.json
      aterms/
        aterm_{cid}.json  (one per compiled goal)
    schema/
      required_columns.json
    data_fingerprints.json
"""

import json
import logging
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import core.paths as _paths

log = logging.getLogger(__name__)

_ENGINE_VERSION = "1.0"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _package_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_atoms() -> Dict[str, Dict]:
    raw = json.loads(_paths.ATOMS_PATH.read_text(encoding="utf-8"))
    default = raw.get("_default", raw)
    return {v["canonical_id"]: v for v in default.values() if "canonical_id" in v}


# ── Step 1: Dependency resolution ────────────────────────────────────────────

def _resolve_dependencies(
    canonical_index: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Verify every depends_on CID referenced in any aterm is present in
    the canonical_index. Returns a dict of {cid: [missing_deps]} for
    any CID whose dependencies aren't fully satisfied.
    """
    missing: Dict[str, List[str]] = {}
    for cid, entry in canonical_index.items():
        # depends_on is stored on the aterm file, not always in canonical_index
        # Load the aterm for the full dep list
        aterm_path = _paths.ATERMS_DIR / f"aterm_{cid}.json"
        if not aterm_path.exists():
            continue
        aterm     = json.loads(aterm_path.read_text(encoding="utf-8"))
        deps      = aterm.get("depends_on", [])
        not_found = [d for d in deps if d not in canonical_index]
        if not_found:
            missing[cid] = not_found

    return missing


# ── Step 2: Schema validation ─────────────────────────────────────────────────

def _validate_schemas(
    canonical_index: Dict[str, Any],
    atoms: Dict[str, Dict],
) -> List[Dict]:
    """
    For every aterm, verify that the columns referenced in its formula_line
    exist in the atom's field definitions.

    Returns a list of validation issues [{cid, column, issue}].
    We check by extracting column names from the atom's fields and confirming
    the aterm's slot entity matches a known atom.
    """
    issues = []

    # Build column map: atom_cid → set of known column names
    col_map: Dict[str, set] = {}
    for atom_cid, atom in atoms.items():
        col_map[atom_cid] = {f["name"] for f in atom.get("fields", [])}

    for cid, entry in canonical_index.items():
        slots  = entry.get("slots", {})
        entity = slots.get("entity", "")
        domain = slots.get("domain", "")

        # Composed goals (Wave 5,6,9) don't have their own atom — they compute
        # from other aterms. Skip schema validation for these.
        aterm_path = _paths.ATERMS_DIR / f"aterm_{cid}.json"
        if aterm_path.exists():
            aterm = json.loads(aterm_path.read_text(encoding="utf-8"))
            if aterm.get("kind") == "composed":
                continue

        # Find the primary atom for this goal
        atom_cid = next(
            (a for a in atoms if domain.lower() in a.lower() and entity.lower() in a.lower()),
            None
        )
        if not atom_cid:
            issues.append({"cid": cid, "issue": f"No atom found for entity='{entity}'"})
            continue

        known_cols = col_map.get(atom_cid, set())
        formula    = entry.get("formula_line", "")

        # Extract column-like tokens from the formula and check against known columns
        # We look for tokens that appear between quotes in the formula string
        import re
        quoted = re.findall(r"'([^']+)'", formula)
        for token in quoted:
            # Skip atom CIDs and operator names and known non-column strings
            if token in atoms:
                continue
            if "_" in token and any(c.isupper() for c in token):
                continue
            # Check if it looks like a column name (lowercase with underscores)
            if re.match(r'^[a-z][a-z0-9_]*$', token) and token not in known_cols:
                # Could be a filter VALUE not a column name — only flag if it
                # matches a pattern that looks like a column (has underscore or
                # is longer than typical values)
                if "_" in token or len(token) > 15:
                    issues.append({
                        "cid":    cid,
                        "column": token,
                        "issue":  f"Column '{token}' not found in atom '{atom_cid}'"
                    })

    return issues


# ── Step 3: required_columns.json ─────────────────────────────────────────────

def _build_required_columns(atoms: Dict[str, Dict]) -> Dict:
    """
    Describe exactly which CSV files and columns the customer must provide
    for the deployment package to work at their site.

    This is the contract between the factory and the customer's data team.
    """
    requirements = {}
    for atom_cid, atom in atoms.items():
        fields = atom.get("fields", [])
        requirements[atom_cid] = {
            "description":   f"Required CSV for {atom.get('record_type', 'record')} entity",
            "suggested_filename": f"{atom_cid}.csv",
            "grain_keys":    atom.get("grain_keys", []),
            "dedup_sort_col": atom.get("dedup_sort_col"),
            "record_type":   atom.get("record_type", "record"),
            "required_columns": [
                {
                    "name":        f["name"],
                    "role":        f.get("role", ""),
                    "type":        f.get("type", "string"),
                    "nullable":    f.get("role") not in ("primary_key", "time"),
                    "fk_target":   f.get("fk_target_atom"),
                }
                for f in fields
            ],
        }
    return requirements


# ── Step 4: composition_rules.json ────────────────────────────────────────────

def _build_composition_rules(canonical_index: Dict) -> Dict:
    """
    Build a dependency graph the customer runtime uses to resolve composed
    goals at query time — e.g. to compute average_order_value, the runtime
    looks up order_amount_total and order_count, then evaluates the formula.

    Structure: {cid: {formula, depends_on, dep_oracle_values}}
    """
    rules = {}
    for cid in canonical_index:
        aterm_path = _paths.ATERMS_DIR / f"aterm_{cid}.json"
        if not aterm_path.exists():
            continue
        aterm = json.loads(aterm_path.read_text(encoding="utf-8"))
        deps  = aterm.get("depends_on", [])
        if not deps:
            continue

        rules[cid] = {
            "formula":          aterm.get("formula_line", ""),
            "depends_on":       deps,
            "dep_oracle_values": aterm.get("dep_values", {}),
            "complexity":       aterm.get("slots", {}).get("measure", ""),
        }

    return rules


# ── Step 5 + 6: Package assembly + MANIFEST ───────────────────────────────────

def package_vertical(vertical: str, version: Optional[str] = None) -> Dict[str, Any]:
    """
    Build a deployment package for a vertical.

    Args:
        vertical: e.g. "supply_chain"
        version:  optional version string (default: "v1")

    Returns:
        {status, package_path, manifest, validation_issues, missing_deps}
    """
    log.info(f"[deployment_engine] Building package for vertical='{vertical}'")

    if not _paths.CANONICAL_INDEX_PATH.exists():
        raise ValueError("canonical_index.json not found. Run compilation first.")
    if not _paths.LOCK_REGISTRY_PATH.exists():
        raise ValueError("lock_registry.json not found. Run compilation first.")

    canonical_index = json.loads(_paths.CANONICAL_INDEX_PATH.read_text(encoding="utf-8"))
    lock_registry   = json.loads(_paths.LOCK_REGISTRY_PATH.read_text(encoding="utf-8"))
    atoms           = _load_atoms()

    if not canonical_index:
        raise ValueError("canonical_index is empty. Nothing to package.")

    ver = version or "v1"
    ts  = _package_timestamp()

    # ── Step 1: Dependency resolution ────────────────────────────────────────
    missing_deps = _resolve_dependencies(canonical_index)
    if missing_deps:
        log.warning(f"[deployment_engine] Missing dependencies: {missing_deps}")

    # ── Step 2: Schema validation ─────────────────────────────────────────────
    schema_issues = _validate_schemas(canonical_index, atoms)
    if schema_issues:
        log.warning(f"[deployment_engine] Schema issues: {len(schema_issues)}")

    # ── Steps 3 + 4: Build derived artifacts ─────────────────────────────────
    required_columns  = _build_required_columns(atoms)
    composition_rules = _build_composition_rules(canonical_index)

    # ── Step 5: Assemble staging directory ───────────────────────────────────
    pkg_name = f"{vertical}_{ver}_{ts}"
    pkg_dir  = _paths.PACKAGES_DIR / pkg_name
    if pkg_dir.exists():
        shutil.rmtree(pkg_dir)
    pkg_dir.mkdir(parents=True)

    # library/
    lib_dir = pkg_dir / "library"
    lib_dir.mkdir()

    # canonical_index.json
    (lib_dir / "canonical_index.json").write_text(
        json.dumps(canonical_index, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # lock_registry.json
    (lib_dir / "lock_registry.json").write_text(
        json.dumps(lock_registry, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # composition_rules.json
    (lib_dir / "composition_rules.json").write_text(
        json.dumps(composition_rules, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # library/aterms/
    aterms_dst = lib_dir / "aterms"
    aterms_dst.mkdir()
    aterm_count  = 0
    locked_count = 0
    for af in _paths.ATERMS_DIR.glob("aterm_*.json"):
        aterm = json.loads(af.read_text(encoding="utf-8"))
        # Only package ai_locked aterms — unlocked ones aren't ready to ship
        if aterm.get("ai_locked", False):
            shutil.copy2(af, aterms_dst / af.name)
            aterm_count  += 1
            locked_count += 1
        else:
            log.warning(f"[deployment_engine] Skipping unlocked aterm: {af.name}")

    # schema/
    schema_dir = pkg_dir / "schema"
    schema_dir.mkdir()
    (schema_dir / "required_columns.json").write_text(
        json.dumps(required_columns, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # data_fingerprints.json — for cache invalidation at customer runtime
    if _paths.DATA_FINGERPRINTS_PATH.exists():
        shutil.copy2(_paths.DATA_FINGERPRINTS_PATH, pkg_dir / "data_fingerprints.json")

    # GPL_dialects.json — intent resolver needs vocabulary at customer runtime
    from core.paths import DIALECTS_DIR
    dialect_path = DIALECTS_DIR / f"{vertical}_dialect.json"
    if dialect_path.exists():
        shutil.copy2(dialect_path, lib_dir / "GPL_dialects.json")
    else:
        log.warning(f"[deployment_engine] No dialect file found for '{vertical}' — intent resolver will have limited vocabulary")

    # ── Step 6: MANIFEST.json ─────────────────────────────────────────────────
    # TODO: add HMAC-SHA256 signature here when customer runtime is built.
    # Key: derived from customer_id + "GPL_FACTORY_V1" + vertical
    # The runtime validates the signature before integrating elements.
    manifest = {
        "package_name":    pkg_name,
        "vertical":        vertical,
        "package_version": ver,
        "engine_version":  _ENGINE_VERSION,
        "created_at":      _now(),
        "metric_count":    aterm_count,
        "locked_count":    locked_count,
        "lock_rate":       round(locked_count / max(aterm_count, 1) * 100, 1),
        "composition_rules_count": len(composition_rules),
        "atoms_included":  list(atoms.keys()),
        "validation": {
            "missing_dep_count":  len(missing_deps),
            "schema_issue_count": len(schema_issues),
            "ready_to_deploy":    len(missing_deps) == 0 and locked_count == aterm_count,
        },
        "data_requirements": [
            {
                "atom_canonical_id": atom_cid,
                "suggested_filename": f"{atom_cid}.csv",
                "column_count": len(atoms[atom_cid].get("fields", [])),
            }
            for atom_cid in atoms
        ],
        # Signing placeholder — not yet implemented
        "signature": None,
        "signature_algorithm": "HMAC-SHA256 (not yet implemented — added with customer runtime)",
    }

    (pkg_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ── Zip the package ───────────────────────────────────────────────────────
    zip_path = _paths.PACKAGES_DIR / f"{pkg_name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(pkg_dir.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(pkg_dir))

    # Clean up staging directory
    shutil.rmtree(pkg_dir)

    log.info(
        f"[deployment_engine] Package ready: {zip_path.name} "
        f"({aterm_count} aterms, {locked_count} locked)"
    )

    return {
        "status":           "ready" if manifest["validation"]["ready_to_deploy"] else "partial",
        "package_path":     str(zip_path),
        "package_name":     pkg_name,
        "metric_count":     aterm_count,
        "locked_count":     locked_count,
        "lock_rate":        manifest["lock_rate"],
        "composition_rules": len(composition_rules),
        "missing_deps":     missing_deps,
        "schema_issues":    schema_issues,
        "manifest":         manifest,
    }


def get_packages(vertical: str) -> List[Dict]:
    """List all deployment packages for a vertical."""
    packages = []
    for zf in sorted(_paths.PACKAGES_DIR.glob(f"{vertical}_*.zip"), reverse=True):
        # Read manifest from inside the zip
        try:
            with zipfile.ZipFile(zf) as z:
                manifest = json.loads(z.read("MANIFEST.json").decode("utf-8"))
            packages.append({
                "package_name":    manifest.get("package_name"),
                "package_version": manifest.get("package_version"),
                "created_at":      manifest.get("created_at"),
                "metric_count":    manifest.get("metric_count"),
                "locked_count":    manifest.get("locked_count"),
                "lock_rate":       manifest.get("lock_rate"),
                "ready_to_deploy": manifest.get("validation", {}).get("ready_to_deploy"),
                "file":            zf.name,
                "size_kb":         round(zf.stat().st_size / 1024, 1),
            })
        except Exception as e:
            log.warning(f"[deployment_engine] Could not read manifest from {zf.name}: {e}")
    return packages
