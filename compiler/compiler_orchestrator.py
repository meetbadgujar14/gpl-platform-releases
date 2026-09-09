"""
compiler/compiler_orchestrator.py
====================================
Top-level compilation orchestrator for a vertical.

Reads all wave_*.json goal files, routes each goal to the correct branch,
persists aterms, and writes the canonical_index + lock_registry.

COMPILATION ORDER:
  Pass 1: Branch A — algebraic goals (Waves 1, 2, 3, 7, 8)
  Pass 2: Branch C — composed goals (Waves 5, 6, 9)
           Retried until all dependencies resolve or no progress is made.
  Pass 3: Remaining goals → ROUTE_LLM_WIZARD (Branch B, not yet built)
           Currently emitted as "pending_branch_b" in the summary.

OUTPUTS:
  data/aterms/aterm_{canonical_id}.json   — one per compiled goal
  data/canonical_index.json               — fast lookup: cid → {oracle, formula, ...}
  data/lock_registry.json                 — trust/lock status per cid
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import core.paths as _paths
from compiler.branch_a import compile_goal as branch_a
from compiler.branch_b import compile_goal as branch_b
from compiler.branch_c import compile_goal as branch_c
from compiler.decision_tree import ROUTE_COMPOSITION, ROUTE_LLM_WIZARD
from compiler.operators.gpl_operators import build_namespace

log = logging.getLogger(__name__)

# Wave files processed by Branch A (algebraic, no dependencies)
_BRANCH_A_WAVES = {"wave_01", "wave_02", "wave_03", "wave_07", "wave_08",
                   "wave_A", "wave_B", "wave_C", "wave_D", "wave_E"}
# Wave files processed by Branch C (composed, depend on Branch A results)
_BRANCH_C_WAVES = {"wave_05", "wave_06", "wave_09"}


def _now_str() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_atoms(atoms_path: Path) -> Dict[str, Dict]:
    """Load atoms.json → {canonical_id: atom_dict}."""
    raw = json.loads(atoms_path.read_text(encoding="utf-8"))
    # TinyDB format: {_default: {1: {atom}, 2: {atom}, ...}}
    default = raw.get("_default", raw)
    return {v["canonical_id"]: v for v in default.values() if "canonical_id" in v}


def _load_field_values(fv_path: Path) -> Dict:
    if not fv_path.exists():
        return {}
    return json.loads(fv_path.read_text(encoding="utf-8"))


def _load_goals(vertical: str) -> List[Dict]:
    """Load all goal objects from all wave files for a vertical."""
    goals_dir = _paths.GOALS_DIR / vertical
    goals     = []
    for wf in sorted(goals_dir.glob("wave_*.json")):
        d = json.loads(wf.read_text(encoding="utf-8"))
        for g in d.get("goals", []):
            g["_wave_file"] = wf.stem
            goals.append(g)
    return goals


def _find_atom_for_goal(goal: Dict, atoms: Dict[str, Dict]) -> Optional[Dict]:
    """
    Find the atom that backs a goal's entity.
    First tries the explicit 'atom' field in the goal's slots, then
    searches by entity name match in the atom canonical_id.
    """
    slots  = goal.get("slots", {})
    entity = slots.get("entity", "").lower()
    domain = slots.get("domain", "").lower()

    # Exact match via atom canonical_id pattern: domain_entity_*
    for cid, atom in atoms.items():
        cid_lower = cid.lower()
        if domain in cid_lower and entity in cid_lower:
            return atom

    return None


def _persist_aterm(aterm: Dict) -> None:
    """Write one aterm to data/aterms/aterm_{canonical_id}.json."""
    cid  = aterm["canonical_id"]
    path = _paths.ATERMS_DIR / f"aterm_{cid}.json"
    path.write_text(
        json.dumps(aterm, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _update_canonical_index(
    canonical_index: Dict, aterm: Dict
) -> None:
    """Add or update one entry in the in-memory canonical_index."""
    cid = aterm["canonical_id"]
    canonical_index[cid] = {
        "best_aterm_id": f"aterm:{cid}",
        "slots":         aterm.get("slots", {}),
        "oracle_value":  aterm.get("oracle_value"),
        "formula_line":  aterm.get("formula_line", ""),
        "verified":      aterm.get("verified", False),
        "ai_locked":     aterm.get("ai_locked", False),
        "source_goal":   aterm.get("source_goal", cid),
        "wave":          aterm.get("wave"),
    }


def _update_lock_registry(lock_registry: Dict, aterm: Dict) -> None:
    """Add or update one entry in the in-memory lock_registry."""
    cid = aterm["canonical_id"]
    lock_registry.setdefault("aterms", {})[cid] = {
        "ai_locked":    aterm.get("ai_locked", False),
        "hard_locked":  False,
        "ai_lock_date": aterm.get("created_at", ""),
        "lock_method":  aterm.get("lock_method", ""),
        "verify_reason": aterm.get("verify_reason", ""),
        "oracle_value": aterm.get("oracle_value"),
        "formula":      aterm.get("formula_line", ""),
    }


def compile_vertical(vertical: str) -> Dict[str, Any]:
    """
    Compile all goals for a vertical.

    Returns a summary dict with counts per status and any errors.
    """
    log.info(f"[orchestrator] Starting compilation for vertical='{vertical}'")

    # ── Load inputs ───────────────────────────────────────────────────────────
    atoms        = _load_atoms(_paths.ATOMS_PATH)
    field_values = _load_field_values(_paths.FIELD_VALUES_PATH)
    goals        = _load_goals(vertical)
    csv_root     = str(_paths.MOCK_DATA_DIR)

    # Load atom relationships for cross-entity ambiguity detection in Branch B
    from services.relationship_registry import list_relationships
    rels = list_relationships()

    # Ensure operators are initialised with the CSV root
    build_namespace(csv_root)


    # Carry forward any previously compiled aterms
    canonical_index: Dict = {}
    lock_registry:   Dict = {"aterms": {}, "compilation_blocklist": []}

    if _paths.CANONICAL_INDEX_PATH.exists():
        try:
            canonical_index = json.loads(
                _paths.CANONICAL_INDEX_PATH.read_text(encoding="utf-8")
            )
        except Exception:
            canonical_index = {}

    if _paths.LOCK_REGISTRY_PATH.exists():
        try:
            lock_registry = json.loads(
                _paths.LOCK_REGISTRY_PATH.read_text(encoding="utf-8")
            )
        except Exception:
            lock_registry = {"aterms": {}, "compilation_blocklist": []}

    # ── Counters ──────────────────────────────────────────────────────────────
    stats = {
        "compiled_a":    0,
        "compiled_b":    0,
        "compiled_c":    0,
        "routed_b":      0,
        "failed":        0,
        "skipped":       0,
        "errors":        [],
    }
    pending_branch_b: List[Dict] = []

    # ── Pass 1: Branch A — algebraic goals ───────────────────────────────────
    deferred_c: List[Dict] = []

    for goal in goals:
        cid        = goal.get("canonical_id", "")
        wave_file  = goal.get("_wave_file", "")
        complexity = goal.get("complexity", "PRIMITIVE")

        # Skip if already compiled in a previous run
        if cid in canonical_index:
            stats["skipped"] += 1
            continue

        # Route to correct branch:
        # RULE 1: A-E wave goals always go to Branch B — their "formulas" are
        #         natural language descriptions from domain_goals_agent, not
        #         evaluable arithmetic. Even if they have depends_on + formula,
        #         Branch C cannot eval them. Branch B handles everything A-E.
        # RULE 2: COMPOSED_1/2 goals and KPI goals with evaluable arithmetic
        #         formulas (containing {cid} placeholders) go to Branch C.
        # RULE 3: KPI goals with no formula and no deps need Branch B.
        # RULE 4: Everything else → Branch A.

        has_formula   = bool(goal.get("composition_formula", "").strip())
        has_deps      = bool(goal.get("depends_on", []))
        wave_file     = goal.get("_wave_file", "")
        is_ae_wave    = wave_file in {"wave_A","wave_B","wave_C","wave_D","wave_E"}

        # A formula is "evaluable" only if it contains {cid} placeholders
        # that reference real arithmetic — not plain English descriptions
        formula_text  = goal.get("composition_formula", "")
        is_math_formula = "{" in formula_text and "}" in formula_text and (
            any(op in formula_text for op in ["/", "*", "+", "-"])
        )

        # A-E goals → always Branch B
        if is_ae_wave:
            log.info(f"[orchestrator] Branch B (A-E wave): {cid}")
            pending_branch_b.append(goal)
            stats["routed_b"] += 1
            continue

        # Composed/KPI with proper math formula → Branch C
        if complexity in ("COMPOSED_1", "COMPOSED_2") or (
            complexity == "KPI" and has_formula and has_deps and is_math_formula
        ):
            deferred_c.append(goal)
            continue

        # KPI with no evaluable formula → Branch B
        if complexity == "KPI" and not is_math_formula:
            log.info(f"[orchestrator] Branch B (KPI no math formula): {cid}")
            pending_branch_b.append(goal)
            stats["routed_b"] += 1
            continue

        # Find backing atom
        atom = _find_atom_for_goal(goal, atoms)
        if not atom:
            log.warning(f"[orchestrator] No atom found for {cid} — skipping")
            stats["errors"].append({"cid": cid, "reason": "no_atom_found"})
            stats["failed"] += 1
            continue

        result = branch_a(goal, atom, field_values, csv_root)

        if result["status"] == "compiled":
            aterm = result["aterm"]
            _persist_aterm(aterm)
            _update_canonical_index(canonical_index, aterm)
            _update_lock_registry(lock_registry, aterm)
            stats["compiled_a"] += 1
            log.debug(f"[orchestrator] Branch A compiled: {cid}")

        elif result["status"] == "routed":
            route = result["route"]
            if route == ROUTE_COMPOSITION:
                deferred_c.append(goal)
            else:
                log.info(f"[orchestrator] Branch B needed: {cid}")
                pending_branch_b.append(goal)
                stats["routed_b"] += 1

        elif result["status"] == "failed":
            reason = result.get("reason", "unknown")
            log.warning(f"[orchestrator] Branch A failed: {cid} — {reason}")
            stats["errors"].append({"cid": cid, "reason": reason})
            stats["failed"] += 1

    log.info(
        f"[orchestrator] Pass 1 done: "
        f"compiled={stats['compiled_a']} deferred_c={len(deferred_c)} "
        f"routed_b={stats['routed_b']} failed={stats['failed']}"
    )

    # ── Pass 2: Branch B — LLM wizard for A-E goals ──────────────────────────
    if pending_branch_b:
        log.info(
            f"[orchestrator] Pass 2 — Branch B: {len(pending_branch_b)} goals"
        )
        for goal in pending_branch_b:
            cid = goal.get("canonical_id", "")

            if cid in canonical_index:
                stats["skipped"] += 1
                continue

            atom = _find_atom_for_goal(goal, atoms)
            if not atom:
                log.warning(f"[orchestrator] Branch B: no atom for {cid}")
                stats["errors"].append({"cid": cid, "reason": "no_atom_found"})
                stats["failed"] += 1
                continue

            result = branch_b(
                goal, atom, field_values, csv_root, canonical_index,
                atoms_all=atoms, relationships=rels,
            )

            if result["status"] == "compiled":
                aterm = result["aterm"]
                _persist_aterm(aterm)
                _update_canonical_index(canonical_index, aterm)
                _update_lock_registry(lock_registry, aterm)
                stats["compiled_b"] += 1
                log.info(f"[orchestrator] Branch B compiled: {cid}")
            else:
                reason = result.get("reason", "unknown")
                log.warning(f"[orchestrator] Branch B failed: {cid} — {reason}")
                stats["errors"].append({"cid": cid, "reason": reason})
                stats["failed"] += 1

        log.info(f"[orchestrator] Pass 2 done: compiled_b={stats['compiled_b']}")

    # ── Pass 3: Branch C — composed goals (runs after Branch B so all component aterms are available) ──
    # Retry until no progress or all compiled
    max_retries = 5
    for attempt in range(max_retries):
        if not deferred_c:
            break

        still_deferred = []
        progress       = False

        for goal in deferred_c:
            cid = goal.get("canonical_id", "")

            if cid in canonical_index:
                stats["skipped"] += 1
                progress = True
                continue

            result = branch_c(goal, canonical_index)

            if result["status"] == "compiled":
                aterm = result["aterm"]
                _persist_aterm(aterm)
                _update_canonical_index(canonical_index, aterm)
                _update_lock_registry(lock_registry, aterm)
                stats["compiled_c"] += 1
                progress = True
                log.debug(f"[orchestrator] Branch C compiled: {cid}")

            elif result["status"] == "deferred":
                still_deferred.append(goal)

            elif result["status"] == "failed":
                reason = result.get("reason", "unknown")
                log.warning(f"[orchestrator] Branch C failed: {cid} — {reason}")
                stats["errors"].append({"cid": cid, "reason": reason})
                stats["failed"] += 1
                progress = True

        deferred_c = still_deferred
        if not progress:
            log.warning(
                f"[orchestrator] No progress on attempt {attempt + 1} — "
                f"{len(deferred_c)} composed goals remain unresolved"
            )
            break

    # Any remaining deferred_c goals have unresolvable dependencies
    for goal in deferred_c:
        cid = goal.get("canonical_id", "")
        log.warning(f"[orchestrator] Composition unresolvable: {cid}")
        stats["errors"].append({"cid": cid, "reason": "unresolvable_dependencies"})
        stats["failed"] += 1

    log.info(
        f"[orchestrator] Pass 3 done: compiled_c={stats['compiled_c']}"
    )

    # ── Persist canonical_index and lock_registry ─────────────────────────────
    _paths.CANONICAL_INDEX_PATH.write_text(
        json.dumps(canonical_index, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    _paths.LOCK_REGISTRY_PATH.write_text(
        json.dumps(lock_registry, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # ── Write pending_branch_b.json ───────────────────────────────────────────
    # Contains only goals that STILL need Branch B after Pass 3 ran.
    # Includes compile_error so we know WHY each goal failed.
    errors_by_cid = {e["cid"]: e["reason"] for e in stats["errors"]}
    still_pending = []
    for g in pending_branch_b:
        cid = g.get("canonical_id", "")
        if cid not in canonical_index:
            entry = dict(g)
            if cid in errors_by_cid:
                entry["compile_error"] = errors_by_cid[cid]
            still_pending.append(entry)

    pending_path = _paths.GOALS_DIR / vertical / "pending_branch_b.json"
    pending_path.write_text(
        json.dumps({
            "vertical":        vertical,
            "generated_at":    _now_str(),
            "pending_count":   len(still_pending),
            "pending_goals":   still_pending,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    total_compiled = stats["compiled_a"] + stats["compiled_b"] + stats["compiled_c"]
    total_goals    = len(goals)

    log.info(
        f"[orchestrator] Compilation complete — vertical={vertical} "
        f"total={total_goals} compiled={total_compiled} "
        f"(a={stats['compiled_a']} b={stats['compiled_b']} c={stats['compiled_c']}) "
        f"skipped={stats['skipped']} failed={stats['failed']}"
    )

    return {
        "status":                  "success" if stats["failed"] == 0 else "partial",
        "vertical":                vertical,
        "total_goals":             total_goals,
        "compiled_a":              stats["compiled_a"],
        "compiled_b":              stats["compiled_b"],
        "compiled_c":              stats["compiled_c"],
        "total_compiled":          total_compiled,
        "skipped":                 stats["skipped"],
        "routed_b":                stats["routed_b"],
        "failed":                  stats["failed"],
        "errors":                  stats["errors"],
        "canonical_index_path":    str(_paths.CANONICAL_INDEX_PATH),
        "lock_registry_path":      str(_paths.LOCK_REGISTRY_PATH),
        "aterms_dir":              str(_paths.ATERMS_DIR),
        "pending_branch_b_path":   str(pending_path),
    }
