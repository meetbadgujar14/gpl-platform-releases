"""
customer/onboarding.py
=======================
Customer Onboarding Pipeline — runs entirely on the factory side,
but writes to customer-namespaced paths.

Flow (triggered automatically when customer uploads CSVs):
  1. VerticalSchemaAgent CUSTOMER mode  → reconcile schema → customer atoms
  2. MockDataAgent CUSTOMER mode        → schema-matched mock data
  3. discover_enums                     → field_values.json
  4. SeedAgent                          → seed file
  5. VocabularyAgent                    → dialect file
  6. GoalGenerator (Waves 1-9)          → goals
  7. DomainGoalsAgent (Waves A-E)       → domain goals
  8. Compiler                           → aterms + canonical_index
  9. Verifier                           → ai_locked status
  10. DeploymentEngine                  → zip package
  11. Unpack package                    → customer knowledge_store/

The factory pipeline functions are reused but pointed at customer-namespaced
data directories via a context swap. After the pipeline, paths are restored.

Customer data (sot_csv/) is never touched by the factory pipeline.
Only the schema (column names + types) is passed to VerticalSchemaAgent.
"""

import contextlib
import json
import logging
import shutil
import threading
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional

log = logging.getLogger(__name__)

# ── Global lock for the path-swap context ─────────────────────────────────────
# The path swap patches module-level globals in core.paths, atom_registry, and
# relationship_registry. These globals are shared across ALL threads, so only
# ONE customer pipeline can run at a time while the swap is active.
# This lock prevents factory operations from reading customer paths (or vice
# versa) during concurrent execution.
_PATH_SWAP_LOCK = threading.Lock()


# ── Path context manager ───────────────────────────────────────────────────────

@contextlib.contextmanager
def _customer_path_context(vertical: str) -> Generator[Dict[str, Path], None, None]:
    """
    Temporarily redirect all factory data paths to customer-namespaced paths.
    Patches core.paths, atom_registry, and relationship_registry module globals.
    Restores everything on exit.

    THREAD SAFETY: acquires _PATH_SWAP_LOCK for the full duration of the swap.
    Only one customer pipeline may hold this lock at a time (matches the
    "one at a time" architecture requirement).
    """
    import core.paths as p
    import services.atom_registry as ar
    import services.relationship_registry as rr

    # ── Paths to swap in core.paths ───────────────────────────────────────────
    path_swaps = {
        "DATA_DIR":                p.CUSTOMER_DATA_DIR,
        "ATOMS_PATH":              p.CUSTOMER_ATOMS_PATH,
        "ATOM_RELATIONSHIPS_PATH": p.CUSTOMER_ATOM_REL_PATH,
        "FIELD_VALUES_PATH":       p.CUSTOMER_FIELD_VALUES_PATH,
        "DATA_FINGERPRINTS_PATH":  p.CUSTOMER_FINGERPRINTS_PATH,
        "MOCK_DATA_DIR":           p.CUSTOMER_MOCK_DATA_DIR,
        "SEEDS_DIR":               p.CUSTOMER_SEEDS_DIR,
        "DIALECTS_DIR":            p.CUSTOMER_DIALECTS_DIR,
        "GOALS_DIR":               p.CUSTOMER_GOALS_DIR,
        "ATERMS_DIR":              p.CUSTOMER_ATERMS_DIR,
        "CANONICAL_INDEX_PATH":    p.CUSTOMER_CANONICAL_INDEX,
        "LOCK_REGISTRY_PATH":      p.CUSTOMER_LOCK_REGISTRY,
        "PACKAGES_DIR":            p.CUSTOMER_PACKAGES_DIR,
    }

    with _PATH_SWAP_LOCK:
        # Save originals
        orig_paths = {k: getattr(p, k) for k in path_swaps}

        # Save registry module globals
        orig_ar_atoms  = ar.ATOMS_PATH
        orig_rr_atoms  = rr.ATOMS_PATH
        orig_rr_rel    = rr._REL_PATH

        try:
            # Apply path swaps
            for k, v in path_swaps.items():
                setattr(p, k, v)

            # Patch registry module globals so they read/write customer files
            ar.ATOMS_PATH  = p.CUSTOMER_ATOMS_PATH
            rr.ATOMS_PATH  = p.CUSTOMER_ATOMS_PATH
            rr._REL_PATH   = p.CUSTOMER_ATOM_REL_PATH

            yield orig_paths

        finally:
            # Restore core.paths
            for k, v in orig_paths.items():
                setattr(p, k, v)

            # Restore registry globals
            ar.ATOMS_PATH  = orig_ar_atoms
            rr.ATOMS_PATH  = orig_rr_atoms
            rr._REL_PATH   = orig_rr_rel


def _emit(log_cb: Optional[Callable], step: str, msg: str, status: str = "info") -> None:
    """Emit a progress log line to the callback and the logger."""
    entry = {"step": step, "msg": msg, "status": status}
    log.info(f"[onboarding] [{step}] {msg}")
    if log_cb:
        log_cb(entry)


def run_onboarding(
    vertical: str,
    customer_schema: Dict,
    log_cb: Optional[Callable] = None,
    changed_table_names: Optional[set] = None,
) -> Dict[str, Any]:
    """
    Run the full customer onboarding pipeline.

    Args:
        vertical:        e.g. "supply_chain"
        customer_schema: {tables: [{name, columns: [{name, type}]}], relationships: [...]}
                         — schema only, no data values
        log_cb:          optional callback({"step", "msg", "status"}) for live progress
        changed_table_names: table names that are new/changed since the last
                         Generate call (i.e. currently-pending tables, before
                         merge into committed history). Used to scope mock
                         data + domain goal regeneration to only the atoms
                         connected to what actually changed. If None or
                         empty, everything regenerates (safe default, and
                         correct behavior for the very first Generate call).

    Returns:
        {status, vertical, steps, knowledge_store_path, metrics_available, errors}
    """
    steps: List[Dict] = []
    errors: List[str] = []

    def emit(step, msg, status="info"):
        entry = {"step": step, "msg": msg, "status": status}
        steps.append(entry)
        _emit(log_cb, step, msg, status)

    emit("start", f"Starting onboarding for vertical='{vertical}'")

    with _customer_path_context(vertical):

        # ── Step 1: VerticalSchemaAgent CUSTOMER mode ──────────────────────────
        emit("schema", "Reconciling customer schema against factory atoms (CUSTOMER mode)...")
        try:
            from agents.vertical_schema_agent import run as vsa_run
            result = vsa_run(
                vertical=vertical,
                mode="CUSTOMER",
                customer_schema=customer_schema,
            )
            if result.get("status") == "error":
                emit("schema", f"Schema agent error: {result.get('error')}", "error")
                errors.append(result.get("error", "VerticalSchemaAgent failed"))
                return _fail(vertical, steps, errors)

            n_atoms = len(result.get("atoms_created", [])) + len(result.get("atoms_updated", []))
            emit("schema", f"Atom reconciliation complete — {n_atoms} atoms written", "ok")
        except Exception as e:
            emit("schema", f"Failed: {e}", "error")
            errors.append(str(e))
            return _fail(vertical, steps, errors)

        # ── Compute the affected-atom connected component ──────────────────────
        # Now that atoms.json is up to date, map changed table names to atom
        # canonical_ids and walk the FK relationship graph outward from them.
        # This is what lets mock data + domain goal generation skip atoms
        # that have no relationship path to whatever just changed.
        only_atom_ids = None
        if changed_table_names:
            try:
                from services import impact
                changed_atom_ids = impact.table_names_to_atom_ids(changed_table_names, vertical)
                affected = impact.compute_affected_atoms(vertical, changed_atom_ids)
                if affected:
                    only_atom_ids = affected
                    emit(
                        "schema",
                        f"Scoping regeneration to {len(affected)} affected atom(s): "
                        f"{', '.join(sorted(affected))}",
                        "info",
                    )
                else:
                    emit(
                        "schema",
                        "Could not determine an affected-atom scope — regenerating the full vertical",
                        "info",
                    )
            except Exception as e:
                log.warning(f"[onboarding] Impact computation failed, falling back to full regen: {e}")

        # ── Step 2: MockDataAgent CUSTOMER mode ────────────────────────────────
        emit("mock_data", "Generating schema-matched mock data (CUSTOMER mode)...")
        try:
            from agents.mock_data_agent import run as mock_run
            result = mock_run(vertical=vertical, mode="CUSTOMER", only_atom_ids=only_atom_ids)
            if result.get("status") == "error":
                emit("mock_data", f"Mock data error: {result.get('error')}", "error")
                errors.append(result.get("error", "MockDataAgent failed"))
                return _fail(vertical, steps, errors)
            n_reused = len(result.get("reused_atoms", []))
            emit(
                "mock_data",
                f"Mock data generated — {result.get('total_rows', 0)} rows across "
                f"{result.get('atoms_processed', 0)} atoms"
                + (f" ({n_reused} reused, untouched)" if n_reused else ""),
                "ok",
            )
        except Exception as e:
            emit("mock_data", f"Failed: {e}", "error")
            errors.append(str(e))
            return _fail(vertical, steps, errors)

        # ── Step 3: discover_enums ─────────────────────────────────────────────
        emit("enums", "Discovering enum values from mock data...")
        try:
            from services.discover_enums import discover_enums as enums_run
            result = enums_run(vertical=vertical)
            emit("enums", f"field_values.json written — {len(result)} enum columns", "ok")
            # Note: sot_ingestion writes real SOT enum values directly into field_values.json
            # at upload time, so no merge-back step is needed here.

        except Exception as e:
            emit("enums", f"Failed: {e}", "error")
            errors.append(str(e))
            return _fail(vertical, steps, errors)

        # ── Step 4: SeedAgent ──────────────────────────────────────────────────
        emit("seed", "Generating seed file...")
        try:
            from agents.seed_agent import run as seed_run
            result = seed_run(vertical=vertical)
            if result.get("status") == "error":
                emit("seed", f"Seed error: {result.get('error')}", "error")
                errors.append(result.get("error", "SeedAgent failed"))
                return _fail(vertical, steps, errors)
            emit("seed", "Seed file written", "ok")
        except Exception as e:
            emit("seed", f"Failed: {e}", "error")
            errors.append(str(e))
            return _fail(vertical, steps, errors)

        # ── Step 5: VocabularyAgent ────────────────────────────────────────────
        emit("vocabulary", "Generating GPL dialect / vocabulary...")
        try:
            from agents.vocabulary_agent import run as vocab_run
            result = vocab_run(vertical=vertical)
            if result.get("status") == "error":
                emit("vocabulary", f"Vocab error: {result.get('error')}", "error")
                errors.append(result.get("error", "VocabularyAgent failed"))
                return _fail(vertical, steps, errors)
            emit("vocabulary", "GPL_dialects.json written", "ok")
        except Exception as e:
            emit("vocabulary", f"Failed: {e}", "error")
            errors.append(str(e))
            return _fail(vertical, steps, errors)

        # ── Step 6: Goal Generation (Waves 1-9) ────────────────────────────────
        emit("goals", "Generating goals (Waves 1-9)...")
        try:
            from services.goal_generator import generate_goals
            result = generate_goals(vertical=vertical)
            n = result.get("total_goals", 0)
            emit("goals", f"Waves 1-9 complete — {n} goals generated", "ok")
        except Exception as e:
            emit("goals", f"Failed: {e}", "error")
            errors.append(str(e))
            return _fail(vertical, steps, errors)

        # ── Step 7: Domain Goals (Waves A-E) ───────────────────────────────────
        emit("domain_goals", "Generating domain goals (Waves A-E)...")
        try:
            from agents.domain_goals_agent import generate_domain_goals
            all_cids = list(result.get("all_cids", []))
            result_ae = generate_domain_goals(
                vertical=vertical,
                wave_19_cids=all_cids,
                only_atom_ids=only_atom_ids,
            )
            n_ae = result_ae.get("total_ae_goals", 0)
            emit("domain_goals", f"Waves A-E complete — {n_ae} domain goals generated", "ok")
        except Exception as e:
            emit("domain_goals", f"Failed: {e}", "error")
            errors.append(str(e))
            return _fail(vertical, steps, errors)

        # ── Step 8: Compilation ────────────────────────────────────────────────
        emit("compile", "Compiling goals (Branch A → C → B)...")
        try:
            from compiler.compiler_orchestrator import compile_vertical
            result = compile_vertical(vertical=vertical)
            total = result.get("total_compiled", 0)
            emit("compile", f"Compilation complete — {total} aterms compiled", "ok")
        except Exception as e:
            emit("compile", f"Failed: {e}", "error")
            errors.append(str(e))
            return _fail(vertical, steps, errors)

        # ── Step 9: Verification ───────────────────────────────────────────────
        emit("verify", "Verifying compiled aterms...")
        try:
            from compiler.verifier import verify_vertical
            result = verify_vertical(vertical=vertical)
            locked = result.get("ai_locked_count", 0)
            emit("verify", f"Verification complete — {locked} aterms ai_locked", "ok")
        except Exception as e:
            emit("verify", f"Failed: {e}", "error")
            errors.append(str(e))
            return _fail(vertical, steps, errors)

        # ── Step 10: Package ───────────────────────────────────────────────────
        emit("package", "Building deployment package...")
        try:
            from compiler.deployment_engine import package_vertical
            result = package_vertical(vertical=vertical)
            pkg_path = result.get("package_path", "")
            metrics  = result.get("metric_count", 0)
            locked   = result.get("locked_count", 0)
            emit("package", f"Package built — {metrics} metrics, {locked} locked: {Path(pkg_path).name}", "ok")
        except Exception as e:
            emit("package", f"Failed: {e}", "error")
            errors.append(str(e))
            return _fail(vertical, steps, errors)

    # ── Step 11: Unpack to knowledge_store/ ───────────────────────────────────
    # (outside path context — writes to customer_runtime/knowledge_store/)
    emit("unpack", "Unpacking deployment package to knowledge store...")
    try:
        ks_path = _unpack_package(pkg_path)
        emit("unpack", f"Knowledge store ready at {ks_path}", "ok")
    except Exception as e:
        emit("unpack", f"Failed: {e}", "error")
        errors.append(str(e))
        return _fail(vertical, steps, errors)

    # Count available metrics
    from core.paths import CUSTOMER_KNOWLEDGE_DIR
    ks_index = CUSTOMER_KNOWLEDGE_DIR / "canonical_index.json"
    metrics_available = 0
    if ks_index.exists():
        try:
            idx = json.loads(ks_index.read_text(encoding="utf-8"))
            metrics_available = len(idx)
        except Exception:
            pass

    emit("done", f"Onboarding complete — {metrics_available} metrics available in knowledge store", "ok")

    return {
        "status":              "ready",
        "vertical":            vertical,
        "steps":               steps,
        "knowledge_store_path": str(CUSTOMER_KNOWLEDGE_DIR),
        "metrics_available":   metrics_available,
        "errors":              errors,
    }


def _unpack_package(zip_path: str) -> str:
    """
    Unpack the deployment zip into customer_runtime/knowledge_store/.
    Clears any previous knowledge store first.
    """
    from core.paths import CUSTOMER_KNOWLEDGE_DIR

    ks = CUSTOMER_KNOWLEDGE_DIR
    if ks.exists():
        shutil.rmtree(ks)
    ks.mkdir(parents=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(ks)

    # Flatten: move library/* up to knowledge_store/ root for easy access
    lib_dir = ks / "library"
    if lib_dir.exists():
        for f in lib_dir.iterdir():
            dest = ks / f.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(f), str(ks))
        shutil.rmtree(lib_dir)

    log.info(f"[onboarding] Knowledge store unpacked to {ks}")
    return str(ks)


def _fail(vertical: str, steps: List, errors: List) -> Dict:
    return {
        "status":              "error",
        "vertical":            vertical,
        "steps":               steps,
        "knowledge_store_path": None,
        "metrics_available":   0,
        "errors":              errors,
    }


# ── Multi-tenant path context ──────────────────────────────────────────────────

@contextlib.contextmanager
def _customer_path_context_for_customer(customer_id: str, vertical: str):
    """
    Like _customer_path_context() but scoped to a specific customer_id.
    Uses get_customer_paths(customer_id) to build isolated per-customer paths.

    The per-customer lock is held by _run_pipeline() in customer_onboard_router.py
    BEFORE this context manager is entered, so thread safety is guaranteed at
    the router level (one pipeline per customer at a time).
    """
    import core.paths as p
    import services.atom_registry as ar
    import services.relationship_registry as rr
    from core.paths import get_customer_paths

    cpaths = get_customer_paths(customer_id)

    path_swaps = {
        "DATA_DIR":                cpaths["DATA_DIR"],
        "ATOMS_PATH":              cpaths["ATOMS_PATH"],
        "ATOM_RELATIONSHIPS_PATH": cpaths["ATOM_REL_PATH"],
        "FIELD_VALUES_PATH":       cpaths["FIELD_VALUES_PATH"],
        "DATA_FINGERPRINTS_PATH":  cpaths["FINGERPRINTS_PATH"],
        "MOCK_DATA_DIR":           cpaths["MOCK_DATA_DIR"],
        "SEEDS_DIR":               cpaths["SEEDS_DIR"],
        "DIALECTS_DIR":            cpaths["DIALECTS_DIR"],
        "GOALS_DIR":               cpaths["GOALS_DIR"],
        "ATERMS_DIR":              cpaths["ATERMS_DIR"],
        "CANONICAL_INDEX_PATH":    cpaths["CANONICAL_INDEX"],
        "LOCK_REGISTRY_PATH":      cpaths["LOCK_REGISTRY"],
        "PACKAGES_DIR":            cpaths["PACKAGES_DIR"],
    }

    # Save originals
    orig_paths = {k: getattr(p, k) for k in path_swaps}
    orig_ar_atoms = ar.ATOMS_PATH
    orig_rr_atoms = rr.ATOMS_PATH
    orig_rr_rel   = rr._REL_PATH

    try:
        for k, v in path_swaps.items():
            setattr(p, k, v)
        ar.ATOMS_PATH = cpaths["ATOMS_PATH"]
        rr.ATOMS_PATH = cpaths["ATOMS_PATH"]
        rr._REL_PATH  = cpaths["ATOM_REL_PATH"]
        yield cpaths
    finally:
        for k, v in orig_paths.items():
            setattr(p, k, v)
        ar.ATOMS_PATH = orig_ar_atoms
        rr.ATOMS_PATH = orig_rr_atoms
        rr._REL_PATH  = orig_rr_rel
