"""
desktop/relocate.py — relocate the project data tree to a user-writable directory.

The project reads and writes files relative to PROJECT_ROOT (core/paths.py).
In a frozen PyInstaller app the bundle is read-only, so on first launch we:

  1. Copy the bundled data tree to %APPDATA%\\GPLPlatform\\repo\\
  2. Patch core.paths.PROJECT_ROOT to point there

This is a no-op in plain dev mode (the repo is already writable).
Activates when sys.frozen is True OR GPL_DESKTOP_STANDALONE=1.

Directories seeded (everything the servers read/write at runtime):
  data/               factory atoms, seeds, dialects, goals, aterms
  mock_data/          synthetic CSVs
  customer_runtime/   per-customer state
  static/             HTML UIs (read-only at runtime, but easier to seed than exclude)
  logs/               created empty so logging doesn't fail

Call order (from server.py, before importing run / run_customer):
    relocate.activate_if_needed()
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path

_log = logging.getLogger("gpl.relocate")
_activated = False


def is_standalone() -> bool:
    return bool(getattr(sys, "frozen", False)) or os.environ.get("GPL_DESKTOP_STANDALONE") == "1"


def writable_root() -> Path:
    """User-writable directory where the data tree is seeded.

    Windows: %APPDATA%\\GPLPlatform\\repo
    Other:   ~/.local/share/GPLPlatform/repo
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share"))
    return base / "GPLPlatform" / "repo"


def _bundle_root() -> Path:
    """Root of the bundled (read-only) project tree inside the frozen app."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    # Standalone mode for testing without freezing
    return Path(__file__).resolve().parent.parent


def _seed_dirs() -> list[str]:
    """Top-level directories to copy from bundle → writable root."""
    return ["data", "mock_data", "customer_runtime", "static"]


def activate_if_needed() -> None:
    """Seed the writable directory and set COMPILER_DATA_ROOT if in standalone mode."""
    global _activated
    if _activated:
        return
    if not is_standalone():
        _log.debug("relocate: plain dev mode — skipping")
        _activated = True
        return

    dest = writable_root()
    src = _bundle_root()

    _log.info("relocate: seeding writable repo at %s", dest)

    for dir_name in _seed_dirs():
        src_dir = src / dir_name
        dst_dir = dest / dir_name
        if not src_dir.exists():
            continue
        if dst_dir.exists():
            _log.debug("relocate: %s already exists — skipping seed", dst_dir)
            continue
        try:
            shutil.copytree(src_dir, dst_dir)
            _log.info("relocate: seeded %s", dir_name)
        except Exception as e:
            _log.error("relocate: failed to seed %s: %s", dir_name, e)

    # Ensure logs dir exists
    (dest / "logs").mkdir(parents=True, exist_ok=True)

    # Tell the project where its root is
    os.environ["GPL_PROJECT_ROOT"] = str(dest)

    _activated = True
    _log.info("relocate: complete — project root = %s", dest)


def remap_project_root() -> None:
    """Patch core.paths.PROJECT_ROOT to point at the writable copy.

    Must be called AFTER core.paths has been imported (i.e. after the servers
    have been imported), so the module object exists in sys.modules.
    """
    new_root_str = os.environ.get("GPL_PROJECT_ROOT")
    if not new_root_str:
        return

    new_root = Path(new_root_str)

    try:
        import core.paths as _paths
        old = _paths.PROJECT_ROOT

        if old == new_root:
            return  # already correct

        _paths.PROJECT_ROOT            = new_root
        _paths.DATA_DIR                = new_root / "data"
        _paths.ATOMS_PATH              = new_root / "data" / "atoms.json"
        _paths.ATOM_RELATIONSHIPS_PATH = new_root / "data" / "atom_relationships.json"
        _paths.FIELD_VALUES_PATH       = new_root / "data" / "field_values.json"
        _paths.DATA_FINGERPRINTS_PATH  = new_root / "data" / "data_fingerprints.json"
        _paths.MOCK_DATA_DIR           = new_root / "data" / "mock_data"
        _paths.SEEDS_DIR               = new_root / "data" / "seeds"
        _paths.DIALECTS_DIR            = new_root / "data" / "dialects"
        _paths.GOALS_DIR               = new_root / "data" / "goals"
        _paths.ATERMS_DIR              = new_root / "data" / "aterms"
        _paths.CANONICAL_INDEX_PATH    = new_root / "data" / "canonical_index.json"
        _paths.LOCK_REGISTRY_PATH      = new_root / "data" / "lock_registry.json"
        _paths.PACKAGES_DIR            = new_root / "packages"
        _paths.LOGS_DIR                = new_root / "logs"
        _paths.CUSTOMER_RUNTIME_DIR    = new_root / "customer_runtime"
        _paths.CUSTOMER_SOT_DIR        = new_root / "customer_runtime" / "sot_csv"
        _paths.CUSTOMER_KNOWLEDGE_DIR  = new_root / "customer_runtime" / "knowledge_store"
        _paths.CUSTOMER_DATA_DIR       = new_root / "customer_runtime" / "data"

        _log.info("relocate: patched core.paths.PROJECT_ROOT %s → %s", old, new_root)
    except ImportError:
        _log.debug("relocate: core.paths not yet imported — remap skipped")
    except Exception as e:
        _log.error("relocate: remap failed: %s", e)
