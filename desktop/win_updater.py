"""
desktop/win_updater.py — Windows folder-swap apply step.

This module runs in a DETACHED copy of the exe (spawned by updater.apply_update_windows).
It is invoked before Qt starts — main.py checks sys.argv[1] == '--apply-update'
and calls win_updater.apply(zip_path) directly.

Sequence:
  1. Wait for all processes with handles in the install dir to exit (up to 60s)
  2. Extract the ZIP to a staging directory
  3. Swap staging → install (rename-based, atomic-ish)
  4. Relaunch the new exe with --verify-startup
  5. Wait for startup sentinel (up to 60s)
  6. Rollback to backup if sentinel never appears
  7. Clean up
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

_log = logging.getLogger("gpl.win_updater")

APP_EXE_NAME = "GPLPlatform.exe"

# Processes that hold handles everywhere on the system — never terminate these
_PROTECTED_PROCS = {
    "msmpeng.exe", "mssense.exe", "securityhealthservice.exe",
    "explorer.exe", "svchost.exe", "csrss.exe", "smss.exe",
    "wininit.exe", "winlogon.exe", "services.exe", "lsass.exe",
    "system", "registry",
}

_SENTINEL_PATH = Path(
    os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming")
) / "GPLPlatform" / "update" / "startup_ok"

_LOCAL_APPDATA = Path(
    os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _install_dir() -> Path:
    return Path(os.environ.get("GPL_UPDATE_APP_DIR", str(Path(sys.executable).parent)))


def _staging_dir() -> Path:
    return _LOCAL_APPDATA / "GPLPlatform" / "staging"


def _backup_dir() -> Path:
    return _LOCAL_APPDATA / "GPLPlatform" / "app_backup"


def _rename_retry(src: Path, dst: Path, attempts: int = 20) -> None:
    """Rename with exponential backoff — Windows file lock races (AV, indexer)."""
    delay = 0.25
    last_err = None
    for i in range(attempts):
        try:
            src.rename(dst)
            return
        except OSError as e:
            last_err = e
            jitter = random.uniform(0, delay * 0.3)
            time.sleep(min(delay + jitter, 10.0))
            delay = min(delay * 2, 10.0)
    raise RuntimeError(f"rename {src} → {dst} failed after {attempts} attempts: {last_err}")


def _wait_for_processes(install: Path, timeout: float = 60.0) -> None:
    """Wait until no running process has files open inside the install dir."""
    try:
        import psutil
    except ImportError:
        _log.warning("psutil not available — skipping process wait (may cause rename failure)")
        time.sleep(3.0)
        return

    install_str = str(install).lower()
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        blockers = []
        for proc in psutil.process_iter(["pid", "name", "exe", "cwd"]):
            try:
                name = (proc.info["name"] or "").lower()
                if name in _PROTECTED_PROCS:
                    continue
                exe = (proc.info["exe"] or "").lower()
                cwd = (proc.info["cwd"] or "").lower()
                if install_str in exe or install_str in cwd:
                    blockers.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        if not blockers:
            break

        elapsed = time.monotonic() - (deadline - timeout)
        if elapsed > 30:
            for proc in blockers:
                try:
                    _log.warning("Terminating blocking process: %s (pid %d)", proc.name(), proc.pid)
                    proc.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

        time.sleep(0.5)


# ── Main apply logic ──────────────────────────────────────────────────────────

def _setup_logging() -> None:
    log_dir = Path(os.environ.get("APPDATA") or "") / "GPLPlatform" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "updater.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def apply(zip_path_str: str) -> None:
    """Entry point called from main.py when --apply-update is in sys.argv."""
    _setup_logging()
    _log.info("win_updater: apply started — zip=%s", zip_path_str)

    zip_path = Path(zip_path_str)
    install = _install_dir()
    staging = _staging_dir()
    backup = _backup_dir()

    # ── Step 1: wait for the main app to fully exit ───────────────────────────
    _log.info("Waiting for main app processes to exit ...")
    _wait_for_processes(install)
    _log.info("Install dir is clear")

    # ── Step 2: extract ZIP to staging ───────────────────────────────────────
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    _log.info("Extracting %s → %s", zip_path, staging)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(staging)

    # Sanity check — the zip must contain the exe
    if not (staging / APP_EXE_NAME).exists():
        _log.error("ZIP does not contain %s — aborting", APP_EXE_NAME)
        _launch_old(install, "--rollback-done")
        return

    # ── Step 3: save Inno uninstaller files so Add/Remove Programs keeps working
    unins_files: list[Path] = []
    for fname in ("unins000.exe", "unins000.dat"):
        p = install / fname
        if p.exists():
            unins_files.append(p)

    # ── Step 4: atomic folder swap ────────────────────────────────────────────
    _log.info("Swapping install dir ...")
    try:
        # Back up current install
        if backup.exists():
            shutil.rmtree(backup)
        _rename_retry(install, backup)

        # Move staging to install position
        _rename_retry(staging, install)

        # Restore uninstaller files
        for src_p in unins_files:
            dst_p = install / src_p.name
            try:
                shutil.copy2(backup / src_p.name, dst_p)
            except Exception as e:
                _log.debug("Could not restore %s: %s (non-fatal)", src_p.name, e)

    except Exception as e:
        _log.error("Folder swap failed: %s — attempting rollback", e)
        _rollback(install, backup)
        _launch_old(install, "--rollback-done")
        return

    # ── Step 5: delete old sentinel and launch new version ───────────────────
    try:
        _SENTINEL_PATH.unlink(missing_ok=True)
    except Exception:
        pass

    _launch_new(install)

    # ── Step 6: wait for startup sentinel ────────────────────────────────────
    _log.info("Waiting for startup sentinel ...")
    deadline = time.monotonic() + 60.0
    sentinel_ok = False
    while time.monotonic() < deadline:
        if _SENTINEL_PATH.exists():
            sentinel_ok = True
            break
        time.sleep(0.5)

    if not sentinel_ok:
        _log.error("New version did not write startup sentinel — rolling back")
        # Kill the new version
        _kill_new(install)
        _rollback(install, backup)
        _launch_old(install, "--rollback-done")
        return

    # ── Step 7: clean up on success ───────────────────────────────────────────
    _log.info("Update successful! Cleaning up ...")
    try:
        shutil.rmtree(backup, ignore_errors=True)
    except Exception:
        pass
    try:
        zip_path.unlink(missing_ok=True)
    except Exception:
        pass
    # Staging was already moved; remove if somehow still there
    try:
        shutil.rmtree(staging, ignore_errors=True)
    except Exception:
        pass

    _log.info("win_updater: done")
    sys.exit(0)


def _launch_new(install: Path) -> None:
    exe = install / APP_EXE_NAME
    _log.info("Launching new version: %s --verify-startup --restored-from-update", exe)
    subprocess.Popen(
        [str(exe), "--verify-startup", "--restored-from-update"],
        cwd=str(install),
        close_fds=True,
        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
    )


def _launch_old(install: Path, *flags: str) -> None:
    exe = install / APP_EXE_NAME
    _log.info("Relaunching previous version: %s %s", exe, " ".join(flags))
    subprocess.Popen(
        [str(exe), *flags],
        cwd=str(install),
        close_fds=True,
        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    sys.exit(0)


def _rollback(install: Path, backup: Path) -> None:
    if not backup.exists():
        _log.error("No backup dir found at %s — cannot roll back", backup)
        return
    _log.info("Rolling back: %s → %s", backup, install)
    if install.exists():
        shutil.rmtree(install, ignore_errors=True)
    try:
        _rename_retry(backup, install)
        _log.info("Rollback complete")
    except Exception as e:
        _log.error("Rollback rename failed: %s", e)


def _kill_new(install: Path) -> None:
    try:
        import psutil
        install_str = str(install).lower()
        for proc in psutil.process_iter(["pid", "exe"]):
            try:
                exe = (proc.info["exe"] or "").lower()
                if install_str in exe:
                    proc.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except ImportError:
        pass
