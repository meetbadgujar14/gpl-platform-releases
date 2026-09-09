"""
desktop/updater.py — update mechanism for GPL Platform.

Flow:
  1. Check:    fetch latest.json from GitHub Releases, compare version
  2. Download: stream the ZIP to %TEMP%, verify SHA-256
  3. Apply:    hand off to win_updater via a self-copy folder-swap

GitHub Releases manifest URL:
  https://github.com/meetbadgujar14/gpl-platform-releases/releases/latest/download/latest.json

Manifest format:
  {
    "windows": {
      "version": "1.0.1",
      "url":     "https://github.com/.../releases/download/v1.0.1/GPLPlatform-1.0.1-windows.zip",
      "sha256":  "abc123...",
      "notes":   "what changed"
    }
  }
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .version import APP_VERSION

_log = logging.getLogger("gpl.updater")

MANIFEST_URL = (
    "https://github.com/meetbadgujar14/gpl-platform-releases"
    "/releases/latest/download/latest.json"
)

# Directory that holds the throwaway updater copy
_UPDATER_DIR_NAME = "updater"


# ── Data ──────────────────────────────────────────────────────────────────────

@dataclass
class UpdateInfo:
    version: str
    url: str
    sha256: str
    notes: str


# ── Version comparison ────────────────────────────────────────────────────────

def _parse(v: str) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in v.strip().lstrip("v").split("."))
    except Exception:
        return (0,)


def is_newer(remote: str, local: str = APP_VERSION) -> bool:
    return _parse(remote) > _parse(local)


# ── Check ─────────────────────────────────────────────────────────────────────

class _CheckThread(threading.Thread):
    """Background thread: fetch manifest and call back with result."""

    def __init__(
        self,
        on_update: Callable[[UpdateInfo], None],
        on_error: Callable[[str], None],
        skipped_version: str = "",
        manual: bool = False,
        on_no_update: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(name="gpl-update-check", daemon=True)
        self._on_update = on_update
        self._on_error = on_error
        self._skipped = skipped_version
        self._manual = manual
        self._on_no_update = on_no_update

    def run(self) -> None:
        try:
            import json as _json
            req = urllib.request.Request(
                MANIFEST_URL,
                headers={"User-Agent": f"GPLPlatform/{APP_VERSION}"},
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                data = _json.loads(r.read())

            info = data.get("windows", {})
            remote_ver = info.get("version", "")
            if not remote_ver:
                _log.debug("update check: no windows version in manifest")
                return

            if not is_newer(remote_ver):
                _log.debug("update check: already up to date (%s)", APP_VERSION)
                if self._manual and self._on_no_update:
                    self._on_no_update()
                return

            # On auto-check, skip if user dismissed this version
            if not self._manual and remote_ver == self._skipped:
                _log.debug("update check: user skipped %s", remote_ver)
                return

            self._on_update(UpdateInfo(
                version=remote_ver,
                url=info.get("url", ""),
                sha256=info.get("sha256", ""),
                notes=info.get("notes", ""),
            ))

        except Exception as e:
            _log.debug("update check failed: %s", e)
            if self._manual:
                self._on_error(f"Could not reach GitHub to check for updates.\n\nError: {e}")


def check_async(
    on_update: Callable[[UpdateInfo], None],
    on_error: Callable[[str], None],
    skipped_version: str = "",
    manual: bool = False,
    on_no_update: Optional[Callable[[], None]] = None,
) -> None:
    """Start a background update check. Non-blocking."""
    _CheckThread(on_update, on_error, skipped_version, manual, on_no_update).start()


# ── Download + verify ─────────────────────────────────────────────────────────

class _DownloadThread(threading.Thread):
    """Stream the update ZIP to %TEMP%, verify SHA-256, then call back."""

    def __init__(
        self,
        info: UpdateInfo,
        on_progress: Callable[[int, int], None],   # (bytes_done, total_bytes)
        on_done: Callable[[Path], None],
        on_error: Callable[[str], None],
    ) -> None:
        super().__init__(name="gpl-update-download", daemon=True)
        self._info = info
        self._on_progress = on_progress
        self._on_done = on_done
        self._on_error = on_error

    def run(self) -> None:
        tmp = None
        try:
            import httpx

            fd, tmp_path = tempfile.mkstemp(suffix=".zip", prefix="GPLPlatform-update-")
            os.close(fd)
            tmp = Path(tmp_path)

            sha = hashlib.sha256()
            done = 0
            total = 0

            # httpx follows redirects automatically (GitHub releases redirect to CDN)
            with httpx.stream(
                "GET",
                self._info.url,
                follow_redirects=True,
                timeout=60.0,
                headers={"User-Agent": f"GPLPlatform/{APP_VERSION}"},
            ) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                with tmp.open("wb") as f:
                    for chunk in r.iter_bytes(chunk_size=65536):
                        f.write(chunk)
                        sha.update(chunk)
                        done += len(chunk)
                        self._on_progress(done, total)

            digest = sha.hexdigest()
            if self._info.sha256 and digest.lower() != self._info.sha256.lower():
                _log.error(
                    "SHA-256 mismatch: got %s expected %s", digest, self._info.sha256
                )
                tmp.unlink(missing_ok=True)
                self._on_error(
                    "Download verification failed (SHA-256 mismatch).\n"
                    "Please try again."
                )
                return

            _log.info("Download verified: %s", tmp)
            self._on_done(tmp)

        except Exception as e:
            _log.error("Download failed: %s", e)
            if tmp and tmp.exists():
                tmp.unlink(missing_ok=True)
            self._on_error(f"Download failed:\n{e}")


def download_async(
    info: UpdateInfo,
    on_progress: Callable[[int, int], None],
    on_done: Callable[[Path], None],
    on_error: Callable[[str], None],
) -> None:
    """Start background download. Non-blocking."""
    _DownloadThread(info, on_progress, on_done, on_error).start()


# ── Apply (Windows) ───────────────────────────────────────────────────────────

def _install_dir() -> Path:
    """Return the directory containing the running exe (frozen) or this file (dev)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


def _local_appdata() -> Path:
    return Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))


def apply_update_windows(zip_path: Path) -> None:
    """Stage a throwaway copy of ourselves and spawn it to do the folder-swap.

    We cannot replace a running exe on Windows, so:
      1. Copy GPL Platform.exe + _internal to %LOCALAPPDATA%\\GPLPlatform\\updater\\
      2. Spawn that copy with --apply-update <zip> (detached, new process group)
      3. Quit the main app

    win_updater.py intercepts --apply-update before Qt starts.
    """
    install = _install_dir()
    updater_dir = _local_appdata() / "GPLPlatform" / _UPDATER_DIR_NAME

    exe_name = "GPLPlatform.exe"
    src_exe = install / exe_name
    if not src_exe.exists():
        # Dev mode — nothing to swap
        _log.warning("apply_update: not frozen, skipping folder-swap")
        return

    # Remove stale updater dir from a previous run
    if updater_dir.exists():
        shutil.rmtree(updater_dir, ignore_errors=True)
    updater_dir.mkdir(parents=True, exist_ok=True)

    # Copy exe
    updater_exe = updater_dir / exe_name
    shutil.copy2(src_exe, updater_exe)

    # Copy _internal (Python runtime, Qt, all packages)
    internal_src = install / "_internal"
    if internal_src.exists():
        shutil.copytree(internal_src, updater_dir / "_internal")

    # Set env so win_updater knows where the real install is
    env = os.environ.copy()
    env["GPL_UPDATE_APP_DIR"] = str(install)

    _log.info("Spawning updater copy: %s --apply-update %s", updater_exe, zip_path)

    subprocess.Popen(
        [str(updater_exe), "--apply-update", str(zip_path)],
        env=env,
        cwd=str(updater_dir),
        close_fds=True,
        creationflags=(
            subprocess.DETACHED_PROCESS |
            subprocess.CREATE_NEW_PROCESS_GROUP |
            subprocess.CREATE_NO_WINDOW
        ),
    )


# ── Startup sentinel ──────────────────────────────────────────────────────────

def _sentinel_dir() -> Path:
    appdata = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
    return appdata / "GPLPlatform" / "update"


def write_startup_sentinel() -> None:
    """Write the file win_updater polls for to confirm the new build is healthy."""
    d = _sentinel_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / "startup_ok").write_text(APP_VERSION)
    _log.debug("startup sentinel written")


def cleanup_stale_updater() -> None:
    """Delete a leftover updater copy from a previous self-update run."""
    updater_dir = _local_appdata() / "GPLPlatform" / _UPDATER_DIR_NAME
    if updater_dir.exists():
        try:
            shutil.rmtree(updater_dir)
            _log.debug("cleaned up stale updater dir")
        except Exception as e:
            _log.debug("cleanup stale updater (non-fatal): %s", e)
