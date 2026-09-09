"""
desktop/server.py — in-process lifecycle for both GPL servers.

Starts two uvicorn daemon threads:
  - Factory  (run:app)          on port 8080
  - Runtime  (run_customer:app) on port 8081

Both are started AFTER apply_env() has set all os.environ values,
because core/config.py reads os.getenv() at class-body evaluation time
(module import). Importing any router before env is set = empty API key.

Call order from main.py:
    config.apply_env(cfg)          # set env FIRST
    server.start_both()            # import and start servers
    server.wait_healthy(...)       # poll /api/health on both ports
    server.stop_both()             # called on window close
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional

_log = logging.getLogger("gpl.server")

_factory_server = None
_runtime_server = None
_factory_thread: Optional[threading.Thread] = None
_runtime_thread: Optional[threading.Thread] = None


def _find_wkhtmltopdf() -> Optional[str]:
    """Return the path to wkhtmltopdf — bundled inside the frozen app or system-installed."""
    # When frozen by PyInstaller the binary is bundled at _internal/wkhtmltopdf/wkhtmltopdf.exe
    if getattr(sys, "frozen", False):
        bundled = Path(sys._MEIPASS) / "wkhtmltopdf" / "wkhtmltopdf.exe"
        if bundled.exists():
            return str(bundled)
    # Fall back to whatever is on PATH (dev mode or user-installed)
    import shutil
    return shutil.which("wkhtmltopdf")


def _configure_pdfkit() -> None:
    """Point pdfkit at the bundled wkhtmltopdf binary."""
    path = _find_wkhtmltopdf()
    if path:
        os.environ["WKHTMLTOPDF_PATH"] = path
        try:
            import pdfkit
            pdfkit.configuration(wkhtmltopdf=path)
            _log.info("pdfkit configured with wkhtmltopdf at: %s", path)
        except Exception as e:
            _log.warning("pdfkit configuration failed (non-fatal): %s", e)
    else:
        _log.warning(
            "wkhtmltopdf not found — PDF generation will not work. "
            "Install wkhtmltopdf or rebuild the app with the binary bundled."
        )


def _make_server(app_module: str, port: int):
    """Create a uvicorn Server for the given ASGI app module string and port."""
    import uvicorn

    config = uvicorn.Config(
        app_module,
        host="127.0.0.1",
        port=port,
        log_level="info",
        log_config=None,    # don't let uvicorn run dictConfig — we own logging
        reload=False,
    )
    srv = uvicorn.Server(config)
    srv.install_signal_handlers = lambda: None  # not on the main thread
    return srv


def start_both() -> None:
    """Import both app modules and start them on daemon threads.

    Must only be called after apply_env() has written all os.environ values.
    The imports happen here — this is intentionally the first place
    core.config (and therefore os.getenv) is touched.
    """
    global _factory_server, _runtime_server, _factory_thread, _runtime_thread

    _configure_pdfkit()

    # Add the repo root to sys.path so `import run` and `import run_customer` resolve.
    # In frozen mode PyInstaller already handles this; in dev mode we add it explicitly.
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    _log.info("Starting Factory on port 8080 ...")
    _factory_server = _make_server("run:app", 8080)
    _factory_thread = threading.Thread(
        target=_factory_server.run,
        name="gpl-factory",
        daemon=True,
    )
    _factory_thread.start()

    _log.info("Starting Customer Runtime on port 8081 ...")
    _runtime_server = _make_server("run_customer:app", 8081)
    _runtime_thread = threading.Thread(
        target=_runtime_server.run,
        name="gpl-runtime",
        daemon=True,
    )
    _runtime_thread.start()


def _poll(url: str, timeout: float = 2.0) -> bool:
    """Single HTTP GET; returns True on any 2xx response."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def wait_healthy(
    timeout: float = 60.0,
    on_tick=None,
) -> tuple[bool, bool]:
    """Poll both servers until they respond or the deadline passes.

    Returns (factory_ok, runtime_ok).
    `on_tick` is called every poll cycle — use it to pump the Qt event loop.
    """
    factory_url = "http://127.0.0.1:8080/"
    runtime_url = "http://127.0.0.1:8081/"

    factory_ok = False
    runtime_ok = False
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if on_tick:
            on_tick()
        if not factory_ok:
            factory_ok = _poll(factory_url)
        if not runtime_ok:
            runtime_ok = _poll(runtime_url)
        if factory_ok and runtime_ok:
            break
        # Check that threads are still alive
        if _factory_thread and not _factory_thread.is_alive() and not factory_ok:
            _log.error("Factory thread died before becoming healthy")
            break
        if _runtime_thread and not _runtime_thread.is_alive() and not runtime_ok:
            _log.error("Runtime thread died before becoming healthy")
            break
        time.sleep(0.4)

    return factory_ok, runtime_ok


def stop_both() -> None:
    """Signal both uvicorn servers to shut down."""
    global _factory_server, _runtime_server
    for srv in (_factory_server, _runtime_server):
        if srv is not None:
            try:
                srv.should_exit = True
            except Exception:
                pass
    _factory_server = None
    _runtime_server = None
