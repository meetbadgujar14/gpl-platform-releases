"""
desktop/main.py — GPL Platform desktop app entry point.

Startup order:
  1. Intercept --apply-update (hand off to win_updater, never reaches Qt)
  2. Intercept --verify-startup (write sentinel, continue normally)
  3. Load config  →  show settings dialog if no API key
  4. apply_env()  ← MUST happen before any import of core.config
  5. relocate.activate_if_needed()
  6. server.start_both()
  7. Loading screen (polls both ports)
  8. relocate.remap_project_root()
  9. Show main window (two tabbed WebViews)
  10. 4s after window opens → background update check
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# ── Bootstrap: intercept update flags before importing Qt ────────────────────

if "--apply-update" in sys.argv:
    idx = sys.argv.index("--apply-update")
    zip_path = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
    if zip_path:
        from desktop.win_updater import apply
        apply(zip_path)
    sys.exit(1)

_VERIFY_STARTUP = "--verify-startup" in sys.argv
_ROLLBACK_DONE  = "--rollback-done"  in sys.argv

# ── Qt imports ────────────────────────────────────────────────────────────────

from PySide6.QtCore import (
    QTimer, Qt, QThread, Signal, QUrl,
)
from PySide6.QtWebEngineCore import QWebEngineProfile
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QApplication, QLabel, QMainWindow, QMessageBox,
    QProgressDialog, QTabWidget, QVBoxLayout, QWidget,
    QMenuBar, QMenu,
)
from PySide6.QtGui import QAction, QCloseEvent

# ── Internal imports ──────────────────────────────────────────────────────────

from .version import APP_VERSION

_log = logging.getLogger("gpl.main")


# ── Logging setup ─────────────────────────────────────────────────────────────

def _setup_logging() -> None:
    appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    log_dir = Path(appdata) / "GPLPlatform" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "app.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


# ── Loading screen ────────────────────────────────────────────────────────────

_LOADING_STYLE = """
QWidget { background: #0f172a; }
QLabel#title { color: #f8fafc; font-size: 26px; font-weight: 700; }
QLabel#sub   { color: #94a3b8; font-size: 14px; }
QLabel#status{ color: #64748b; font-size: 13px; }
"""


class LoadingScreen(QWidget):
    ready = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"GPL Platform {APP_VERSION}")
        self.setMinimumSize(480, 280)
        self.setStyleSheet(_LOADING_STYLE)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(14)

        title = QLabel("GPL Platform")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        sub = QLabel("Starting servers...")
        sub.setObjectName("sub")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._status = QLabel("Waiting for Factory (8080) and Runtime (8081)...")
        self._status.setObjectName("status")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(title)
        layout.addWidget(sub)
        layout.addSpacing(12)
        layout.addWidget(self._status)

        self._timer = QTimer(self)
        self._timer.setInterval(400)
        self._timer.timeout.connect(self._tick)
        self._factory_ok = False
        self._runtime_ok = False

    def start_polling(self) -> None:
        self._timer.start()

    def _tick(self) -> None:
        from . import server

        if not self._factory_ok:
            self._factory_ok = server._poll("http://127.0.0.1:8080/")
        if not self._runtime_ok:
            self._runtime_ok = server._poll("http://127.0.0.1:8081/")

        parts = []
        parts.append("✓ Factory" if self._factory_ok else "⏳ Factory (8080)")
        parts.append("✓ Runtime" if self._runtime_ok else "⏳ Runtime (8081)")
        self._status.setText("   ".join(parts))

        if self._factory_ok and self._runtime_ok:
            self._timer.stop()
            self.ready.emit()


# ── Main window ───────────────────────────────────────────────────────────────

_WINDOW_STYLE = """
QMainWindow { background: #0f172a; }
QTabWidget::pane { border: none; background: #ffffff; }
QTabBar::tab {
    background: #1e293b; color: #94a3b8;
    padding: 9px 22px; font-size: 13px; font-weight: 500;
    border: none; border-right: 1px solid #334155;
}
QTabBar::tab:selected { background: #2563eb; color: #ffffff; }
QTabBar::tab:hover:!selected { background: #334155; color: #f1f5f9; }
"""


class MainWindow(QMainWindow):
    # Signals for thread-safe UI updates from background threads
    _sig_update_found  = Signal(object)
    _sig_update_error  = Signal(str)
    _sig_no_update     = Signal()

    def __init__(self, cfg) -> None:
        super().__init__()
        self._cfg = cfg
        self.setWindowTitle(f"GPL Platform {APP_VERSION}")
        self.resize(1400, 860)
        self.setStyleSheet(_WINDOW_STYLE)
        self._build_menu()
        self._build_tabs()
        # Connect signals to slots (always runs on main thread)
        self._sig_update_found.connect(self._show_update_dialog)
        self._sig_update_error.connect(lambda msg: QMessageBox.warning(self, "Update check failed", msg))
        self._sig_no_update.connect(lambda: QMessageBox.information(
            self, "Up to date", f"You are running the latest version ({APP_VERSION})."
        ))

    def _build_menu(self) -> None:
        mb = QMenuBar(self)
        self.setMenuBar(mb)

        file_menu = QMenu("&File", self)
        mb.addMenu(file_menu)

        settings_action = QAction("⚙  Settings (API Key)", self)
        settings_action.triggered.connect(self._open_settings)
        file_menu.addAction(settings_action)

        file_menu.addSeparator()

        update_action = QAction("🔄  Check for Updates", self)
        update_action.triggered.connect(self._check_updates_manual)
        file_menu.addAction(update_action)

        file_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def _build_tabs(self) -> None:
        tabs = QTabWidget(self)
        tabs.setTabsClosable(False)
        self.setCentralWidget(tabs)

        # Factory tab
        factory_view = QWebEngineView()
        factory_view.load(QUrl("http://127.0.0.1:8080/"))
        tabs.addTab(factory_view, "🏭  Factory")

        # Runtime tab
        runtime_view = QWebEngineView()
        runtime_view.load(QUrl("http://127.0.0.1:8081/"))
        tabs.addTab(runtime_view, "⚡  Runtime")

        self._tabs = tabs

    def _open_settings(self) -> None:
        from .settings_dialog import SettingsDialog
        dlg = SettingsDialog(self._cfg, parent=self)
        if dlg.exec():
            # API key was changed — apply it and inform the user
            from .config import apply_env
            apply_env(self._cfg)
            QMessageBox.information(
                self, "Settings saved",
                "API key saved. It will be used by all new requests.\n"
                "Restart the app to apply the change to in-flight compilations."
            )

    # ── Update UI ─────────────────────────────────────────────────────────────

    def _check_updates_manual(self) -> None:
        from . import updater
        updater.check_async(
            on_update=self._on_update_found,
            on_error=self._on_update_error,
            on_no_update=self._on_no_update,
            skipped_version=self._cfg.skipped_version,
            manual=True,
        )

    def _on_no_update(self) -> None:
        self._sig_no_update.emit()

    def _on_update_found(self, info) -> None:
        self._sig_update_found.emit(info)

    def _show_update_dialog(self, info) -> None:
        from . import updater
        msg = (
            f"GPL Platform {info.version} is available.\n"
            f"You are running {APP_VERSION}.\n\n"
            f"{info.notes or 'No release notes.'}\n\n"
            "Download and install now?"
        )
        reply = QMessageBox.question(
            self, "Update available", msg,
            QMessageBox.StandardButton.Yes |
            QMessageBox.StandardButton.No |
            QMessageBox.StandardButton.Ignore,
        )
        if reply == QMessageBox.StandardButton.Ignore:
            self._cfg.skipped_version = info.version
            self._cfg.save()
        elif reply == QMessageBox.StandardButton.Yes:
            self._start_download(info)

    def _on_update_error(self, msg: str) -> None:
        self._sig_update_error.emit(msg)

    def _start_download(self, info) -> None:
        from . import updater

        dlg = QProgressDialog("Downloading update...", "Cancel", 0, 100, self)
        dlg.setWindowTitle("GPL Platform Update")
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setMinimumWidth(360)
        dlg.show()

        def on_progress(done: int, total: int) -> None:
            if total > 0:
                pct = int(done * 100 / total)
                QTimer.singleShot(0, lambda: dlg.setValue(pct))

        def on_done(zip_path) -> None:
            QTimer.singleShot(0, lambda: self._on_download_done(zip_path, dlg))

        def on_error(msg: str) -> None:
            QTimer.singleShot(0, lambda: (dlg.close(), QMessageBox.critical(self, "Download failed", msg)))

        updater.download_async(info, on_progress, on_done, on_error)

    def _on_download_done(self, zip_path, dlg) -> None:
        dlg.close()
        from . import updater
        QMessageBox.information(
            self, "Installing update",
            "GPL Platform will close and restart to install the update."
        )
        updater.apply_update_windows(zip_path)
        QApplication.quit()

    def closeEvent(self, event: QCloseEvent) -> None:
        from . import server
        server.stop_both()
        event.accept()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    _setup_logging()
    _log.info("GPL Platform %s starting", APP_VERSION)

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("GPL Platform")
    app.setOrganizationName("GPLPlatform")
    app.setApplicationVersion(APP_VERSION)

    # Write startup sentinel for the updater's rollback check
    if _VERIFY_STARTUP:
        from . import updater as _u
        _u.write_startup_sentinel()
        _log.info("startup sentinel written (--verify-startup)")

    if _ROLLBACK_DONE:
        QMessageBox.information(
            None, "Update rolled back",
            "The new version failed to start. GPL Platform has been restored to the previous version."
        )

    # Clean up any leftover updater copy from a previous run
    from . import updater as _upd
    _upd.cleanup_stale_updater()

    # ── Load config ───────────────────────────────────────────────────────────
    from .config import load_config, apply_env
    cfg = load_config()

    if not cfg.is_configured:
        from .settings_dialog import SettingsDialog
        dlg = SettingsDialog(cfg)
        if dlg.exec() != SettingsDialog.DialogCode.Accepted:
            return 0  # User quit at first-run dialog
        cfg = dlg.config

    # ── CRITICAL: set env BEFORE any server/router import ────────────────────
    apply_env(cfg)

    # ── Relocate data tree to writable location (frozen app only) ─────────────
    from . import relocate
    relocate.activate_if_needed()

    # ── Start both servers ────────────────────────────────────────────────────
    from . import server
    try:
        server.start_both()
    except Exception as e:
        QMessageBox.critical(
            None, "Failed to start servers",
            f"Could not start the GPL servers:\n\n{e}\n\n"
            "Check that ports 8080 and 8081 are free and try again."
        )
        return 1

    # ── Loading screen ────────────────────────────────────────────────────────
    loading = LoadingScreen()
    loading.show()
    loading.start_polling()

    main_win: list[MainWindow] = []

    def on_ready() -> None:
        # Remap project root now that core.paths has been imported by the servers
        relocate.remap_project_root()

        win = MainWindow(cfg)
        main_win.append(win)
        win.show()
        loading.close()

        # Write sentinel now if we didn't write it earlier (normal launch path)
        if not _VERIFY_STARTUP:
            _upd.write_startup_sentinel()

        # Auto-check disabled — use File → Check for Updates to check manually

    loading.ready.connect(on_ready)

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
