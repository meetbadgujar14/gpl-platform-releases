"""
desktop/config.py — settings persistence for GPL Platform desktop app.

Only the ANTHROPIC_API_KEY is user-entered and saved.
All other env vars (model, ports, mock row counts) are hardcoded defaults
set into os.environ by apply_env() before either server imports anything.

Config file location:
  Windows: %LOCALAPPDATA%\\GPLPlatform\\GPLPlatform\\config.json
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from PySide6.QtCore import QStandardPaths

# ── Hard-coded defaults (never stored, always set fresh on launch) ────────────
_DEFAULTS = {
    "ANTHROPIC_MODEL":      "claude-sonnet-4-6",
    "MOCK_ROWS_DIMENSION":  "15",
    "MOCK_ROWS_RECORD":     "50",
    "MOCK_ROWS_STATE":      "80",
    "MOCK_ROWS_SNAPSHOT":   "50",
    "MOCK_ROWS_EVENT":      "50",
    "HOST":                 "0.0.0.0",
    "PORT":                 "8080",
    "RELOAD":               "false",          # never hot-reload in the frozen app
    "FACTORY_URL":          "http://localhost:8080",
    "RUNTIME_CALLBACK_URL": "http://localhost:8081",
}


def _config_path() -> Path:
    base = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppConfigLocation
    )
    root = Path(base) if base else (Path.home() / ".config" / "GPLPlatform")
    return root / "config.json"


@dataclass
class Config:
    api_key: str = ""
    skipped_version: str = ""

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key.strip())

    def save(self) -> None:
        path = _config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        try:
            path.chmod(0o600)
        except Exception:
            pass


def load_config() -> Config:
    """Load persisted settings. Returns defaults if file doesn't exist yet."""
    path = _config_path()
    if path.exists():
        try:
            data = json.loads(path.read_text())
            return Config(
                api_key=data.get("api_key", ""),
                skipped_version=data.get("skipped_version", ""),
            )
        except Exception:
            pass
    return Config()


def apply_env(cfg: Config) -> None:
    """Set ALL required env vars into os.environ.

    Must be called BEFORE any import of core.config or any router,
    because core/config.py reads os.getenv() at class-body evaluation time.
    """
    # The one user-supplied value
    os.environ["ANTHROPIC_API_KEY"] = cfg.api_key

    # Hard-coded defaults — only set if not already in the environment
    # so a developer can still override them from outside
    for key, value in _DEFAULTS.items():
        os.environ.setdefault(key, value)
