"""
desktop/settings_dialog.py — API key entry dialog.

Shows on first launch when no key is saved.
Also accessible via File → Settings at any time.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QVBoxLayout,
)

from .config import Config

_STYLE = """
QDialog { background: #ffffff; }
QLabel  { color: #0f172a; font-size: 13px; }
#title  { color: #0f172a; font-size: 20px; font-weight: 600; }
#hint   { color: #64748b; font-size: 12px; }

/* Always pin fg+bg together — dark-mode Qt would render white-on-white otherwise */
QLineEdit {
    padding: 11px 12px;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    background: #ffffff;
    color: #0f172a;
    font-size: 14px;
    selection-background-color: #2563eb;
    selection-color: #ffffff;
}
QLineEdit:focus { border: 1px solid #2563eb; }

#reveal {
    background: #f1f5f9; color: #334155;
    border: 1px solid #cbd5e1; border-radius: 8px;
    padding: 11px 14px; font-size: 13px;
}
#reveal:hover { background: #e2e8f0; }

#save {
    background: #2563eb; color: #ffffff;
    border: none; border-radius: 8px;
    padding: 11px 26px; font-weight: 600; font-size: 14px;
}
#save:hover  { background: #1d4ed8; }
#cancel { background: transparent; color: #475569; border: none; padding: 11px 16px; font-size: 14px; }
#cancel:hover { color: #0f172a; }
"""


class SettingsDialog(QDialog):
    """Single-field dialog: ask for the Anthropic API key."""

    def __init__(self, cfg: Config, parent=None) -> None:
        super().__init__(parent)
        self._cfg = cfg
        self._first_run = not cfg.is_configured
        self.setWindowTitle("GPL Platform")
        self.setMinimumWidth(480)
        self.setStyleSheet(_STYLE)
        self._build()

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(32, 28, 32, 26)
        outer.setSpacing(14)

        title = QLabel("GPL Platform")
        title.setObjectName("title")
        outer.addWidget(title)

        outer.addWidget(QLabel("Enter your Anthropic API key to get started."))

        self._key_field = QLineEdit(self._cfg.api_key)
        self._key_field.setEchoMode(QLineEdit.EchoMode.Password)
        self._key_field.setPlaceholderText("sk-ant-...")
        self._key_field.setClearButtonEnabled(True)

        self._reveal_btn = QPushButton("Show")
        self._reveal_btn.setObjectName("reveal")
        self._reveal_btn.setCheckable(True)
        self._reveal_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._reveal_btn.toggled.connect(self._toggle_reveal)

        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(self._key_field)
        row.addWidget(self._reveal_btn)
        outer.addLayout(row)

        hint = QLabel(
            "Your key is stored locally and never sent anywhere except Anthropic's API."
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        outer.addWidget(hint)

        btns = QHBoxLayout()
        btns.addStretch(1)

        cancel = QPushButton("Quit" if self._first_run else "Cancel")
        cancel.setObjectName("cancel")
        cancel.clicked.connect(self.reject)

        save = QPushButton("Save and Continue" if self._first_run else "Save")
        save.setObjectName("save")
        save.setDefault(True)
        save.clicked.connect(self._on_save)

        btns.addWidget(cancel)
        btns.addWidget(save)
        outer.addSpacing(4)
        outer.addLayout(btns)

    def _toggle_reveal(self, on: bool) -> None:
        self._key_field.setEchoMode(
            QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password
        )
        self._reveal_btn.setText("Hide" if on else "Show")

    def _on_save(self) -> None:
        key = self._key_field.text().strip()
        if not key:
            QMessageBox.warning(
                self, "API key required",
                "An Anthropic API key is required to run GPL Platform."
            )
            return
        if not key.startswith("sk-"):
            QMessageBox.warning(
                self, "Invalid key",
                "Anthropic API keys start with 'sk-'. Please check your key."
            )
            return
        self._cfg.api_key = key
        self._cfg.save()
        self.accept()

    @property
    def config(self) -> Config:
        return self._cfg
