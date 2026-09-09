# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for GPL Platform — Windows onedir build.

Usage (from repo root, with desktop venv active):
    desktop\\venv\\Scripts\\pyinstaller --noconfirm desktop\\packaging\\gpl_platform.spec

Output:
    desktop\\packaging\\dist\\GPLPlatform\\GPLPlatform.exe
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT   = Path(SPEC).parent.parent.parent          # gpl_agent_v29/
DESKTOP_DIR = Path(SPEC).parent.parent                 # desktop/
DIST_DIR    = Path(SPEC).parent / "dist"
BUILD_DIR   = Path(SPEC).parent / "build"

# ── Collect package data ──────────────────────────────────────────────────────

datas = []

# Project tree (data, mock_data, customer_runtime, static)
for folder in ("data", "mock_data", "customer_runtime", "static"):
    src = REPO_ROOT / folder
    if src.exists():
        datas.append((str(src), folder))

# wkhtmltopdf binary (must be downloaded before building — see CI workflow)
wkhtmltopdf_src = DESKTOP_DIR / "packaging" / "wkhtmltopdf" / "wkhtmltopdf.exe"
if wkhtmltopdf_src.exists():
    datas.append((str(wkhtmltopdf_src.parent), "wkhtmltopdf"))

# Collect PySide6 WebEngine resources
pyside6_datas, pyside6_binaries, pyside6_hiddenimports = collect_all("PySide6")
datas += pyside6_datas

# ── Hidden imports ────────────────────────────────────────────────────────────

hiddenimports = [
    # uvicorn
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.loops.uvloop",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.http.httptools_impl",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    # fastapi / starlette
    "fastapi",
    "starlette",
    "starlette.routing",
    "starlette.middleware",
    "starlette.staticfiles",
    "starlette.templating",
    # data layer
    "tinydb",
    "tinydb.storages",
    "tinydb.middlewares",
    "duckdb",
    "pandas",
    "openpyxl",
    "pdfkit",
    # anthropic
    "anthropic",
    "httpx",
    # project modules
    "core",
    "core.config",
    "core.paths",
    "core.log_stream",
    "routers",
    "agents",
    "compiler",
    "customer",
    "services",
    # email-validator (pydantic optional dep)
    "email_validator",
] + pyside6_hiddenimports + collect_submodules("uvicorn")

# ── Analysis ──────────────────────────────────────────────────────────────────

a = Analysis(
    [str(DESKTOP_DIR / "packaging" / "entry.py")],
    pathex=[str(REPO_ROOT)],
    binaries=pyside6_binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Qt tools we don't need — their symlink layout can break signing
        "PyInstaller.hooks.rthooks",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="GPLPlatform",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,                          # no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(DESKTOP_DIR / "packaging" / "icon" / "icon.ico"),
    version=str(DESKTOP_DIR / "packaging" / "version_info.txt"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="GPLPlatform",
)
