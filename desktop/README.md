# GPL Platform — Desktop App

A PySide6 window that wraps both GPL servers in-process and provides auto-updating.

| Server | Port | Tab label |
|---|---|---|
| Factory (`run.py`) | 8080 | 🏭 Factory |
| Customer Runtime (`run_customer.py`) | 8081 | ⚡ Runtime |

Both servers start as uvicorn daemon threads inside the same process. Real data never leaves the machine.

---

## Run from source (dev)

```powershell
# From the repo root (gpl_agent_v29/)

# 1. Create venv
python -m venv desktop\venv
desktop\venv\Scripts\python -m pip install --upgrade pip

# 2. Install all deps (PySide6 is ~500MB — takes a few minutes)
desktop\venv\Scripts\pip install -r desktop\requirements.txt

# 3. Launch
desktop\venv\Scripts\python -m desktop.main
```

On first launch a dialog asks for your Anthropic API key (`sk-ant-...`).  
Key is saved to `%LOCALAPPDATA%\GPLPlatform\GPLPlatform\config.json`.

---

## Build a standalone .exe

```powershell
# From the repo root, with the desktop venv active

# Download wkhtmltopdf into desktop\packaging\wkhtmltopdf\ first (see CI workflow)
# then:
desktop\venv\Scripts\pyinstaller --noconfirm desktop\packaging\gpl_platform.spec

# Output: desktop\packaging\dist\GPL Platform\GPL Platform.exe

# Build installer (requires Inno Setup — jrsoftware.org/isdl.php)
iscc /Q /DMyAppVersion=1.0.0 desktop\packaging\installer.iss
# Output: desktop\packaging\dist\GPLPlatform_Setup_v1.0.0.exe
```

---

## Release a new version

1. Bump `APP_VERSION` in `desktop/version.py`
2. Commit and push to the `gpl-app` branch
3. Push a tag: `git tag v1.0.1 && git push origin v1.0.1`
4. GitHub Actions builds, packages, and publishes to [gpl-platform-releases](https://github.com/meetbadgujar14/gpl-platform-releases) automatically

Running apps check for updates 4 seconds after launch and prompt the user.

---

## Files

| File | Purpose |
|---|---|
| `main.py` | Qt entry: config → dialog → servers → loading → tabbed window |
| `server.py` | Starts factory + runtime as in-process uvicorn threads |
| `config.py` | Saves/loads API key to `%LOCALAPPDATA%` |
| `settings_dialog.py` | First-run API key dialog + File → Settings |
| `relocate.py` | Seeds writable data copy on first frozen launch |
| `updater.py` | Manifest check + download + SHA-256 verify |
| `win_updater.py` | Detached folder-swap apply + rollback |
| `version.py` | `APP_VERSION` — single source of truth |
| `requirements.txt` | All Python deps for this venv |
| `packaging/gpl_platform.spec` | PyInstaller spec (Windows onedir) |
| `packaging/installer.iss` | Inno Setup first-install script |
| `packaging/entry.py` | Frozen app entry point |
| `packaging/version_info.txt` | Windows exe metadata |

---

## GitHub Actions secret needed

Add `RELEASE_PAT` to your repo secrets — a GitHub Personal Access Token with `contents: write` permission on the `gpl-platform-releases` repo.
