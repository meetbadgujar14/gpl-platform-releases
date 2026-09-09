"""Single source of truth for the GPL Platform desktop app version.

Bump this before every release. Consumed by:
  - main.py (window title, startup sentinel)
  - updater.py (version comparison)
  - packaging/gpl_platform.spec (exe metadata)
  - CI / Inno Setup (installer version)
"""

APP_VERSION = "1.0.0"
