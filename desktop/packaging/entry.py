"""
desktop/packaging/entry.py — frozen app entry point.

PyInstaller calls this file. We fix sys.path so that:
  - `import run` and `import run_customer` resolve to the bundled server modules
  - `from desktop.main import main` resolves correctly

Then we hand off to desktop.main.main().
"""

import sys
import os
from pathlib import Path

# In the frozen bundle, sys._MEIPASS is the _internal directory.
# The repo root (with run.py, run_customer.py, core/, routers/, etc.)
# is bundled there.
if getattr(sys, "frozen", False):
    bundle = Path(sys._MEIPASS)
    if str(bundle) not in sys.path:
        sys.path.insert(0, str(bundle))

from desktop.main import main

if __name__ == "__main__":
    sys.exit(main())
