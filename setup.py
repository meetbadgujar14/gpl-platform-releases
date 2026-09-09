"""
setup.py
========
GPL Agents — automated setup script.

Run once before starting:
    python setup.py

What it does:
  1. Checks Python version (3.8+ required)
  2. Creates a virtual environment (venv/)
  3. Upgrades pip
  4. Installs all dependencies from requirements.txt
  5. Verifies installation
  6. Checks .env for API key
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def step(n, text):
    print(f"\n[Step {n}] {text}")


def run_cmd(cmd, desc):
    print(f"  → {desc}...")
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"  ✓ {desc} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ {desc} failed")
        if e.stderr:
            print(f"  Error: {e.stderr.strip()}")
        return False


def main():
    header("GPL Agents — Setup")

    root       = Path(__file__).parent
    os.chdir(root)
    is_windows = platform.system() == "Windows"
    python_cmd = "python" if is_windows else "python3"
    venv_dir   = "venv"

    # Step 1 — Python version
    step(1, "Checking Python version")
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 8):
        print(f"  ✗ Python 3.8+ required. Found: {v.major}.{v.minor}.{v.micro}")
        sys.exit(1)
    print(f"  ✓ Python {v.major}.{v.minor}.{v.micro}")

    # Step 2 — Create venv
    step(2, "Creating virtual environment")
    if not run_cmd(f"{python_cmd} -m venv {venv_dir}", "Create venv"):
        sys.exit(1)

    # Paths inside venv
    if is_windows:
        pip    = os.path.join(venv_dir, "Scripts", "pip.exe")
        python = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        pip    = os.path.join(venv_dir, "bin", "pip")
        python = os.path.join(venv_dir, "bin", "python")

    # Step 3 — Upgrade pip
    step(3, "Upgrading pip")
    if not run_cmd(f"{python} -m pip install --upgrade pip", "Upgrade pip"):
        sys.exit(1)

    # Step 4 — Install dependencies
    step(4, "Installing dependencies")
    if not run_cmd(f"{pip} install -r requirements.txt", "Install dependencies"):
        sys.exit(1)

    # Step 5 — Verify
    step(5, "Verifying installation")
    packages = ["fastapi", "uvicorn", "anthropic", "tinydb", "python-dotenv", "duckdb", "pandas"]
    all_ok = True
    for pkg in packages:
        r = subprocess.run(f"{pip} show {pkg}", shell=True, capture_output=True, text=True)
        if r.returncode == 0:
            print(f"  ✓ {pkg}")
        else:
            print(f"  ✗ {pkg} NOT installed")
            all_ok = False
    if not all_ok:
        print("\n  ✗ Some packages failed. Check errors above.")
        sys.exit(1)

    # Step 6 — Check .env
    step(6, "Checking configuration")
    env_file = root / ".env"
    if env_file.exists():
        print("  ✓ .env file found")
        try:
            content = env_file.read_text(encoding="utf-8")
            if "ANTHROPIC_API_KEY=" in content and "sk-ant-" in content:
                print("  ✓ Anthropic API key configured")
            else:
                print("  ⚠ ANTHROPIC_API_KEY not found in .env")
                print("    Edit .env and add: ANTHROPIC_API_KEY=sk-ant-...")
        except Exception:
            print("  ⚠ Could not read .env — check it manually")
    else:
        print("  ✗ .env file not found")
        print("    Create .env with: ANTHROPIC_API_KEY=sk-ant-...")

    # Done
    header("Setup complete!")
    print("\nNext steps:")
    print("  1. Make sure ANTHROPIC_API_KEY is set in .env")
    print("  2. Start the server:")
    if is_windows:
        print("     • Double-click start.bat")
        print(f"     • OR run: {python} run.py")
    else:
        print("     • Run: ./start.sh")
        print(f"     • OR run: {python} run.py")
    print("\n  3. Open http://localhost:8080 in your browser")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
