"""
run_customer.py — GPL Customer Runtime entry point
===================================================
Runs on port 8081.  Serves only the customer router + customer UI.
The factory (run.py) stays on port 8080.

Start both:
    python run.py           # factory  → http://localhost:8080
    python run_customer.py  # runtime  → http://localhost:8081
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# ── File logging ───────────────────────────────────────────────────────────────
_LOG_DIR = Path(__file__).parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(_LOG_DIR / "customer_runtime.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

# Ensure all customer runtime directories exist before any agent runs
import core.paths  # noqa: F401 — side-effect: mkdir for seeds/, dialects/, goals/, aterms/ etc.

# Install real-time broadcast handler for the customer UI terminal
from core.log_stream import install_handler as _install_log_stream
_install_log_stream()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from routers.auth_router import router as auth_router
from routers.customer_router import router as customer_router
from routers.receive_router import router as receive_router

app = FastAPI(title="GPL Customer Runtime", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)
app.include_router(customer_router)
app.include_router(receive_router)


@app.on_event("startup")
async def _resume_atom_queue_workers():
    """
    On startup, scan all customer directories for pending atom creation queue
    entries and start a background worker for each customer that has items
    waiting.  This resumes any requests that were queued but not yet sent
    before the runtime was last shut down.
    """
    from customer.atom_queue import start_all_workers_from_disk
    from core.paths import CUSTOMER_RUNTIME_DIR
    customers_root = CUSTOMER_RUNTIME_DIR / "customers"
    try:
        start_all_workers_from_disk(customers_root)
    except Exception as e:
        logging.getLogger(__name__).warning(
            f"[startup] atom_queue worker resume failed (non-fatal): {e}"
        )

_STATIC = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_STATIC), name="static")


@app.get("/")
async def home():
    return FileResponse(os.path.join(_STATIC, "customer", "customer.html"))


@app.get("/ui/customer")
async def customer_ui():
    return FileResponse(os.path.join(_STATIC, "customer", "customer.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "run_customer:app",
        host="0.0.0.0",
        port=8081,
        reload=os.getenv("RELOAD", "true").lower() == "true",
    )
