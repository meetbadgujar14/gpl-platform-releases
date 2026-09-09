"""
run.py — GPL Agents entry point

Start:
    python run.py
"""
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# ── File logging ──────────────────────────────────────────────────────────────
_LOG_DIR = Path(__file__).parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(_LOG_DIR / "factory.log", encoding="utf-8"),
        logging.StreamHandler(),
    ]
)

# Install real-time broadcast handler for UI terminal
from core.log_stream import install_handler as _install_log_stream
_install_log_stream()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from routers.agents_router import router
from routers.customer_onboard_router import router as onboard_router

app = FastAPI(title="GPL Agents", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router)
app.include_router(onboard_router)

_STATIC = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_STATIC), name="static")


@app.get("/")
async def home():
    return FileResponse(os.path.join(_STATIC, "agents", "agents.html"))


@app.get("/ui/agents")
async def agents_ui():
    return FileResponse(os.path.join(_STATIC, "agents", "agents.html"))


if __name__ == "__main__":
    import uvicorn
    from core.config import settings
    uvicorn.run("run:app", host=settings.HOST, port=settings.PORT, reload=settings.RELOAD)


@app.get("/ui/customer")
async def customer_ui():
    return FileResponse(os.path.join(_STATIC, "customer", "customer.html"))
