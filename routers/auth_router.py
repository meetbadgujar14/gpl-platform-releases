"""
routers/auth_router.py
=======================
Runtime-side authentication.

POST /api/auth/signup  — register a new customer, save to customers.json
POST /api/auth/login   — validate credentials, return session token
POST /api/auth/logout  — invalidate token
GET  /api/auth/me      — return current session info (used on page reload)

Token is returned as a plain string. Client stores it in localStorage and
sends it on every request as:
  Header:      X-Session-Token: <token>
  Query param: ?token=<token>  (fallback for simple GETs)
"""

import json
import re
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter()

# ── In-memory session store ───────────────────────────────────────────────────
# token → { customer_id, username, display_name, created_at }
_sessions: dict = {}

# ── customers.json path ───────────────────────────────────────────────────────
_CUSTOMERS_FILE = Path(__file__).parent.parent / "customers.json"


def _load_customers() -> list:
    try:
        with open(_CUSTOMERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def _save_customers(customers: list) -> None:
    with open(_CUSTOMERS_FILE, "w", encoding="utf-8") as f:
        json.dump(customers, f, indent=2, ensure_ascii=False)


# ── Public helper — used by customer_router.py & receive_router.py ────────────

def get_customer_id(token: str) -> str:
    """
    Resolve a session token to a customer_id.
    Raises HTTP 401 if token is missing or invalid.
    """
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated — please log in.")
    session = _sessions.get(token)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired session — please log in again.")
    return session["customer_id"]


def get_session(token: str) -> dict:
    """Returns full session dict. Raises 401 if invalid."""
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    session = _sessions.get(token)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired session.")
    return session


# ── Models ────────────────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    username:     str
    display_name: str
    password:     str

class LoginRequest(BaseModel):
    username: str
    password: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/api/auth/signup")
async def signup(req: SignupRequest):
    """
    Register a new customer.
    Generates customer_id from username, saves to customers.json.
    Returns a session token so the user is immediately logged in after signup.
    """
    # Validate username — alphanumeric + underscore only
    username = req.username.strip().lower()
    if not username:
        raise HTTPException(400, "Username cannot be empty.")
    if not re.match(r'^[a-z0-9_]+$', username):
        raise HTTPException(400, "Username may only contain letters, numbers, and underscores.")
    if len(req.password) < 4:
        raise HTTPException(400, "Password must be at least 4 characters.")
    if not req.display_name.strip():
        raise HTTPException(400, "Display name cannot be empty.")

    customers = _load_customers()

    # Check username not already taken
    if any(c.get("username") == username for c in customers):
        raise HTTPException(409, f"Username '{username}' is already taken.")

    # Generate customer_id
    customer_id = f"customer_{username}"
    # Ensure customer_id is also unique (edge case)
    if any(c.get("customer_id") == customer_id for c in customers):
        raise HTTPException(409, f"Username '{username}' is already taken.")

    # Add to customers.json
    new_customer = {
        "customer_id":  customer_id,
        "username":     username,
        "password":     req.password,
        "display_name": req.display_name.strip(),
        "created_at":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    customers.append(new_customer)
    _save_customers(customers)

    # Auto-login — return session token
    token = str(uuid.uuid4())
    _sessions[token] = {
        "customer_id":  customer_id,
        "username":     username,
        "display_name": req.display_name.strip(),
        "created_at":   time.time(),
    }

    return JSONResponse({
        "token":        token,
        "customer_id":  customer_id,
        "display_name": req.display_name.strip(),
    }, status_code=201)


@router.post("/api/auth/login")
async def login(req: LoginRequest):
    customers = _load_customers()
    match = next(
        (c for c in customers
         if c.get("username") == req.username.strip().lower()
         and c.get("password") == req.password),
        None,
    )
    if not match:
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    token = str(uuid.uuid4())
    _sessions[token] = {
        "customer_id":  match["customer_id"],
        "username":     match["username"],
        "display_name": match.get("display_name", match["customer_id"]),
        "created_at":   time.time(),
    }
    return JSONResponse({
        "token":        token,
        "customer_id":  match["customer_id"],
        "display_name": match.get("display_name", match["customer_id"]),
    })


@router.post("/api/auth/logout")
async def logout(token: str = ""):
    if token and token in _sessions:
        del _sessions[token]
    return JSONResponse({"status": "logged_out"})


@router.get("/api/auth/me")
async def me(token: str = ""):
    """Used by customer.html on page load to check if already logged in."""
    session = get_session(token)
    return JSONResponse({
        "customer_id":  session["customer_id"],
        "username":     session["username"],
        "display_name": session["display_name"],
    })
