"""
core/job_manager.py
====================
Simple in-memory job manager for long-running pipeline steps.
Each job has: id, status, progress, result, error.
Status: pending → running → done | failed
"""
import uuid
import time
import threading
from typing import Dict, Any, Optional

_jobs: Dict[str, Dict] = {}
_lock = threading.Lock()


def create_job(label: str) -> str:
    """Create a new job and return its ID."""
    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = {
            "id":         job_id,
            "label":      label,
            "status":     "pending",
            "progress":   "",
            "result":     None,
            "error":      None,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
    return job_id


def update_job(job_id: str, **kwargs):
    """Update job fields."""
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)
            _jobs[job_id]["updated_at"] = time.time()


def get_job(job_id: str) -> Optional[Dict]:
    with _lock:
        return _jobs.get(job_id)


def run_job(job_id: str, fn, *args, **kwargs):
    """Run fn(*args, **kwargs) in a background thread, updating job state."""
    def _run():
        update_job(job_id, status="running")
        try:
            result = fn(*args, **kwargs)
            update_job(job_id, status="done", result=result)
        except Exception as e:
            update_job(job_id, status="failed", error=str(e))
    t = threading.Thread(target=_run, daemon=True)
    t.start()
