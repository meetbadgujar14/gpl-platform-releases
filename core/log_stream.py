"""
core/log_stream.py
==================
Real-time log streaming via Server-Sent Events.
Captures all Python logging calls and broadcasts to connected browser clients.
"""
import logging
import queue
import threading
import time
from typing import Generator

# Global broadcast queue — all log records go here
_queues: list[queue.Queue] = []
_lock = threading.Lock()


class _BroadcastHandler(logging.Handler):
    """Logging handler that puts records into all connected client queues."""

    LEVEL_CLASS = {
        logging.DEBUG:    "dim",
        logging.INFO:     "info",
        logging.WARNING:  "warn",
        logging.ERROR:    "error",
        logging.CRITICAL: "error",
    }

    def emit(self, record: logging.LogRecord):
        try:
            level_cls = self.LEVEL_CLASS.get(record.levelno, "info")
            # Shorten logger name for readability
            name = record.name.split(".")[-1] if "." in record.name else record.name
            msg  = self.format(record)
            # Strip ANSI color codes if any
            import re
            msg = re.sub(r'\x1b\[[0-9;]*m', '', msg)
            payload = f"data: {level_cls}|{name}|{msg}\n\n"
            with _lock:
                dead = []
                for q in _queues:
                    try:
                        q.put_nowait(payload)
                    except queue.Full:
                        dead.append(q)
                for q in dead:
                    _queues.remove(q)
        except Exception:
            pass


def install_handler():
    """Install the broadcast handler on the root logger. Call once at startup."""
    handler = _BroadcastHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s"))
    root = logging.getLogger()
    # Avoid adding twice
    for h in root.handlers:
        if isinstance(h, _BroadcastHandler):
            return
    root.addHandler(handler)


def event_stream() -> Generator[str, None, None]:
    """
    Generator that yields SSE-formatted log lines.
    Each client gets its own queue.
    """
    client_q: queue.Queue = queue.Queue(maxsize=500)
    with _lock:
        _queues.append(client_q)

    # Send a hello event so client knows connection is live
    yield "data: info|system|Log stream connected\n\n"

    try:
        while True:
            try:
                msg = client_q.get(timeout=15)
                yield msg
            except queue.Empty:
                # Send keepalive comment every 15s
                yield ": keepalive\n\n"
    finally:
        with _lock:
            if client_q in _queues:
                _queues.remove(client_q)
