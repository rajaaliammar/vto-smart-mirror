"""Compatibility shim so `uvicorn app.main:app` still works."""

import sys
from pathlib import Path

_backend_root = Path(__file__).resolve().parents[1]
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

from main import app  # noqa: E402  (canonical app lives in backend/main.py)

__all__ = ["app"]
