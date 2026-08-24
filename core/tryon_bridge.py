"""Shared command file so FastAPI can switch the live mirror garment."""

import json
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMAND_PATH = PROJECT_ROOT / "backend" / "static" / "tryon_command.json"


def _read_raw() -> dict:
    if not COMMAND_PATH.exists():
        return {}
    try:
        data = json.loads(COMMAND_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_raw(payload: dict) -> None:
    COMMAND_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = COMMAND_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(COMMAND_PATH)


def request_switch(garment_id: str, slot: str = "upper") -> dict:
    current = _read_raw()
    seq = int(current.get("seq") or 0) + 1
    payload = {
        "garment_id": str(garment_id),
        "slot": slot,
        "seq": seq,
        "applied": False,
        "requested_at": time.time(),
    }
    _write_raw(payload)
    return payload


def read_command() -> dict:
    return _read_raw()


def consume_if_new(last_seq: int) -> Optional[dict]:
    data = _read_raw()
    seq = int(data.get("seq") or 0)
    if seq <= last_seq or not data.get("garment_id"):
        return None
    return data


def mark_applied(seq: int, garment_id: str) -> None:
    data = _read_raw()
    if int(data.get("seq") or 0) != int(seq):
        return
    data["applied"] = True
    data["applied_garment_id"] = garment_id
    data["applied_at"] = time.time()
    _write_raw(data)
