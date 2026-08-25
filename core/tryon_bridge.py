"""Shared command file so FastAPI can switch the live mirror garment."""

import json
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMAND_PATH = PROJECT_ROOT / "backend" / "static" / "tryon_command.json"
STUDIO_PATH = PROJECT_ROOT / "backend" / "static" / "studio_state.json"

DEFAULT_STUDIO = {
    "color_key": "original",
    "gestures_enabled": True,
    "fit_scale": 1.55,
}


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
    return _queue_command("switch", garment_id=str(garment_id), slot=slot)


def request_snapshot() -> dict:
    return _queue_command("snapshot")


def request_record() -> dict:
    return _queue_command("record")


def _queue_command(action: str, garment_id: str = "", slot: str = "") -> dict:
    current = _read_raw()
    seq = int(current.get("seq") or 0) + 1
    payload = {
        "action": action,
        "garment_id": garment_id,
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
    if seq <= last_seq:
        return None
    if not data.get("action") and not data.get("garment_id"):
        return None
    return data


def mark_applied(seq: int, garment_id: str = "") -> None:
    data = _read_raw()
    if int(data.get("seq") or 0) != int(seq):
        return
    data["applied"] = True
    if garment_id:
        data["applied_garment_id"] = garment_id
    data["applied_at"] = time.time()
    _write_raw(data)


def read_studio() -> dict:
    if not STUDIO_PATH.exists():
        return dict(DEFAULT_STUDIO)
    try:
        data = json.loads(STUDIO_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return dict(DEFAULT_STUDIO)
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_STUDIO)
    merged = dict(DEFAULT_STUDIO)
    merged.update(data)
    return merged


def write_studio(
    color_key: Optional[str] = None,
    gestures_enabled: Optional[bool] = None,
    fit_scale: Optional[float] = None,
) -> dict:
    current = read_studio()
    if color_key is not None:
        current["color_key"] = str(color_key)
    if gestures_enabled is not None:
        current["gestures_enabled"] = bool(gestures_enabled)
    if fit_scale is not None:
        current["fit_scale"] = float(max(1.05, min(2.15, fit_scale)))
    current["updated_at"] = time.time()
    STUDIO_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STUDIO_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(current, indent=2), encoding="utf-8")
    tmp.replace(STUDIO_PATH)
    return current
