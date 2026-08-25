"""Shared live JPEG preview so FastAPI can stream the Smart Mirror output."""

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREVIEW_DIR = PROJECT_ROOT / "backend" / "static" / "live"
PREVIEW_PATH = PREVIEW_DIR / "preview.jpg"
META_PATH = PREVIEW_DIR / "preview_meta.json"
STALE_SEC = 1.5
KEEP_STALE_SEC = 4.0


def _is_lock_error(exc: BaseException) -> bool:
    if isinstance(exc, PermissionError):
        return True
    winerror = getattr(exc, "winerror", None)
    return winerror in (5, 32)  # ACCESS_DENIED / SHARING_VIOLATION


def _replace_unlocked(src: Path, dest: Path) -> None:
    """Atomic replace; skip the frame if Windows still has dest open."""
    try:
        src.replace(dest)
        return
    except PermissionError:
        pass
    except OSError as exc:
        if not _is_lock_error(exc):
            raise
    try:
        src.unlink()
    except OSError:
        pass


def open_preview_readonly(path: Path):
    """Read-only binary handle that allows other processes to replace the file."""
    if sys.platform == "win32":
        import ctypes
        import msvcrt
        from ctypes import wintypes

        generic_read = 0x80000000
        share = 0x00000001 | 0x00000002 | 0x00000004  # READ | WRITE | DELETE
        open_existing = 3
        file_attribute_normal = 0x80
        invalid = wintypes.HANDLE(-1).value

        create_file = ctypes.windll.kernel32.CreateFileW
        create_file.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        create_file.restype = wintypes.HANDLE
        handle = create_file(
            str(path),
            generic_read,
            share,
            None,
            open_existing,
            file_attribute_normal,
            None,
        )
        if handle in (None, 0, invalid, 0xFFFFFFFF):
            raise PermissionError(f"Could not open {path} with shared read")
        fd = msvcrt.open_osfhandle(int(handle), os.O_RDONLY)
        return os.fdopen(fd, "rb")

    return open(path, "rb")


def read_shared_bytes(path: Path) -> Optional[bytes]:
    try:
        with open_preview_readonly(path) as handle:
            data = handle.read()
        return data or None
    except (PermissionError, FileNotFoundError, OSError):
        return None


def publish_preview(frame, quality: int = 78, meta: Optional[dict] = None) -> None:
    if frame is None:
        return
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return
    tmp = PREVIEW_PATH.with_suffix(".jpg.tmp")
    try:
        tmp.write_bytes(buffer.tobytes())
        _replace_unlocked(tmp, PREVIEW_PATH)
    except PermissionError:
        pass
    except OSError as exc:
        if not _is_lock_error(exc):
            raise
    if meta is not None:
        payload = dict(meta)
        payload["updated_at"] = time.time()
        meta_tmp = META_PATH.with_suffix(".json.tmp")
        try:
            meta_tmp.write_text(json.dumps(payload), encoding="utf-8")
            _replace_unlocked(meta_tmp, META_PATH)
        except PermissionError:
            pass
        except OSError as exc:
            if not _is_lock_error(exc):
                raise


def read_preview_jpeg(max_age: float = KEEP_STALE_SEC) -> Optional[bytes]:
    if not PREVIEW_PATH.exists():
        return None
    try:
        age = time.time() - PREVIEW_PATH.stat().st_mtime
        if age > max_age:
            return None
        return read_shared_bytes(PREVIEW_PATH)
    except (PermissionError, OSError):
        return None


def preview_is_live(max_age: float = STALE_SEC) -> bool:
    if not PREVIEW_PATH.exists():
        return False
    try:
        return (time.time() - PREVIEW_PATH.stat().st_mtime) <= max_age
    except OSError:
        return False


def read_preview_meta() -> dict:
    if not META_PATH.exists():
        return {}
    try:
        raw = read_shared_bytes(META_PATH)
        if not raw:
            return {}
        data = json.loads(raw.decode("utf-8"))
        return data if isinstance(data, dict) else {}
    except (PermissionError, OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}


def placeholder_jpeg(message: str = "Start Smart Mirror for live overlay") -> bytes:
    canvas = np.full((720, 1280, 3), (12, 10, 16), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    title = "VTO AI Studio"
    (tw, th), _ = cv2.getTextSize(title, font, 1.4, 3)
    cv2.putText(
        canvas,
        title,
        ((1280 - tw) // 2, 300),
        font,
        1.4,
        (255, 196, 90),
        3,
        cv2.LINE_AA,
    )
    (tw, th), _ = cv2.getTextSize(message, font, 0.9, 2)
    cv2.putText(
        canvas,
        message,
        ((1280 - tw) // 2, 380),
        font,
        0.9,
        (180, 180, 200),
        2,
        cv2.LINE_AA,
    )
    ok, buffer = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return buffer.tobytes() if ok else b""
