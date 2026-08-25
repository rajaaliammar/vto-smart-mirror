"""VTO Smart Mirror FastAPI entrypoint (Phase 10).

Run from the backend folder:
    uvicorn main:app --host 0.0.0.0 --port 8000

Or from the project root:
    python -m uvicorn main:app --app-dir backend --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
from pathlib import Path
import sys
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

BACKEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_ROOT.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.routers.captures import router as captures_router
from app.routers.catalog import router as catalog_router
from app.routers.tryon import router as tryon_router
from core.preview_stream import (
    PREVIEW_PATH,
    open_preview_readonly,
    placeholder_jpeg,
    preview_is_live,
    read_preview_jpeg,
    read_preview_meta,
)
from core.tryon_bridge import read_command

STATIC_DIR = BACKEND_ROOT / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "garments").mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "captures").mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "live").mkdir(parents=True, exist_ok=True)

CAPTURES_DIR = PROJECT_ROOT / "captures"
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
(CAPTURES_DIR / "videos").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)

_placeholder = placeholder_jpeg()
_waiting = placeholder_jpeg("Waiting for processed CV overlay...")
_last_jpeg = None


def _read_preview_file():
    """Open preview.jpg read-only with Windows share flags; never keep the lock."""
    try:
        if not PREVIEW_PATH.exists():
            return None
        with open_preview_readonly(PREVIEW_PATH) as handle:
            data = handle.read()
        return data or None
    except (PermissionError, FileNotFoundError, OSError):
        return None


def _next_jpeg() -> bytes:
    """Serve the pose-tracked, HUD-composited JPEG published by the OpenCV mirror."""
    global _last_jpeg
    try:
        preview = _read_preview_file() or read_preview_jpeg()
    except (PermissionError, OSError):
        preview = None
    if preview:
        _last_jpeg = preview
        return preview
    if _last_jpeg:
        return _last_jpeg
    return _waiting if preview_is_live(max_age=8.0) else _placeholder


def mjpeg_frames():
    """Yield processed OpenCV frames as multipart JPEG for <img src='/video_feed'>."""
    while True:
        jpeg = _next_jpeg()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(jpeg)).encode("ascii") + b"\r\n\r\n"
            + jpeg
            + b"\r\n"
        )
        time.sleep(0.04)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INFO] FastAPI server starting on :8000")
    yield
    print("[INFO] FastAPI server shutting down.")


app = FastAPI(
    title="VTO Smart Mirror API",
    description="Catalog, try-on control, live overlay stream, and capture API for the Virtual Try-On Smart Mirror.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(catalog_router, prefix="/api/v1")
app.include_router(tryon_router, prefix="/api/v1")
app.include_router(captures_router, prefix="/api/v1")


def _model_status() -> dict:
    pose_file = PROJECT_ROOT / "pose_landmarker.task"
    openh264 = sorted(PROJECT_ROOT.glob("openh264*.dll"))
    return {
        "pose_landmarker": {
            "name": "MediaPipe Pose Landmarker Heavy",
            "file": pose_file.name,
            "present": pose_file.is_file(),
        },
        "hands": {
            "name": "MediaPipe Hands",
            "present": True,
        },
        "openh264": {
            "file": openh264[0].name if openh264 else None,
            "present": bool(openh264),
        },
    }


@app.get("/health")
def health(request: Request):
    command = read_command()
    meta = read_preview_meta()
    return {
        "status": "ok",
        "service": "vto-smart-mirror",
        "version": app.version,
        "models": _model_status(),
        "stream": {
            "live": preview_is_live(),
            "source": "publish_preview",
        },
        "tryon": {
            "garment_id": meta.get("garment_id") or command.get("garment_id"),
            "garment_name": meta.get("garment_name"),
            "slot": command.get("slot"),
            "action": command.get("action"),
            "seq": command.get("seq", 0),
            "applied": bool(command.get("applied")),
        },
        "docs": str(request.base_url).rstrip("/") + "/docs",
    }


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        mjpeg_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="garment_static")
app.mount("/captures", StaticFiles(directory=str(CAPTURES_DIR)), name="captures")
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
