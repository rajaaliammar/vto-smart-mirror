"""VTO Smart Mirror FastAPI entrypoint (Phase 10).

Run from the backend folder:
    uvicorn main:app --host 0.0.0.0 --port 8000

Or from the project root:
    python -m uvicorn main:app --app-dir backend --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
from pathlib import Path
import sys

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
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
from core.tryon_bridge import read_command

STATIC_DIR = BACKEND_ROOT / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "garments").mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "captures").mkdir(parents=True, exist_ok=True)

CAPTURES_DIR = PROJECT_ROOT / "captures"
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
(CAPTURES_DIR / "videos").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INFO] FastAPI server starting on :8000")
    yield
    print("[INFO] FastAPI server shutting down.")


app = FastAPI(
    title="VTO Smart Mirror API",
    description="Catalog, try-on control, and capture API for the Virtual Try-On Smart Mirror.",
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
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/captures", StaticFiles(directory=str(CAPTURES_DIR)), name="captures")


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
    return {
        "status": "ok",
        "service": "vto-smart-mirror",
        "version": app.version,
        "models": _model_status(),
        "tryon": {
            "garment_id": command.get("garment_id"),
            "slot": command.get("slot"),
            "seq": command.get("seq", 0),
            "applied": bool(command.get("applied")),
        },
        "docs": str(request.base_url).rstrip("/") + "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
