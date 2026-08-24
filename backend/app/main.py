from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routers.catalog import router as catalog_router

BACKEND_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = BACKEND_ROOT / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "garments").mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "captures").mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="VTO Smart Mirror API",
    description="Garment catalog and asset API for the Virtual Try-On desktop client.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000",
    ],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(catalog_router, prefix="/api/v1")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

PROJECT_ROOT = BACKEND_ROOT.parent
CAPTURES_DIR = PROJECT_ROOT / "captures"
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/captures", StaticFiles(directory=str(CAPTURES_DIR)), name="captures")


@app.get("/health")
def health():
    return {"status": "ok", "service": "vto-smart-mirror"}
