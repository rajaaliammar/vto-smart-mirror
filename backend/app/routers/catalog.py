import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Dict, List

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.schemas.garment import Garment, GarmentListResponse, GarmentUploadResponse

router = APIRouter(prefix="/garments", tags=["garments"])

BACKEND_ROOT = Path(__file__).resolve().parents[2]
GARMENTS_DIR = BACKEND_ROOT / "static" / "garments"
CATALOG_PATH = BACKEND_ROOT / "static" / "garments_catalog.json"
PROJECT_SAMPLE_ROOT = BACKEND_ROOT.parent / "assets" / "sample_clothes"
FOLDER_CATEGORIES = {
    "tshirts": "tshirt",
    "tshirt": "tshirt",
    "shirts": "tshirt",
    "pants": "pants",
    "jeans": "jeans",
    "bottoms": "pants",
    "trousers": "pants",
}

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_SIZES = ["S", "M", "L", "XL"]
DEFAULT_SCALE = 1.55


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or uuid.uuid4().hex[:8]


def _display_name(stem: str) -> str:
    return stem.replace("_", " ").replace("-", " ").title()


def _image_url(request: Request, filename: str) -> str:
    return str(request.base_url).rstrip("/") + f"/static/garments/{filename}"


def _load_catalog() -> Dict[str, dict]:
    if CATALOG_PATH.exists():
        with CATALOG_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, dict):
                return data
    return {}


def _save_catalog(catalog: Dict[str, dict]) -> None:
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CATALOG_PATH.open("w", encoding="utf-8") as handle:
        json.dump(catalog, handle, indent=2)


def _folder_category(name: str) -> str:
    key = name.lower().strip()
    return FOLDER_CATEGORIES.get(key, key.rstrip("s") or "tshirt")


def _infer_category(filename: str, fallback: str = "tshirt") -> str:
    stem = Path(filename).stem.lower()
    if "jean" in stem or "pant" in stem:
        return "jeans"
    return fallback


def _seed_from_disk() -> None:
    """Copy sample clothes into static/garments and register missing entries."""
    GARMENTS_DIR.mkdir(parents=True, exist_ok=True)
    catalog = _load_catalog()

    if PROJECT_SAMPLE_ROOT.is_dir():
        for folder in PROJECT_SAMPLE_ROOT.iterdir():
            if folder.is_dir():
                category = _folder_category(folder.name)
                for source in folder.iterdir():
                    if source.suffix.lower() not in ALLOWED_EXT:
                        continue
                    dest = GARMENTS_DIR / source.name.lower()
                    if not dest.exists():
                        shutil.copy2(source, dest)
                    garment_id = _slug(source.stem)
                    if garment_id in catalog:
                        if _infer_category(source.name, category) != "tshirt":
                            catalog[garment_id]["category"] = _infer_category(source.name, category)
                        continue
                    catalog[garment_id] = {
                        "id": garment_id,
                        "name": _display_name(source.stem),
                        "category": _infer_category(source.name, category),
                        "filename": dest.name,
                        "available_sizes": DEFAULT_SIZES,
                        "default_scale": DEFAULT_SCALE,
                    }
            elif folder.is_file() and folder.suffix.lower() in ALLOWED_EXT:
                dest = GARMENTS_DIR / folder.name.lower()
                if not dest.exists():
                    shutil.copy2(folder, dest)

    for image_path in sorted(GARMENTS_DIR.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in ALLOWED_EXT:
            continue
        garment_id = _slug(image_path.stem)
        if garment_id in catalog:
            continue
        catalog[garment_id] = {
            "id": garment_id,
            "name": _display_name(image_path.stem),
            "category": _infer_category(image_path.name),
            "filename": image_path.name,
            "available_sizes": DEFAULT_SIZES,
            "default_scale": DEFAULT_SCALE,
        }

    _save_catalog(catalog)


LOWER_CATEGORIES = {"pants", "jeans", "pant", "bottom", "trousers"}


def _slot_for(category: str, filename: str = "") -> str:
    key = str(category or "").lower()
    name = str(filename or "").lower()
    if key in LOWER_CATEGORIES or "jean" in name or "pant" in name:
        return "lower"
    return "upper"


def _to_model(entry: dict, request: Request) -> Garment:
    filename = entry["filename"]
    category = entry.get("category", "tshirt")
    return Garment(
        id=entry["id"],
        name=entry["name"],
        category=category,
        slot=_slot_for(category, filename),
        image_url=_image_url(request, filename),
        available_sizes=entry.get("available_sizes", DEFAULT_SIZES),
        default_scale=float(entry.get("default_scale", DEFAULT_SCALE)),
    )


@router.get("", response_model=GarmentListResponse)
def list_garments(request: Request):
    _seed_from_disk()
    catalog = _load_catalog()
    garments: List[Garment] = []
    for entry in catalog.values():
        image_path = GARMENTS_DIR / entry["filename"]
        if not image_path.exists():
            continue
        garments.append(_to_model(entry, request))
    garments.sort(key=lambda item: item.name.lower())
    upper = [item for item in garments if item.slot == "upper"]
    lower = [item for item in garments if item.slot == "lower"]
    return GarmentListResponse(
        count=len(garments),
        garments=garments,
        upper=upper,
        lower=lower,
    )


@router.get("/{garment_id}", response_model=Garment)
def get_garment(garment_id: str, request: Request):
    _seed_from_disk()
    catalog = _load_catalog()
    entry = catalog.get(garment_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Garment '{garment_id}' not found.")
    image_path = GARMENTS_DIR / entry["filename"]
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image for '{garment_id}' is missing.")
    return _to_model(entry, request)


@router.post("/upload", response_model=GarmentUploadResponse)
async def upload_garment(
    request: Request,
    file: UploadFile = File(...),
    name: str = Form(None),
    category: str = Form("tshirt"),
    sizes: str = Form("S,M,L,XL"),
    default_scale: float = Form(DEFAULT_SCALE),
):
    original = Path(file.filename or "garment.png").name
    suffix = Path(original).suffix.lower()
    if suffix not in ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Use PNG, JPG, or WEBP.",
        )

    GARMENTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = _slug(name or Path(original).stem)
    unique_name = f"{stem}-{uuid.uuid4().hex[:6]}{suffix}"
    dest = GARMENTS_DIR / unique_name

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    dest.write_bytes(contents)

    garment_id = _slug(stem)
    catalog = _load_catalog()
    if garment_id in catalog:
        garment_id = f"{garment_id}-{uuid.uuid4().hex[:4]}"

    size_list = [part.strip().upper() for part in sizes.split(",") if part.strip()]
    entry = {
        "id": garment_id,
        "name": name or _display_name(stem),
        "category": category or "tshirt",
        "filename": unique_name,
        "available_sizes": size_list or DEFAULT_SIZES,
        "default_scale": float(default_scale),
    }
    catalog[garment_id] = entry
    _save_catalog(catalog)

    garment = _to_model(entry, request)
    return GarmentUploadResponse(message="Garment uploaded successfully.", garment=garment)
