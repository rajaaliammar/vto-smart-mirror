from fastapi import APIRouter, HTTPException, Request

from app.routers.catalog import _load_catalog, _seed_from_disk, _slot_for, _to_model
from app.schemas.garment import TryOnSwitchRequest, TryOnSwitchResponse
from core.tryon_bridge import request_switch

router = APIRouter(prefix="/tryon", tags=["tryon"])


@router.post("/switch", response_model=TryOnSwitchResponse)
def switch_garment(payload: TryOnSwitchRequest, request: Request):
    """Queue a garment change for the live Smart Mirror client."""
    garment_id = str(payload.garment_id or "").strip()
    if not garment_id:
        raise HTTPException(status_code=400, detail="garment_id is required.")

    _seed_from_disk()
    catalog = _load_catalog()
    entry = catalog.get(garment_id)
    if not entry:
        lowered = garment_id.lower()
        entry = next(
            (item for item in catalog.values() if str(item.get("id", "")).lower() == lowered),
            None,
        )
    if not entry:
        raise HTTPException(status_code=404, detail=f"Garment '{garment_id}' not found.")

    slot = _slot_for(entry.get("category", "tshirt"), entry.get("filename", ""))
    command = request_switch(entry["id"], slot=slot)
    garment = _to_model(entry, request)
    return TryOnSwitchResponse(
        message=f"Queued switch to {garment.name} ({slot}).",
        garment_id=entry["id"],
        slot=slot,
        seq=int(command["seq"]),
        applied=False,
    )
