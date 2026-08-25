from fastapi import APIRouter, HTTPException, Request

from app.routers.catalog import _load_catalog, _seed_from_disk, _slot_for, _to_model
from app.schemas.garment import (
    ColorSwatch,
    StudioState,
    StudioUpdateRequest,
    TryOnActionResponse,
    TryOnStatusResponse,
    TryOnSwitchRequest,
    TryOnSwitchResponse,
)
from core.garment_overlay import COLOR_VARIANTS
from core.preview_stream import preview_is_live, read_preview_meta
from core.tryon_bridge import (
    read_command,
    read_studio,
    request_record,
    request_snapshot,
    request_switch,
    write_studio,
)

router = APIRouter(prefix="/tryon", tags=["tryon"])


def _swatch_hex(bgr) -> str:
    b, g, r = [int(v) for v in bgr]
    return f"#{r:02x}{g:02x}{b:02x}"


def _studio_payload() -> StudioState:
    studio = read_studio()
    swatches = [
        ColorSwatch(key=item["key"], label=item["label"], hex=_swatch_hex(item["swatch_bgr"]))
        for item in COLOR_VARIANTS
    ]
    return StudioState(
        color_key=studio.get("color_key", "original"),
        gestures_enabled=bool(studio.get("gestures_enabled", True)),
        fit_scale=float(studio.get("fit_scale", 1.55)),
        swatches=swatches,
    )


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


@router.post("/snapshot", response_model=TryOnActionResponse)
def trigger_snapshot():
    command = request_snapshot()
    return TryOnActionResponse(
        message="Snapshot countdown queued on the Smart Mirror.",
        action="snapshot",
        seq=int(command["seq"]),
    )


@router.post("/record", response_model=TryOnActionResponse)
def trigger_record():
    command = request_record()
    return TryOnActionResponse(
        message="Record toggle queued on the Smart Mirror.",
        action="record",
        seq=int(command["seq"]),
    )


@router.get("/studio", response_model=StudioState)
def get_studio():
    return _studio_payload()


@router.post("/studio", response_model=StudioState)
def update_studio(payload: StudioUpdateRequest):
    write_studio(
        color_key=payload.color_key,
        gestures_enabled=payload.gestures_enabled,
        fit_scale=payload.fit_scale,
    )
    return _studio_payload()


@router.get("/status", response_model=TryOnStatusResponse)
def tryon_status():
    command = read_command()
    meta = read_preview_meta()
    studio = read_studio()
    return TryOnStatusResponse(
        live=preview_is_live(),
        garment_id=meta.get("garment_id") or command.get("garment_id"),
        garment_name=meta.get("garment_name"),
        slot=meta.get("slot") or command.get("slot"),
        fit_size=meta.get("fit_size"),
        fit_score=meta.get("fit_score"),
        color_key=meta.get("color_key") or studio.get("color_key"),
        color_label=meta.get("color_label"),
        gestures_enabled=bool(meta.get("gestures_enabled", studio.get("gestures_enabled", True))),
        recording=bool(meta.get("recording")),
        fps=float(meta.get("fps") or 0),
        action=command.get("action"),
        applied=bool(command.get("applied")),
        seq=int(command.get("seq") or 0),
    )
