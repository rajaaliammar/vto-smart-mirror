from typing import List, Optional

from pydantic import BaseModel, Field


class Garment(BaseModel):
    id: str
    name: str
    category: str = "tshirt"
    slot: str = "upper"
    image_url: str
    available_sizes: List[str] = Field(default_factory=lambda: ["S", "M", "L", "XL"])
    default_scale: float = 1.55


class GarmentListResponse(BaseModel):
    count: int
    garments: List[Garment]
    upper: List[Garment] = Field(default_factory=list)
    lower: List[Garment] = Field(default_factory=list)


class GarmentUploadResponse(BaseModel):
    message: str
    garment: Garment


class GarmentUploadForm(BaseModel):
    name: Optional[str] = None
    category: str = "tshirt"
    available_sizes: Optional[List[str]] = None
    default_scale: float = 1.55


class TryOnSwitchRequest(BaseModel):
    garment_id: str


class TryOnSwitchResponse(BaseModel):
    message: str
    garment_id: str
    slot: str
    seq: int
    applied: bool = False


class TryOnActionResponse(BaseModel):
    message: str
    action: str
    seq: int
    applied: bool = False


class ColorSwatch(BaseModel):
    key: str
    label: str
    hex: str = "#c6c6c6"


class StudioState(BaseModel):
    color_key: str = "original"
    gestures_enabled: bool = True
    fit_scale: float = 1.55
    swatches: List[ColorSwatch] = Field(default_factory=list)


class StudioUpdateRequest(BaseModel):
    color_key: Optional[str] = None
    gestures_enabled: Optional[bool] = None
    fit_scale: Optional[float] = None


class TryOnStatusResponse(BaseModel):
    live: bool = False
    garment_id: Optional[str] = None
    garment_name: Optional[str] = None
    slot: Optional[str] = None
    fit_size: Optional[str] = None
    fit_score: Optional[int] = None
    color_key: Optional[str] = None
    color_label: Optional[str] = None
    gestures_enabled: bool = True
    recording: bool = False
    fps: float = 0.0
    action: Optional[str] = None
    applied: bool = False
    seq: int = 0


class CaptureItem(BaseModel):
    filename: str
    kind: str
    url: str


class CaptureListResponse(BaseModel):
    count: int
    snapshots: List[CaptureItem] = Field(default_factory=list)
    videos: List[CaptureItem] = Field(default_factory=list)
