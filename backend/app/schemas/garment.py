from typing import List, Optional

from pydantic import BaseModel, Field


class Garment(BaseModel):
    id: str
    name: str
    category: str = "tshirt"
    image_url: str
    available_sizes: List[str] = Field(default_factory=lambda: ["S", "M", "L", "XL"])
    default_scale: float = 1.55


class GarmentListResponse(BaseModel):
    count: int
    garments: List[Garment]


class GarmentUploadResponse(BaseModel):
    message: str
    garment: Garment


class GarmentUploadForm(BaseModel):
    name: Optional[str] = None
    category: str = "tshirt"
    available_sizes: Optional[List[str]] = None
    default_scale: float = 1.55
