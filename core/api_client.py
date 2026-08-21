import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests

API_BASE_URL = "http://127.0.0.1:8000/api/v1"
LOCAL_GARMENTS_DIR = os.path.join("assets", "sample_clothes", "tshirts")
VALID_EXTS = (".png", ".jpg", ".jpeg", ".webp")


class GarmentApiClient:
    """Fetches the remote garment catalog, with a local assets/ fallback."""

    def __init__(
        self,
        base_url: str = API_BASE_URL,
        local_dir: str = LOCAL_GARMENTS_DIR,
        timeout: float = 2.5,
    ):
        self.base_url = base_url.rstrip("/")
        self.local_dir = os.path.abspath(local_dir)
        self.timeout = timeout
        self.garments: List[dict] = []
        self.current_index = 0
        self.source = "none"
        self.cache_dir = tempfile.mkdtemp(prefix="vto_garments_")
        self._path_cache: Dict[str, str] = {}
        self.refresh()

    def refresh(self) -> None:
        try:
            response = requests.get(
                f"{self.base_url}/garments",
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            garments = payload.get("garments") or []
            if not garments:
                raise ValueError("API returned an empty catalog")
            self.garments = garments
            self.current_index = 0
            self.source = "api"
            names = ", ".join(item.get("name", item.get("id", "?")) for item in garments)
            print(f"[INFO] Loaded {len(garments)} garment(s) from API: {names}")
        except (requests.RequestException, ValueError, KeyError) as exc:
            print(
                f"[WARN] API server unreachable at {self.base_url} ({exc}). "
                f"Falling back to local files in '{self.local_dir}'."
            )
            self._load_local_fallback()

    def _load_local_fallback(self) -> None:
        self.garments = []
        self.source = "local"
        if os.path.isdir(self.local_dir):
            for filename in sorted(os.listdir(self.local_dir)):
                if not filename.lower().endswith(VALID_EXTS):
                    continue
                stem = os.path.splitext(filename)[0]
                path = os.path.abspath(os.path.join(self.local_dir, filename))
                self.garments.append(
                    {
                        "id": stem.lower().replace(" ", "-"),
                        "name": stem.replace("_", " ").replace("-", " ").title(),
                        "category": "tshirt",
                        "image_url": path,
                        "available_sizes": ["S", "M", "L", "XL"],
                        "default_scale": 1.55,
                    }
                )
        if not self.garments:
            print(f"[WARN] No local garments found in {self.local_dir}.")
        else:
            names = ", ".join(item["name"] for item in self.garments)
            print(f"[INFO] Loaded {len(self.garments)} local garment(s): {names}")

    def get_current(self) -> Optional[dict]:
        if not self.garments:
            return None
        return self.garments[self.current_index]

    def get_current_name(self) -> str:
        current = self.get_current()
        return current["name"] if current else "None"

    def get_current_image_path(self) -> str:
        current = self.get_current()
        if not current:
            return ""
        return self._cache_image(current)

    def next_garment(self) -> str:
        if self.garments:
            self.current_index = (self.current_index + 1) % len(self.garments)
        return self.get_current_image_path()

    def prev_garment(self) -> str:
        if self.garments:
            self.current_index = (self.current_index - 1) % len(self.garments)
        return self.get_current_image_path()

    def _cache_image(self, garment: dict) -> str:
        garment_id = str(garment.get("id") or garment.get("name") or "garment")
        cached = self._path_cache.get(garment_id)
        if cached and os.path.isfile(cached):
            return cached

        location = garment.get("image_url") or ""
        if not location:
            return ""

        if os.path.isfile(location):
            self._path_cache[garment_id] = location
            return location

        if location.startswith("http://") or location.startswith("https://"):
            try:
                response = requests.get(location, timeout=self.timeout)
                response.raise_for_status()
            except requests.RequestException as exc:
                print(f"[WARN] Failed to download garment image ({exc}).")
                return ""

            suffix = Path(urlparse(location).path).suffix.lower() or ".png"
            dest = os.path.join(self.cache_dir, f"{garment_id}{suffix}")
            with open(dest, "wb") as handle:
                handle.write(response.content)
            self._path_cache[garment_id] = dest
            return dest

        print(f"[WARN] Unrecognized garment location: {location}")
        return ""
