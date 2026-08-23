import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests

API_BASE_URL = "http://127.0.0.1:8000/api/v1"
LOCAL_GARMENTS_DIR = os.path.join("assets", "sample_clothes")
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
        local = self._scan_local_garments()
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
            self.garments = self._merge_catalog(garments, local)
            self.current_index = 0
            self.source = "api+local" if local else "api"
            names = ", ".join(item.get("name", item.get("id", "?")) for item in self.garments)
            print(f"[INFO] Loaded {len(self.garments)} garment(s) from {self.source}: {names}")
        except (requests.RequestException, ValueError, KeyError) as exc:
            print(
                f"[WARN] API server unreachable at {self.base_url} ({exc}). "
                f"Falling back to local files in '{self.local_dir}'."
            )
            self.garments = local
            self.current_index = 0
            self.source = "local"
            if not self.garments:
                print(f"[WARN] No local garments found in {self.local_dir}.")
            else:
                names = ", ".join(item["name"] for item in self.garments)
                print(f"[INFO] Loaded {len(self.garments)} local garment(s): {names}")

    def _scan_local_garments(self) -> List[dict]:
        garments = []
        if not os.path.isdir(self.local_dir):
            return garments
        for dirpath, _, filenames in os.walk(self.local_dir):
            category = os.path.basename(dirpath).rstrip("s") or "tshirt"
            if os.path.abspath(dirpath) == os.path.abspath(self.local_dir):
                category = "tshirt"
            for filename in sorted(filenames, key=str.lower):
                if not filename.lower().endswith(VALID_EXTS):
                    continue
                stem = os.path.splitext(filename)[0]
                path = os.path.abspath(os.path.join(dirpath, filename))
                garments.append(
                    {
                        "id": stem.lower().replace(" ", "-"),
                        "name": stem.replace("_", " ").replace("-", " ").title(),
                        "category": category,
                        "image_url": path,
                        "filename": filename.lower(),
                        "available_sizes": ["S", "M", "L", "XL"],
                        "default_scale": 1.55,
                    }
                )
        garments.sort(key=lambda item: str(item.get("name", "")).lower())
        return garments

    @staticmethod
    def _merge_catalog(api_garments: List[dict], local_garments: List[dict]) -> List[dict]:
        merged = list(api_garments)
        seen = set()
        for item in merged:
            seen.add(str(item.get("id", "")).lower())
            url = str(item.get("image_url") or "")
            seen.add(os.path.basename(urlparse(url).path).lower())
        for item in local_garments:
            filename = str(item.get("filename") or os.path.basename(item.get("image_url", "")))
            garment_id = str(item.get("id", "")).lower()
            if garment_id in seen or filename.lower() in seen:
                continue
            merged.append(item)
            seen.add(garment_id)
            seen.add(filename.lower())
        return merged

    def get_current(self) -> Optional[dict]:
        if not self.garments:
            return None
        return self.garments[self.current_index]

    def get_current_name(self) -> str:
        current = self.get_current()
        return current["name"] if current else "None"

    def get_current_category(self) -> str:
        current = self.get_current()
        return str(current.get("category", "tshirt")) if current else "tshirt"

    def get_current_size(self) -> str:
        current = self.get_current()
        if not current:
            return "M"
        sizes = current.get("available_sizes") or ["M"]
        sizes = [str(s).upper() for s in sizes]
        if "M" in sizes:
            return "M"
        return sizes[len(sizes) // 2]

    def get_current_price_label(self) -> str:
        current = self.get_current()
        if current and current.get("price_label"):
            return str(current["price_label"])
        usd, pkr = self._mock_price(current)
        return f"${usd:.2f} / PKR {pkr:,}"

    def _mock_price(self, current) -> tuple:
        presets = ((49.99, 3500), (54.99, 3850), (44.99, 3150), (59.99, 4200))
        if current and current.get("price_usd") is not None:
            usd = float(current["price_usd"])
            pkr = int(current.get("price_pkr") or round(usd * 70))
            return usd, pkr
        idx = self.current_index % len(presets)
        return presets[idx]

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
