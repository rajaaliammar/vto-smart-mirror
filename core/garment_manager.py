import os


class GarmentManager:

    def __init__(self, tshirts_dir: str = "assets/sample_clothes/tshirts"):
        self.tshirts_dir = os.path.abspath(tshirts_dir)
        self.garments = []
        self.current_index = 0
        self.load_garments()

    def load_garments(self):
        """Scan directory for PNG/JPEG garments and store absolute paths."""
        self.garments = []
        if os.path.isdir(self.tshirts_dir):
            valid_exts = (".png", ".jpg", ".jpeg")
            found = [
                os.path.abspath(os.path.join(self.tshirts_dir, filename))
                for filename in os.listdir(self.tshirts_dir)
                if filename.lower().endswith(valid_exts)
            ]
            # Stable N/P order (tshirt1, tshirt2, ...)
            self.garments = sorted(found, key=lambda path: os.path.basename(path).lower())

        if not self.garments:
            print(
                f"[WARN] No garment images found in {self.tshirts_dir}. Placeholder mode."
            )
        else:
            names = ", ".join(os.path.basename(path) for path in self.garments)
            print(f"[INFO] Loaded {len(self.garments)} garment(s): {names}")

    def get_current_garment_path(self) -> str:
        """Get path of currently selected garment."""
        if not self.garments:
            return ""
        return self.garments[self.current_index]

    def get_current_name(self) -> str:
        """Get display name of current garment."""
        if not self.garments:
            return "None"
        filename = os.path.basename(self.garments[self.current_index])
        return os.path.splitext(filename)[0].replace("_", " ").title()

    def next_garment(self) -> str:
        """Switch to next garment and return its path."""
        if self.garments:
            self.current_index = (self.current_index + 1) % len(self.garments)
        return self.get_current_garment_path()

    def prev_garment(self) -> str:
        """Switch to previous garment and return its path."""
        if self.garments:
            self.current_index = (self.current_index - 1) % len(self.garments)
        return self.get_current_garment_path()
