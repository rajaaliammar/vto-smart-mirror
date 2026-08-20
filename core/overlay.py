import cv2
import numpy as np
import os


class GarmentOverlay:

    def __init__(self, garment_path: str = None):
        self.garment = None
        self.garment_path = None
        if garment_path:
            self.load_garment(garment_path)

    def load_garment(self, garment_path: str) -> bool:
        """Reload the active garment and rebuild a shirt-only alpha mask."""
        if not garment_path:
            self.garment = None
            self.garment_path = None
            return False

        image = self._read_bgra(garment_path)
        if image is None:
            print(f"[ERROR] Could not load garment image: {garment_path}")
            return False

        mask = self._extract_shirt_mask(image)
        image[:, :, 3] = mask
        self.garment = image
        self.garment_path = garment_path
        return True

    @staticmethod
    def _read_bgra(garment_path: str):
        image = cv2.imread(garment_path, cv2.IMREAD_UNCHANGED)
        if image is None and os.path.isfile(garment_path):
            data = np.fromfile(garment_path, dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

        if image is None:
            return None

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
        elif image.shape[2] == 3:
            b, g, r = cv2.split(image)
            alpha = np.full(b.shape, 255, dtype=np.uint8)
            image = cv2.merge((b, g, r, alpha))

        return image

    @staticmethod
    def _largest_contour_mask(binary: np.ndarray) -> np.ndarray:
        """Keep only the largest central garment silhouette."""
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        mask = np.zeros(binary.shape, dtype=np.uint8)
        if not contours:
            return binary
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
        return mask

    @staticmethod
    def _extract_shirt_mask(image: np.ndarray) -> np.ndarray:
        """Build a shirt-only alpha mask without color-thresholding fabric.

        Uses existing transparency when the file is already cut out. Otherwise
        flood-fills from the corners (solid white or checkerboard) and keeps
        the largest garment contour.
        """
        h, w = image.shape[:2]
        alpha = image[:, :, 3]
        bgr = image[:, :, :3]

        corner_alpha = (
            int(alpha[0, 0]),
            int(alpha[0, w - 1]),
            int(alpha[h - 1, 0]),
            int(alpha[h - 1, w - 1]),
        )
        already_cut_out = max(corner_alpha) < 24 and int(alpha.max()) > 200

        if already_cut_out:
            binary = np.where(alpha > 16, 255, 0).astype(np.uint8)
            mask = GarmentOverlay._largest_contour_mask(binary)
            return cv2.GaussianBlur(mask, (3, 3), 0)

        # Sample top-left / top-right (and remaining corners) so both checker
        # colors get seeded, then flood-fill with 8-connectivity.
        ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        flags = 8 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
        lo_diff = (20, 20, 20)
        up_diff = (20, 20, 20)
        work = bgr.copy()

        seeds = []
        for cx, cy in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
            step_x = 1 if cx == 0 else -1
            step_y = 1 if cy == 0 else -1
            for i in range(10):
                for j in range(10):
                    seeds.append((cx + i * step_x, cy + j * step_y))

        for x, y in seeds:
            if not (0 <= x < w and 0 <= y < h):
                continue
            if ff_mask[y + 1, x + 1] != 0:
                continue
            cv2.floodFill(
                work,
                ff_mask,
                (int(x), int(y)),
                0,
                lo_diff,
                up_diff,
                flags,
            )

        background = ff_mask[1:-1, 1:-1]
        foreground = np.where(background == 0, 255, 0).astype(np.uint8)

        fg_ratio = float(np.count_nonzero(foreground)) / float(h * w)
        if fg_ratio < 0.04:
            # Flood fill ate the garment (e.g. white shirt ~ background). Keep it.
            return np.full((h, w), 255, dtype=np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
        mask = GarmentOverlay._largest_contour_mask(foreground)
        return cv2.GaussianBlur(mask, (3, 3), 0)

    def apply_overlay(self, frame, body_data):
        if frame is None or self.garment is None:
            return frame
        if not body_data or not isinstance(body_data, dict):
            return frame

        left_shoulder = body_data.get("left_shoulder")
        right_shoulder = body_data.get("right_shoulder")
        center = body_data.get("chest_center")
        shoulder_width = body_data.get("shoulder_width")
        if shoulder_width is None:
            return frame

        if left_shoulder is not None and right_shoulder is not None:
            shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) * 0.5
            shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) * 0.5
        elif center is not None:
            shoulder_center_x, shoulder_center_y = center
        else:
            return frame

        target_width = int(shoulder_width * 2.2)
        if target_width <= 0:
            return frame

        aspect_ratio = self.garment.shape[0] / self.garment.shape[1]
        target_height = int(target_width * aspect_ratio)
        if target_height <= 0:
            return frame

        resized = cv2.resize(
            self.garment,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
        h, w = resized.shape[:2]

        x_offset = int(shoulder_center_x - w // 2)
        y_offset = int(shoulder_center_y - (h * 0.15))

        frame_h, frame_w = frame.shape[:2]

        x1, x2 = max(0, x_offset), min(frame_w, x_offset + w)
        y1, y2 = max(0, y_offset), min(frame_h, y_offset + h)

        gx1, gx2 = max(0, -x_offset), min(w, frame_w - x_offset)
        gy1, gy2 = max(0, -y_offset), min(h, frame_h - y_offset)

        if x1 >= x2 or y1 >= y2 or gx1 >= gx2 or gy1 >= gy2:
            return frame

        crop = resized[gy1:gy2, gx1:gx2]
        frame_roi = frame[y1:y2, x1:x2]
        roi_h = min(crop.shape[0], frame_roi.shape[0])
        roi_w = min(crop.shape[1], frame_roi.shape[1])
        if roi_h <= 0 or roi_w <= 0:
            return frame

        crop = crop[:roi_h, :roi_w]
        frame_roi = frame[y1:y1 + roi_h, x1:x1 + roi_w]

        alpha = (crop[:, :, 3].astype(np.float32) / 255.0)[:, :, np.newaxis]
        blended = alpha * crop[:, :, :3].astype(np.float32) + (1.0 - alpha) * frame_roi.astype(
            np.float32
        )
        frame[y1:y1 + roi_h, x1:x1 + roi_w] = blended.astype(np.uint8)

        return frame
