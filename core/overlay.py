import cv2
import numpy as np
import os

from core.garment_overlay import (
    COLLAR_LIFT,
    blend_garment_roi,
    compute_garment_size,
    feather_alpha_channel,
)

# Smooth pose jitter (hands entering the frame, brief dropouts).
EMA_ALPHA = 0.3


class GarmentOverlay:

    def __init__(self, garment_path: str = None):
        self.garment = None
        self.garment_path = None
        self._ema = {}
        self._cache = {}
        if garment_path:
            self.load_garment(garment_path)

    def _smooth(self, key: str, value: float) -> float:
        prev = self._ema.get(key)
        if prev is None:
            self._ema[key] = value
            return value
        blended = EMA_ALPHA * value + (1.0 - EMA_ALPHA) * prev
        self._ema[key] = blended
        return blended

    def load_garment(self, garment_path: str) -> bool:
        if not garment_path:
            self.garment = None
            self.garment_path = None
            return False

        if garment_path == self.garment_path and self.garment is not None:
            return True

        cached = self._cache.get(garment_path)
        if cached is not None:
            self.garment = cached
            self.garment_path = garment_path
            for key in ("width", "height"):
                self._ema.pop(key, None)
            return True

        image = self._read_bgra(garment_path)
        if image is None:
            print(f"[ERROR] Could not load garment image: {garment_path}")
            return False

        hard, soft = self._extract_shirt_mask(image)
        image[:, :, 3] = soft
        image[hard == 0] = 0
        self.garment = self._crop_to_mask(image, hard)
        self.garment_path = garment_path
        self._cache[garment_path] = self.garment
        for key in ("width", "height"):
            self._ema.pop(key, None)
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
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
        if image.shape[2] == 3:
            b, g, r = cv2.split(image)
            alpha = np.full(b.shape, 255, dtype=np.uint8)
            return cv2.merge((b, g, r, alpha))
        return image

    @staticmethod
    def _crop_to_mask(image: np.ndarray, hard_mask: np.ndarray) -> np.ndarray:
        ys, xs = np.where(hard_mask > 0)
        if xs.size == 0:
            return image
        h, w = hard_mask.shape[:2]
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, x0 = max(0, y0), max(0, x0)
        y1, x1 = min(h, y1), min(w, x1)
        return image[y0:y1, x0:x1].copy()

    @staticmethod
    def _flood_from_border(binary: np.ndarray) -> np.ndarray:
        """Fill every background component that touches the image perimeter."""
        h, w = binary.shape[:2]
        work = binary.copy()
        ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        flags = 8 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)

        seeds = []
        for x in range(0, w, 2):
            seeds.append((x, 0))
            seeds.append((x, h - 1))
        for y in range(0, h, 2):
            seeds.append((0, y))
            seeds.append((w - 1, y))

        for x, y in seeds:
            if work[y, x] == 0 or ff_mask[y + 1, x + 1] != 0:
                continue
            cv2.floodFill(work, ff_mask, (int(x), int(y)), 0, 0, 0, flags)
        return ff_mask[1:-1, 1:-1]

    @staticmethod
    def _largest_central_contour(binary: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        out = np.zeros(binary.shape, dtype=np.uint8)
        if not contours:
            return binary
        h, w = binary.shape[:2]
        cx, cy = w * 0.5, h * 0.5

        def score(contour):
            area = cv2.contourArea(contour)
            m = cv2.moments(contour)
            if m["m00"] < 1:
                return area
            mx, my = m["m10"] / m["m00"], m["m01"] / m["m00"]
            dist = ((mx - cx) ** 2 + (my - cy) ** 2) ** 0.5
            return area - dist * 12.0

        cv2.drawContours(out, [max(contours, key=score)], -1, 255, cv2.FILLED)
        return out

    @staticmethod
    def _try_rembg(bgr: np.ndarray):
        try:
            from rembg import remove
        except ImportError:
            return None
        try:
            rgba = remove(bgr)
            if rgba is None or rgba.ndim < 3 or rgba.shape[2] < 4:
                return None
            alpha = rgba[:, :, 3]
            if int(alpha.max()) < 32:
                return None
            return np.where(alpha > 16, 255, 0).astype(np.uint8)
        except Exception:
            return None

    @classmethod
    def _mask_checker_wipe(cls, bgr: np.ndarray) -> np.ndarray:
        """Treat gray checker cells as background; keep the central garment.

        White fabric is not keyed. Only the darker checker color (sampled at
        the top corners) is dilated so white grid squares join, then the
        perimeter is flood-filled away.
        """
        h, w = bgr.shape[:2]
        size = max(16, min(48, h // 12, w // 12))
        pixels = np.vstack(
            [
                bgr[:size, :size].reshape(-1, 3),
                bgr[:size, -size:].reshape(-1, 3),
            ]
        ).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        _, _, centers = cv2.kmeans(
            pixels, 2, None, criteria, 4, cv2.KMEANS_PP_CENTERS
        )
        luma = centers[:, 0] * 0.114 + centers[:, 1] * 0.587 + centers[:, 2] * 0.299
        dark = centers[int(np.argmin(luma))]

        sample = cv2.cvtColor(bgr[:size, :size], cv2.COLOR_BGR2GRAY)[size // 2]
        jumps = np.where(np.abs(np.diff(sample.astype(np.int16))) > 12)[0]
        period = 10
        if len(jumps) >= 2:
            period = int(np.clip(np.median(np.diff(jumps)), 4, 28))

        lo = np.clip(dark - 30, 0, 255).astype(np.uint8)
        hi = np.clip(dark + 30, 0, 255).astype(np.uint8)
        dark_mask = cv2.inRange(bgr, lo, hi)

        k = period * 2 + 3
        if k % 2 == 0:
            k += 1
        dark_mask = cv2.dilate(dark_mask, np.ones((k, k), np.uint8), iterations=1)

        background = cls._flood_from_border(dark_mask)
        foreground = np.where(background == 0, 255, 0).astype(np.uint8)

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        foreground[hsv[:, :, 1] > 38] = 255
        foreground[background > 0] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
        foreground = cv2.morphologyEx(
            foreground, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )
        hard = cls._largest_central_contour(foreground)
        hard[background > 0] = 0
        return hard

    @staticmethod
    def _clear_edge_fringe(bgr: np.ndarray, hard: np.ndarray) -> np.ndarray:
        """Drop gray/white checker leftovers on the silhouette rim only."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        gray_white = (hsv[:, :, 1] < 28) & (hsv[:, :, 2] > 165)
        ring = cv2.dilate(hard, np.ones((7, 7), np.uint8)) - cv2.erode(
            hard, np.ones((7, 7), np.uint8)
        )
        cleaned = hard.copy()
        cleaned[(ring > 0) & gray_white] = 0
        return cleaned

    @classmethod
    def _extract_shirt_mask(cls, image: np.ndarray):
        bgr = image[:, :, :3]
        h, w = bgr.shape[:2]

        checker_mask = cls._mask_checker_wipe(bgr)
        rembg_mask = cls._try_rembg(bgr)

        if rembg_mask is not None:
            hard = rembg_mask.copy()
            hard[checker_mask == 0] = 0
        else:
            hard = checker_mask

        coverage = float(np.count_nonzero(hard)) / float(h * w)
        if coverage < 0.08 or coverage > 0.92:
            hard = checker_mask

        hard = cls._largest_central_contour(hard)
        hard = cls._clear_edge_fringe(bgr, hard)
        soft = feather_alpha_channel(hard)
        return hard, soft

    def apply_overlay(self, frame, body_data):
        if frame is None or self.garment is None:
            return frame
        if not body_data or not isinstance(body_data, dict):
            return frame

        left_shoulder = body_data.get("left_shoulder")
        right_shoulder = body_data.get("right_shoulder")
        if left_shoulder is None or right_shoulder is None:
            return frame

        frame_h, frame_w = frame.shape[:2]
        lx, ly = float(left_shoulder[0]), float(left_shoulder[1])
        rx, ry = float(right_shoulder[0]), float(right_shoulder[1])

        # Landmarks may be normalized (0-1) or already in pixels.
        if max(abs(lx), abs(ly), abs(rx), abs(ry)) <= 1.5:
            lx, rx = lx * frame_w, rx * frame_w
            ly, ry = ly * frame_h, ry * frame_h

        shoulder_pixel_dist = abs(rx - lx)
        if shoulder_pixel_dist <= 1:
            return frame

        lx = self._smooth("lx", lx)
        ly = self._smooth("ly", ly)
        rx = self._smooth("rx", rx)
        ry = self._smooth("ry", ry)
        shoulder_pixel_dist = abs(rx - lx)

        torso_length = self._torso_length(body_data, frame_w, frame_h, (lx + rx) * 0.5, (ly + ry) * 0.5)
        target_width, target_height = compute_garment_size(
            shoulder_pixel_dist,
            torso_length,
            self.garment.shape[0],
            self.garment.shape[1],
            frame_w,
            frame_h,
        )
        target_width = int(self._smooth("width", float(target_width)))
        target_height = int(self._smooth("height", float(target_height)))
        if target_width <= 0 or target_height <= 0:
            return frame

        resized = cv2.resize(
            self.garment,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
        if resized.shape[2] >= 4:
            resized[:, :, 3] = feather_alpha_channel(resized[:, :, 3])
        garment_h, garment_w = resized.shape[:2]

        shoulder_center_x = int((lx + rx) / 2)
        shoulder_center_y = int((ly + ry) / 2)

        x_offset = int(shoulder_center_x - garment_w // 2)
        y_offset = int(shoulder_center_y - int(garment_h * COLLAR_LIFT))

        x1 = int(np.clip(x_offset, 0, frame_w))
        y1 = int(np.clip(y_offset, 0, frame_h))
        x2 = int(np.clip(x_offset + garment_w, 0, frame_w))
        y2 = int(np.clip(y_offset + garment_h, 0, frame_h))

        gx1 = int(np.clip(x1 - x_offset, 0, garment_w))
        gy1 = int(np.clip(y1 - y_offset, 0, garment_h))
        gx2 = int(np.clip(gx1 + (x2 - x1), 0, garment_w))
        gy2 = int(np.clip(gy1 + (y2 - y1), 0, garment_h))

        if x2 <= x1 or y2 <= y1 or gx2 <= gx1 or gy2 <= gy1:
            return frame

        crop = resized[gy1:gy2, gx1:gx2]
        frame_roi = frame[y1:y2, x1:x2]
        roi_h = min(crop.shape[0], frame_roi.shape[0], y2 - y1)
        roi_w = min(crop.shape[1], frame_roi.shape[1], x2 - x1)
        if roi_h <= 0 or roi_w <= 0:
            return frame

        crop = crop[:roi_h, :roi_w]
        frame_roi = frame[y1:y1 + roi_h, x1:x1 + roi_w]
        if crop.shape[2] < 4:
            return frame

        frame[y1:y1 + roi_h, x1:x1 + roi_w] = blend_garment_roi(frame_roi, crop)
        return frame

    def _torso_length(self, body_data, frame_w, frame_h, _shoulder_cx, shoulder_cy):
        """Vertical shoulder-center to hip-center distance, EMA-smoothed."""
        hinted = body_data.get("torso_length")
        left_hip = body_data.get("left_hip")
        right_hip = body_data.get("right_hip")

        torso = None
        if left_hip is not None and right_hip is not None:
            lhx, lhy = float(left_hip[0]), float(left_hip[1])
            rhx, rhy = float(right_hip[0]), float(right_hip[1])
            if max(abs(lhx), abs(lhy), abs(rhx), abs(rhy)) <= 1.5:
                lhx, rhx = lhx * frame_w, rhx * frame_w
                lhy, rhy = lhy * frame_h, rhy * frame_h
            lhy = self._smooth("lhy", lhy)
            rhy = self._smooth("rhy", rhy)
            hip_cy = (lhy + rhy) * 0.5
            torso = abs(hip_cy - shoulder_cy)
        elif hinted is not None:
            torso = float(hinted)

        if torso is None or torso <= 1:
            return None
        return self._smooth("torso", torso)
