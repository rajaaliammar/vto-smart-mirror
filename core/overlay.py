import cv2
import numpy as np
import os

from core.garment_overlay import (
    AMBIENT_REF_V,
    COLOR_VARIANTS,
    COLLAR_LIFT,
    HIP_SCALE,
    WAIST_RISE,
    adapt_garment_lighting,
    apply_palette_color,
    blend_garment_roi,
    compute_garment_size,
    compute_pants_size,
    estimate_torso_ambient,
    feather_alpha_channel,
)

# Smooth pose jitter (hands entering the frame, brief dropouts).
EMA_ALPHA = 0.3
AMBIENT_EMA = 0.18


class GarmentOverlay:

    def __init__(self, garment_path: str = None):
        self.garment = None
        self.garment_path = None
        self.lower_garment = None
        self.lower_path = None
        self.outfit_mode = "upper"
        self._ema = {}
        self._cache = {}
        self._tint_cache = {}
        self.color_index = 0
        self.fit_scale = 1.55
        self._ambient_v = AMBIENT_REF_V
        if garment_path:
            self.load_garment(garment_path)

    def cycle_color(self):
        self.color_index = (self.color_index + 1) % len(COLOR_VARIANTS)
        return self.current_color()

    def set_color_key(self, key: str):
        wanted = str(key or "").strip().lower()
        for index, variant in enumerate(COLOR_VARIANTS):
            if variant["key"] == wanted:
                self.color_index = index
                return variant
        return self.current_color()

    def current_color(self):
        return COLOR_VARIANTS[self.color_index]

    def toggle_outfit_mode(self) -> str:
        self.outfit_mode = "full" if self.outfit_mode == "upper" else "upper"
        if self.outfit_mode != "full":
            self.clear_lower_garment()
        return self.outfit_mode

    def clear_lower_garment(self) -> None:
        """Drop pants so UPPER ONLY never blits leftover lower assets."""
        self.lower_garment = None
        self.lower_path = None
        for key in ("p_width", "p_height", "hlx", "hly", "hrx", "hry"):
            self._ema.pop(key, None)

    def _tinted_source(self):
        if self.garment is None:
            return None
        variant = self.current_color()
        key = (self.garment_path, variant["key"])
        cached = self._tint_cache.get(key)
        if cached is not None:
            return cached
        tinted = apply_palette_color(self.garment, variant)
        self._tint_cache[key] = tinted
        return tinted

    def _update_ambient(self, frame, body_data) -> float:
        sample = estimate_torso_ambient(frame, body_data)
        if sample is None:
            return self._ambient_v
        self._ambient_v = AMBIENT_EMA * sample + (1.0 - AMBIENT_EMA) * self._ambient_v
        return self._ambient_v

    def _smooth(self, key: str, value: float) -> float:
        prev = self._ema.get(key)
        if prev is None:
            self._ema[key] = value
            return value
        blended = EMA_ALPHA * value + (1.0 - EMA_ALPHA) * prev
        self._ema[key] = blended
        return blended

    def load_garment(self, garment_path: str) -> bool:
        prepared = self._prepare_layer(garment_path)
        if prepared is None:
            if not garment_path:
                self.garment = None
                self.garment_path = None
            return False
        self.garment = prepared
        self.garment_path = garment_path
        for key in ("width", "height"):
            self._ema.pop(key, None)
        return True

    def load_lower_garment(self, garment_path: str) -> bool:
        prepared = self._prepare_layer(garment_path)
        if prepared is None:
            if not garment_path:
                self.lower_garment = None
                self.lower_path = None
            return False
        self.lower_garment = prepared
        self.lower_path = garment_path
        for key in ("p_width", "p_height"):
            self._ema.pop(key, None)
        return True

    def _prepare_layer(self, garment_path: str):
        if not garment_path:
            return None
        cached = self._cache.get(garment_path)
        if cached is not None:
            return cached

        image = self._read_bgra(garment_path)
        if image is None:
            print(f"[ERROR] Could not load garment image: {garment_path}")
            return None

        if self._has_cutout_alpha(image):
            hard = np.where(image[:, :, 3] > 16, 255, 0).astype(np.uint8)
            soft = feather_alpha_channel(hard)
            image[:, :, 3] = np.minimum(image[:, :, 3], soft)
            image[hard == 0] = 0
        else:
            hard, soft = self._extract_shirt_mask(image)
            image[:, :, 3] = soft
            image[hard == 0] = 0

        cropped = self._crop_to_mask(image, np.where(image[:, :, 3] > 8, 255, 0).astype(np.uint8))
        self._cache[garment_path] = cropped
        return cropped

    @staticmethod
    def _has_cutout_alpha(image: np.ndarray) -> bool:
        if image is None or image.ndim < 3 or image.shape[2] < 4:
            return False
        frac = float(np.mean(image[:, :, 3] > 20))
        return 0.06 < frac < 0.92

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
        if frame is None or not body_data or not isinstance(body_data, dict):
            return frame
        has_upper = self.garment is not None
        has_lower = (
            self.outfit_mode == "full"
            and self.lower_garment is not None
            and self.lower_path
        )
        if not has_upper and not has_lower:
            return frame

        ambient_v = self._update_ambient(frame, body_data)
        if has_lower:
            frame = self._apply_lower(frame, body_data, ambient_v)
        if has_upper:
            frame = self._apply_upper(frame, body_data, ambient_v)
        return frame

    def _apply_upper(self, frame, body_data, ambient_v):
        left_shoulder = body_data.get("left_shoulder")
        right_shoulder = body_data.get("right_shoulder")
        if left_shoulder is None or right_shoulder is None:
            return frame

        frame_h, frame_w = frame.shape[:2]
        left = self._as_xy(left_shoulder, frame_w, frame_h)
        right = self._as_xy(right_shoulder, frame_w, frame_h)
        if left is None or right is None:
            return frame
        lx, ly = left
        rx, ry = right

        lx = self._smooth("lx", lx)
        ly = self._smooth("ly", ly)
        rx = self._smooth("rx", rx)
        ry = self._smooth("ry", ry)
        shoulder_pixel_dist = abs(rx - lx)
        if shoulder_pixel_dist <= 1:
            return frame

        source = self._tinted_source()
        if source is None:
            return frame

        torso_length = self._torso_length(body_data, frame_w, frame_h, (lx + rx) * 0.5, (ly + ry) * 0.5)
        target_width, target_height = compute_garment_size(
            shoulder_pixel_dist,
            torso_length,
            source.shape[0],
            source.shape[1],
            frame_w,
            frame_h,
            scale=self.fit_scale,
        )
        target_width = int(self._smooth("width", float(target_width)))
        target_height = int(self._smooth("height", float(target_height)))
        if target_width <= 0 or target_height <= 0:
            return frame

        resized = cv2.resize(source, (target_width, target_height), interpolation=cv2.INTER_AREA)
        resized = adapt_garment_lighting(resized, ambient_v)
        if resized.shape[2] >= 4:
            resized[:, :, 3] = feather_alpha_channel(resized[:, :, 3])

        garment_h, garment_w = resized.shape[:2]
        x_offset = int((lx + rx) / 2 - garment_w // 2)
        y_offset = int((ly + ry) / 2 - int(garment_h * COLLAR_LIFT))
        return self._blit_layer(frame, resized, x_offset, y_offset)

    def _apply_lower(self, frame, body_data, ambient_v):
        if self.outfit_mode != "full":
            return frame
        source = self.lower_garment
        if source is None:
            return frame
        frame_h, frame_w = frame.shape[:2]
        left_hip = self._as_xy(body_data.get("left_hip"), frame_w, frame_h)
        right_hip = self._as_xy(body_data.get("right_hip"), frame_w, frame_h)
        if left_hip is None or right_hip is None:
            return frame

        lx = self._smooth("hlx", left_hip[0])
        ly = self._smooth("hly", left_hip[1])
        rx = self._smooth("hrx", right_hip[0])
        ry = self._smooth("hry", right_hip[1])
        hip_width = abs(rx - lx)
        if hip_width <= 1:
            hip_width = float(body_data.get("hip_width") or 0)
        if hip_width <= 1:
            return frame

        hip_cx = (lx + rx) * 0.5
        hip_cy = (ly + ry) * 0.5
        leg_length = body_data.get("leg_length")
        left_ankle = self._as_xy(body_data.get("left_ankle"), frame_w, frame_h)
        right_ankle = self._as_xy(body_data.get("right_ankle"), frame_w, frame_h)
        if left_ankle is not None and right_ankle is not None:
            ankle_cy = (left_ankle[1] + right_ankle[1]) * 0.5
            leg_length = abs(ankle_cy - hip_cy)
        if not leg_length:
            torso = body_data.get("torso_length")
            leg_length = float(torso) * 1.85 if torso else hip_width * 2.4

        pants_scale = HIP_SCALE * (float(self.fit_scale) / 1.55)
        target_width, target_height = compute_pants_size(
            hip_width,
            float(leg_length),
            source.shape[0],
            source.shape[1],
            frame_w,
            frame_h,
            scale=pants_scale,
        )
        target_width = int(self._smooth("p_width", float(target_width)))
        target_height = int(self._smooth("p_height", float(target_height)))
        if target_width <= 0 or target_height <= 0:
            return frame

        resized = cv2.resize(source, (target_width, target_height), interpolation=cv2.INTER_AREA)
        resized = adapt_garment_lighting(resized, ambient_v)
        if resized.shape[2] >= 4:
            resized[:, :, 3] = feather_alpha_channel(resized[:, :, 3])

        garment_h, garment_w = resized.shape[:2]
        x_offset = int(hip_cx - garment_w // 2)
        y_offset = int(hip_cy - int(garment_h * WAIST_RISE))
        return self._blit_layer(frame, resized, x_offset, y_offset)

    def _blit_layer(self, frame, layer, x_offset, y_offset):
        frame_h, frame_w = frame.shape[:2]
        garment_h, garment_w = layer.shape[:2]
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
        crop = layer[gy1:gy2, gx1:gx2]
        frame_roi = frame[y1:y2, x1:x2]
        roi_h = min(crop.shape[0], frame_roi.shape[0], y2 - y1)
        roi_w = min(crop.shape[1], frame_roi.shape[1], x2 - x1)
        if roi_h <= 0 or roi_w <= 0 or crop.shape[2] < 4:
            return frame
        crop = crop[:roi_h, :roi_w]
        frame[y1:y1 + roi_h, x1:x1 + roi_w] = blend_garment_roi(
            frame[y1:y1 + roi_h, x1:x1 + roi_w], crop
        )
        return frame

    @staticmethod
    def _as_xy(point, frame_w, frame_h):
        if point is None:
            return None
        x, y = float(point[0]), float(point[1])
        if max(abs(x), abs(y)) <= 1.5:
            x, y = x * frame_w, y * frame_h
        return x, y

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
