"""Phase 5 body fitting and PNG edge-feathering for the garment overlay."""

from typing import Optional, Tuple
import os

import cv2
import numpy as np

# Keep the calibrated sleeve-to-sleeve scale (do not restore the 2.1 / 0.65 cap).
SHOULDER_SCALE = 1.55
# Shirt length vs shoulder-center → hip-center vertical span.
TORSO_LENGTH_SCALE = 1.10
COLLAR_LIFT = 0.08
MAX_WIDTH_FRAME_FRAC = 0.50
MAX_HEIGHT_FRAME_FRAC = 0.72
# Independent height may not wander too far from the PNG aspect ratio.
ASPECT_HEIGHT_MIN = 0.80
ASPECT_HEIGHT_MAX = 1.30
FEATHER_RADIUS = 6


def compute_garment_size(
    shoulder_width: float,
    torso_length: Optional[float],
    garment_h: int,
    garment_w: int,
    frame_w: int,
    frame_h: int,
) -> Tuple[int, int]:
    """Scale width from shoulders and length from shoulder-to-hip distance."""
    target_width = int(shoulder_width * SHOULDER_SCALE)
    target_width = int(np.clip(target_width, 40, int(frame_w * MAX_WIDTH_FRAME_FRAC)))

    aspect = garment_h / float(max(garment_w, 1))
    aspect_height = target_width * aspect

    if torso_length is not None and torso_length > 8:
        body_height = float(torso_length) * TORSO_LENGTH_SCALE
        lo = aspect_height * ASPECT_HEIGHT_MIN
        hi = aspect_height * ASPECT_HEIGHT_MAX
        target_height = float(np.clip(body_height, lo, hi))
    else:
        target_height = aspect_height

    target_height = float(np.clip(target_height, 40, int(frame_h * MAX_HEIGHT_FRAME_FRAC)))
    return max(1, int(target_width)), max(1, int(target_height))


def feather_alpha_channel(alpha: np.ndarray, radius: int = FEATHER_RADIUS) -> np.ndarray:
    """Fade PNG silhouette edges so compositing has no hard cutout."""
    if alpha is None or radius < 1:
        return alpha

    hard = np.where(alpha > 16, 255, 0).astype(np.uint8)
    dist = cv2.distanceTransform(hard, cv2.DIST_L2, 5)
    ramp = np.clip(dist / float(radius), 0.0, 1.0)
    softened = np.minimum(alpha.astype(np.float32), ramp * 255.0)

    k = min(radius * 2 + 1, 9)
    if k % 2 == 0:
        k += 1
    softened = cv2.GaussianBlur(softened, (k, k), 0)
    return np.clip(softened, 0, 255).astype(np.uint8)


def blend_garment_roi(frame_roi: np.ndarray, crop: np.ndarray) -> np.ndarray:
    """Premultiplied-style alpha blend of a BGRA crop onto a BGR ROI."""
    alpha = crop[:, :, 3].astype(np.float32) / 255.0
    alpha = alpha[:, :, np.newaxis]
    blended = alpha * crop[:, :, :3].astype(np.float32) + (1.0 - alpha) * frame_roi.astype(
        np.float32
    )
    return blended.astype(np.uint8)


HIP_SCALE = 1.48
LEG_LENGTH_SCALE = 1.06
WAIST_RISE = 0.12
MAX_PANTS_WIDTH_FRAC = 0.52
MAX_PANTS_HEIGHT_FRAC = 0.82
PANTS_ASPECT_MIN = 0.78
PANTS_ASPECT_MAX = 1.28


def compute_pants_size(
    hip_width: float,
    leg_length: Optional[float],
    garment_h: int,
    garment_w: int,
    frame_w: int,
    frame_h: int,
) -> Tuple[int, int]:
    """Scale jeans width from hips and length from hip-to-ankle distance."""
    target_width = int(max(hip_width, 1.0) * HIP_SCALE)
    target_width = int(np.clip(target_width, 40, int(frame_w * MAX_PANTS_WIDTH_FRAC)))

    aspect = garment_h / float(max(garment_w, 1))
    aspect_height = target_width * aspect

    if leg_length is not None and leg_length > 12:
        body_height = float(leg_length) * LEG_LENGTH_SCALE
        lo = aspect_height * PANTS_ASPECT_MIN
        hi = aspect_height * PANTS_ASPECT_MAX
        target_height = float(np.clip(body_height, lo, hi))
    else:
        target_height = aspect_height

    target_height = float(np.clip(target_height, 50, int(frame_h * MAX_PANTS_HEIGHT_FRAC)))
    return max(1, int(target_width)), max(1, int(target_height))


def render_default_jeans(width: int = 500, height: int = 860) -> np.ndarray:
    """Procedural jeans PNG (BGRA) used when no lower-body asset is on disk."""
    img = np.zeros((height, width, 4), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    cx = width // 2
    waist_y = int(height * 0.03)
    hip_y = int(height * 0.18)
    crotch_y = int(height * 0.34)
    knee_y = int(height * 0.64)
    ankle_y = int(height * 0.97)
    waist_half = int(width * 0.23)
    hip_half = int(width * 0.265)
    knee_outer = int(width * 0.20)
    ankle_half = int(width * 0.15)
    inseam = int(width * 0.032)

    left = np.array(
        [
            [cx - waist_half, waist_y],
            [cx - 3, waist_y],
            [cx - 3, crotch_y - 12],
            [cx - inseam, crotch_y],
            [cx - inseam, ankle_y],
            [cx - ankle_half - inseam, ankle_y],
            [cx - knee_outer, knee_y],
            [cx - hip_half, hip_y],
        ],
        dtype=np.int32,
    )
    right = np.array(
        [
            [cx + 3, waist_y],
            [cx + waist_half, waist_y],
            [cx + hip_half, hip_y],
            [cx + knee_outer, knee_y],
            [cx + ankle_half + inseam, ankle_y],
            [cx + inseam, ankle_y],
            [cx + inseam, crotch_y],
            [cx + 3, crotch_y - 12],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [left, right], 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    denim = np.zeros((height, width, 3), dtype=np.uint8)
    denim[:, :] = (86, 54, 26)
    shade = np.linspace(1.08, 0.82, height, dtype=np.float32)[:, None]
    denim = np.clip(denim.astype(np.float32) * shade[:, :, None], 0, 255).astype(np.uint8)
    stitch = (40, 160, 210)
    cv2.line(denim, (cx - waist_half + 8, waist_y + 10), (cx + waist_half - 8, waist_y + 10), stitch, 2)
    cv2.line(denim, (cx - inseam - 6, crotch_y), (cx - inseam - 6, ankle_y - 8), stitch, 1)
    cv2.line(denim, (cx + inseam + 6, crotch_y), (cx + inseam + 6, ankle_y - 8), stitch, 1)

    img[:, :, :3] = denim
    img[:, :, 3] = mask
    img[mask == 0] = 0
    return img


def ensure_default_pants(project_root: Optional[str] = None) -> str:
    """Write assets/sample_clothes/pants/jeans.png if it does not already exist."""
    root = project_root or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dest = os.path.join(root, "assets", "sample_clothes", "pants", "jeans.png")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.isfile(dest) and os.path.getsize(dest) > 2000:
        return dest
    image = render_default_jeans()
    cv2.imwrite(dest, image)
    return dest


# OpenCV hue units (0-179). Swatches are BGR for HUD drawing.
COLOR_VARIANTS = [
    {
        "key": "original",
        "label": "Original",
        "mode": "none",
        "hue_shift": 0,
        "target_hue": None,
        "swatch_bgr": (198, 198, 198),
    },
    {
        "key": "crimson",
        "label": "Crimson Red",
        "mode": "rehue",
        "hue_shift": 0,
        "target_hue": 2,
        "swatch_bgr": (40, 38, 186),
    },
    {
        "key": "royal",
        "label": "Royal Blue",
        "mode": "rehue",
        "hue_shift": 0,
        "target_hue": 118,
        "swatch_bgr": (176, 78, 28),
    },
    {
        "key": "emerald",
        "label": "Emerald Green",
        "mode": "rehue",
        "hue_shift": 0,
        "target_hue": 58,
        "swatch_bgr": (62, 158, 36),
    },
    {
        "key": "charcoal",
        "label": "Charcoal",
        "mode": "charcoal",
        "hue_shift": 0,
        "target_hue": None,
        "swatch_bgr": (46, 46, 46),
    },
]

AMBIENT_REF_V = 128.0
AMBIENT_GAIN_MIN = 0.70
AMBIENT_GAIN_MAX = 1.32
AMBIENT_CONTRAST_MIN = 0.88
AMBIENT_CONTRAST_MAX = 1.18


def apply_color_tint(garment_img, hue_shift):
    """Shift BGRA/BGR garment hue. `hue_shift` is in OpenCV hue units (0-179)."""
    if garment_img is None:
        return garment_img
    shift = int(round(float(hue_shift))) % 180
    if shift == 0:
        return garment_img.copy()

    bgr, alpha = _split_color_alpha(garment_img)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + shift) % 180
    tinted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return _merge_color_alpha(tinted, alpha)


def apply_palette_color(garment_img, variant) -> np.ndarray:
    """Recolor a garment to a named palette entry while keeping fabric texture."""
    if garment_img is None:
        return garment_img
    mode = (variant or {}).get("mode", "none")
    if mode == "none":
        return garment_img.copy()
    if mode == "charcoal":
        return _apply_charcoal(garment_img)

    target = int(variant.get("target_hue", 0)) % 180
    median_h = _median_hue(garment_img)
    shift = (target - int(median_h)) % 180
    tinted = apply_color_tint(garment_img, shift)
    return _lock_hue_and_chroma(tinted, target, min_s=92)


def estimate_torso_ambient(frame, body_data):
    """Median HSV value of a chest patch; None if the torso cannot be sampled."""
    if frame is None or not body_data:
        return None
    patch = _torso_patch(frame, body_data)
    if patch is None:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    return float(np.median(hsv[:, :, 2]))


def adapt_garment_lighting(garment_img, ambient_v):
    """Match garment brightness/contrast to the user's current torso lighting."""
    if garment_img is None or ambient_v is None:
        return garment_img

    gain = float(np.clip(float(ambient_v) / AMBIENT_REF_V, AMBIENT_GAIN_MIN, AMBIENT_GAIN_MAX))
    contrast = float(
        np.clip(
            1.0 + 0.22 * (float(ambient_v) - AMBIENT_REF_V) / AMBIENT_REF_V,
            AMBIENT_CONTRAST_MIN,
            AMBIENT_CONTRAST_MAX,
        )
    )

    bgr, alpha = _split_color_alpha(garment_img)
    opaque = alpha > 16
    if not np.any(opaque):
        return garment_img

    work = bgr.astype(np.float32)
    mean = float(work[opaque].mean())
    work = (work - mean) * contrast + mean * gain
    work = np.clip(work, 0, 255).astype(np.uint8)
    return _merge_color_alpha(work, alpha)


def _split_color_alpha(image: np.ndarray):
    if image.shape[2] >= 4:
        return image[:, :, :3], image[:, :, 3]
    h, w = image.shape[:2]
    return image[:, :, :3], np.full((h, w), 255, dtype=np.uint8)


def _merge_color_alpha(bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return np.dstack((bgr, alpha))


def _opaque_mask(garment_img: np.ndarray) -> np.ndarray:
    if garment_img.shape[2] >= 4:
        return garment_img[:, :, 3] > 20
    return np.ones(garment_img.shape[:2], dtype=bool)


def _median_hue(garment_img: np.ndarray) -> int:
    bgr, alpha = _split_color_alpha(garment_img)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    chroma = (alpha > 20) & (hsv[:, :, 1] > 28) & (hsv[:, :, 2] > 28)
    if np.count_nonzero(chroma) < 40:
        return 0
    return int(np.median(hsv[:, :, 0][chroma]))


def _lock_hue_and_chroma(garment_img: np.ndarray, target_hue: int, min_s: int = 90):
    bgr, alpha = _split_color_alpha(garment_img)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = alpha > 20
    hsv[:, :, 0][mask] = int(target_hue) % 180
    sat = hsv[:, :, 1].astype(np.int16)
    sat[mask] = np.maximum(sat[mask], int(min_s))
    hsv[:, :, 1] = np.clip(sat, 0, 255).astype(np.uint8)
    return _merge_color_alpha(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), alpha)


def _apply_charcoal(garment_img: np.ndarray) -> np.ndarray:
    bgr, alpha = _split_color_alpha(garment_img)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    mask = alpha > 20
    hsv[:, :, 1][mask] *= 0.10
    hsv[:, :, 2][mask] = hsv[:, :, 2][mask] * 0.52 + 22.0
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return _merge_color_alpha(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), alpha)


def _torso_patch(frame, body_data):
    h, w = frame.shape[:2]
    left_shoulder = body_data.get("left_shoulder")
    right_shoulder = body_data.get("right_shoulder")
    if left_shoulder is None or right_shoulder is None:
        return None

    chest_x = (float(left_shoulder[0]) + float(right_shoulder[0])) * 0.5
    chest_y = (float(left_shoulder[1]) + float(right_shoulder[1])) * 0.5
    shoulder_w = abs(float(right_shoulder[0]) - float(left_shoulder[0]))
    if shoulder_w < 8:
        return None

    left_hip = body_data.get("left_hip")
    right_hip = body_data.get("right_hip")
    if left_hip is not None and right_hip is not None:
        hip_y = (float(left_hip[1]) + float(right_hip[1])) * 0.5
        torso_h = abs(hip_y - chest_y)
    else:
        hinted = body_data.get("torso_length")
        torso_h = float(hinted) if hinted else shoulder_w * 1.15

    cx = int(chest_x)
    cy = int(chest_y + torso_h * 0.30)
    bw = max(14, int(shoulder_w * 0.20))
    bh = max(14, int(max(torso_h, 1.0) * 0.16))
    x1 = int(np.clip(cx - bw, 0, w - 1))
    y1 = int(np.clip(cy - bh, 0, h - 1))
    x2 = int(np.clip(cx + bw, 1, w))
    y2 = int(np.clip(cy + bh, 1, h))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return frame[y1:y2, x1:x2]

