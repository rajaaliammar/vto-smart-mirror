"""Phase 5 body fitting and PNG edge-feathering for the garment overlay."""

from typing import Optional, Tuple

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
