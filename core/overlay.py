import cv2
import numpy as np


class GarmentOverlay:

    def __init__(self, garment_path: str):
        self.garment = cv2.imread(garment_path, cv2.IMREAD_UNCHANGED)

    def apply_overlay(self, frame, body_data):
        if not body_data:
            return frame

        center = body_data["chest_center"]
        shoulder_width = body_data["shoulder_width"]

        target_width = int(shoulder_width * 2.2)
        if target_width <= 0:
            return frame

        aspect_ratio = self.garment.shape[0] / self.garment.shape[1]
        target_height = int(target_width * aspect_ratio)

        # Simple High-Quality Resizing (No WarpAffine = No Gray Blocks)
        resized = cv2.resize(
            self.garment,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
        h, w = resized.shape[:2]

        # Precise Neck Position
        x_offset = int(center[0] - w // 2)
        y_offset = int(center[1] - (h * 0.28))

        frame_h, frame_w, _ = frame.shape

        x1, x2 = max(0, x_offset), min(frame_w, x_offset + w)
        y1, y2 = max(0, y_offset), min(frame_h, y_offset + h)

        gx1, gx2 = max(0, -x_offset), min(w, frame_w - x_offset)
        gy1, gy2 = max(0, -y_offset), min(h, frame_h - y_offset)

        if x1 >= x2 or y1 >= y2 or gx1 >= gx2 or gy1 >= gy2:
            return frame

        crop = resized[gy1:gy2, gx1:gx2]
        if crop.shape[2] == 4:
            alpha = (crop[:, :, 3] / 255.0)[:, :, np.newaxis]
            frame[y1:y2, x1:x2] = (alpha * crop[:, :, :3]) + (
                (1.0 - alpha) * frame[y1:y2, x1:x2]
            )

        return frame