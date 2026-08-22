import time

import cv2
import numpy as np


class MirrorHUD:
    """Semi-transparent top bar: garment, FPS, and swipe guides."""

    def __init__(self):
        self._prev_time = time.perf_counter()
        self.fps = 0.0

    def update_fps(self) -> int:
        now = time.perf_counter()
        dt = now - self._prev_time
        self._prev_time = now
        inst = 1.0 / max(dt, 1e-6)
        if self.fps <= 0:
            self.fps = inst
        else:
            self.fps = 0.85 * self.fps + 0.15 * inst
        return int(self.fps)

    def draw(self, frame, garment_name: str, category: str = "tshirt"):
        if frame is None:
            return frame

        fps = self.update_fps()
        h, w = frame.shape[:2]
        bar_h = 78
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (16, 16, 16), -1)
        cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, dst=frame)
        cv2.line(frame, (0, bar_h), (w, bar_h), (0, 180, 255), 2)

        name = garment_name or "None"
        category_label = (category or "tshirt").replace("_", " ").title()
        cv2.putText(
            frame,
            name,
            (24, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"{category_label}   |   FPS {fps}",
            (24, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 220, 255),
            1,
            cv2.LINE_AA,
        )

        self._draw_swipe_guides(frame, w)
        return frame

    @staticmethod
    def _draw_swipe_guides(frame, frame_w: int):
        y = 40
        left_x = frame_w - 310
        right_x = frame_w - 150

        left = np.array(
            [[left_x, y], [left_x + 22, y - 14], [left_x + 22, y + 14]],
            dtype=np.int32,
        )
        right = np.array(
            [[right_x + 88, y], [right_x + 66, y - 14], [right_x + 66, y + 14]],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(frame, left, (0, 200, 255))
        cv2.fillConvexPoly(frame, right, (0, 200, 255))
        cv2.putText(
            frame,
            "Swipe L",
            (left_x + 30, y + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Swipe R",
            (right_x, y + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
