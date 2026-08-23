import time

import cv2
import numpy as np

CURSOR_CORE = (80, 255, 140)
CURSOR_RING = (0, 230, 255)
CURSOR_GLOW = (0, 160, 255)


class MirrorHUD:
    """Retail top bar: garment metadata, live FPS pill, and swipe guides."""

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

    def draw(
        self,
        frame,
        garment_name: str,
        category: str = "tshirt",
        size: str = "M",
        price: str = "$49.99 / PKR 3,500",
        live: bool = True,
    ):
        if frame is None:
            return frame

        fps = self.update_fps()
        _, w = frame.shape[:2]
        bar_h = 92
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (18, 16, 14), -1)
        cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, dst=frame)
        cv2.line(frame, (0, bar_h), (w, bar_h), (0, 180, 255), 2)

        name = garment_name or "None"
        cv2.putText(
            frame,
            name,
            (24, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        (name_w, _), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.78, 2)
        category_label = (category or "tshirt").replace("_", " ").title()
        cv2.putText(
            frame,
            f" / {category_label}",
            (24 + name_w + 6, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (170, 190, 200),
            1,
            cv2.LINE_AA,
        )

        badge_x = self._draw_size_tag(frame, size or "M", 24, 48)
        price_label = price or "$49.99 / PKR 3,500"
        cv2.putText(
            frame,
            price_label,
            (badge_x + 14, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (180, 220, 255),
            1,
            cv2.LINE_AA,
        )

        self._draw_live_pill(frame, w, fps, live=live)
        self._draw_swipe_guides(frame, w)
        return frame

    @staticmethod
    def draw_hand_cursor(frame, point):
        """Sleek ring cursor that tracks the EMA-smoothed fingertip."""
        if frame is None or point is None:
            return frame
        x, y = int(point[0]), int(point[1])
        h, w = frame.shape[:2]
        if x < 0 or y < 0 or x >= w or y >= h:
            return frame

        overlay = frame.copy()
        cv2.circle(overlay, (x, y), 28, CURSOR_GLOW, 2, cv2.LINE_AA)
        cv2.circle(overlay, (x, y), 18, CURSOR_RING, 2, cv2.LINE_AA)
        cv2.circle(overlay, (x, y), 7, CURSOR_CORE, -1, cv2.LINE_AA)
        cv2.circle(overlay, (x, y), 7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, dst=frame)
        cv2.circle(frame, (x, y), 22, CURSOR_RING, 2, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 5, CURSOR_CORE, -1, cv2.LINE_AA)
        return frame

    @staticmethod
    def _draw_size_tag(frame, size: str, x: int, y: int) -> int:
        label = str(size).upper()[:3]
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.55, 2)
        pad_x, pad_y = 10, 6
        x2, y2 = x + tw + pad_x * 2, y + th + pad_y * 2
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x2, y2), (0, 140, 255), -1)
        cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, dst=frame)
        cv2.putText(
            frame,
            label,
            (x + pad_x, y + th + pad_y - 1),
            font,
            0.55,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        return x2

    @staticmethod
    def _draw_live_pill(frame, frame_w: int, fps: int, live: bool = True):
        if live:
            text = f"LIVE TRY-ON | {fps} FPS"
            fill = (32, 96, 36)
            accent = (70, 230, 90)
        else:
            text = f"SNAPSHOT | {fps} FPS"
            fill = (36, 72, 110)
            accent = (0, 200, 255)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.52, 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        pad_x, pad_y = 18, 10
        pill_w = tw + pad_x * 2 + 18
        pill_h = th + pad_y * 2
        x2 = frame_w - 18
        x1 = x2 - pill_w
        y1 = 12
        y2 = y1 + pill_h
        cy = (y1 + y2) // 2

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1 + pill_h // 2, y1), (x2 - pill_h // 2, y2), fill, -1)
        cv2.circle(overlay, (x1 + pill_h // 2, cy), pill_h // 2, fill, -1)
        cv2.circle(overlay, (x2 - pill_h // 2, cy), pill_h // 2, fill, -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, dst=frame)

        cv2.circle(frame, (x1 + 16, cy), 6, accent, -1)
        cv2.putText(
            frame,
            text,
            (x1 + 30, cy + th // 2 - 1),
            font,
            scale,
            (245, 245, 245),
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_swipe_guides(frame, frame_w: int):
        y = 72
        left_x = frame_w - 310
        right_x = frame_w - 150

        left = np.array(
            [[left_x, y], [left_x + 18, y - 11], [left_x + 18, y + 11]],
            dtype=np.int32,
        )
        right = np.array(
            [[right_x + 78, y], [right_x + 60, y - 11], [right_x + 60, y + 11]],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(frame, left, (0, 200, 255))
        cv2.fillConvexPoly(frame, right, (0, 200, 255))
        cv2.putText(
            frame,
            "Swipe L",
            (left_x + 26, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Swipe R",
            (right_x, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
