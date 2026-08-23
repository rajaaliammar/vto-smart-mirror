import time

import cv2
import numpy as np

CURSOR_CORE = (80, 255, 140)
CURSOR_RING = (0, 230, 255)
CURSOR_GLOW = (0, 160, 255)

# Distance-invariant: shoulder span / torso height. Typical adult ~0.83.
SIZE_RATIO_CENTERS = {"S": 0.75, "M": 0.84, "L": 0.93}


def recommend_size(body_data, frame_shape=None):
    """Recommend S/M/L from shoulder-to-torso proportions only."""
    default = {
        "size": "M",
        "score": 0,
        "label": "Size M - -- Fit Match",
        "ratio": None,
    }
    if not body_data:
        return default

    shoulder_px = float(body_data.get("shoulder_width") or 0)
    torso_px = float(body_data.get("torso_length") or 0)
    if shoulder_px < 12 or torso_px < 12:
        return default

    ratio = shoulder_px / torso_px
    size = min(SIZE_RATIO_CENTERS, key=lambda key: abs(SIZE_RATIO_CENTERS[key] - ratio))
    dist = abs(SIZE_RATIO_CENTERS[size] - ratio)
    score = int(np.clip(99.0 - dist * 160.0, 78, 99))
    return {
        "size": size,
        "score": score,
        "label": f"Size {size} - {score}% Fit Match",
        "ratio": round(ratio, 3),
    }


class MirrorHUD:
    """Retail top bar: garment metadata, live FPS pill, and swipe guides."""

    def __init__(self):
        self._prev_time = time.perf_counter()
        self.fps = 0.0
        self._fit_votes = []
        self._stable_size = "M"
        self._smooth_score = 90.0

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
        color_variant=None,
        color_variants=None,
        color_index: int = 0,
        fit_advice=None,
        outfit_mode: str = "upper",
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
        (price_w, _), _ = cv2.getTextSize(price_label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        self._draw_color_swatches(
            frame,
            badge_x + 14 + price_w + 16,
            50,
            color_variant,
            color_variants,
            color_index,
        )

        self._draw_live_pill(frame, w, fps, live=live)
        self._draw_swipe_guides(frame, w)
        self._draw_fit_advisor(frame, fit_advice, outfit_mode, bar_h)
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
    def _draw_color_swatches(frame, x, y, color_variant, color_variants, color_index: int):
        variants = list(color_variants or ())
        if not variants:
            return
        size, gap = 16, 5
        active = int(color_index) if color_variants else 0
        if color_variant:
            for i, item in enumerate(variants):
                if item.get("key") == color_variant.get("key"):
                    active = i
                    break
        for i, item in enumerate(variants):
            sx = int(x + i * (size + gap))
            color = tuple(int(c) for c in item.get("swatch_bgr", (180, 180, 180)))
            cv2.rectangle(frame, (sx, y), (sx + size, y + size), color, -1)
            border = (255, 255, 255) if i == active else (90, 90, 90)
            cv2.rectangle(frame, (sx, y), (sx + size, y + size), border, 2 if i == active else 1)
        label = (color_variant or variants[active]).get("label", "Original")
        text_x = int(x + len(variants) * (size + gap) + 6)
        cv2.putText(
            frame,
            str(label),
            (text_x, y + size - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

    def _stabilize_fit(self, fit_advice):
        if not fit_advice:
            return {"size": self._stable_size, "score": int(self._smooth_score), "label": f"Size {self._stable_size} - -- Fit Match"}
        size = str(fit_advice.get("size") or "M").upper()
        score = float(fit_advice.get("score") or self._smooth_score)
        self._fit_votes.append(size)
        self._fit_votes = self._fit_votes[-12:]
        winner = max(set(self._fit_votes), key=self._fit_votes.count)
        if self._fit_votes.count(winner) >= 4 or winner == self._stable_size:
            self._stable_size = winner
        self._smooth_score = 0.75 * self._smooth_score + 0.25 * score
        shown = int(round(self._smooth_score))
        return {
            "size": self._stable_size,
            "score": shown,
            "label": f"Size {self._stable_size} - {shown}% Fit Match",
        }

    def _draw_fit_advisor(self, frame, fit_advice, outfit_mode: str, bar_h: int):
        advice = self._stabilize_fit(fit_advice)
        h, w = frame.shape[:2]
        y1 = bar_h + 8
        title = "AI FIT ADVISOR"
        body = advice["label"]
        mode = "FULL OUTFIT" if outfit_mode == "full" else "UPPER ONLY"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(title, font, 0.42, 1)
        (bw, bh), _ = cv2.getTextSize(body, font, 0.50, 2)
        (mw, mh), _ = cv2.getTextSize(mode, font, 0.42, 1)
        box_w = max(tw, bw) + 28
        box_h = th + bh + 22
        x1, y2 = 16, min(h - 8, y1 + box_h)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x1 + box_w, y2), (18, 42, 36), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, dst=frame)
        cv2.rectangle(frame, (x1, y1), (x1 + box_w, y2), (70, 210, 160), 1)
        cv2.putText(frame, title, (x1 + 12, y1 + th + 8), font, 0.42, (160, 230, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, body, (x1 + 12, y1 + th + bh + 14), font, 0.50, (240, 240, 240), 2, cv2.LINE_AA)

        mx2 = w - 18
        mx1 = mx2 - mw - 28
        my1 = y1
        my2 = y1 + mh + 16
        chip = frame.copy()
        fill = (40, 90, 40) if outfit_mode == "full" else (50, 50, 50)
        cv2.rectangle(chip, (mx1, my1), (mx2, my2), fill, -1)
        cv2.addWeighted(chip, 0.82, frame, 0.18, 0, dst=frame)
        cv2.putText(
            frame,
            mode,
            (mx1 + 14, my1 + mh + 6),
            font,
            0.42,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

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
