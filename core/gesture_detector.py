import time
from collections import deque

import cv2
import mediapipe as mp

SWIPE_RIGHT = "SWIPE_RIGHT"
SWIPE_LEFT = "SWIPE_LEFT"

INDEX_FINGER_TIP = 8
HISTORY_LEN = 5
MIN_SAMPLES = 3
MIN_SWIPE_DISTANCE = 0.035
COOLDOWN_SEC = 0.45
PROCESS_EVERY_N = 1
ROI_TOP_FRAC = 1.0
POINTER_EMA_ALPHA = 0.62
MISS_TOLERANCE = 3
HANDS_INFER_MAX_WIDTH = 640


class HandGestureDetector:
    """Detect left/right index-finger swipes from MediaPipe Hands landmarks."""

    def __init__(
        self,
        min_detection_confidence: float = 0.4,
        min_tracking_confidence: float = 0.4,
        min_swipe_distance: float = MIN_SWIPE_DISTANCE,
        cooldown_sec: float = COOLDOWN_SEC,
        process_every_n: int = PROCESS_EVERY_N,
        pointer_ema_alpha: float = POINTER_EMA_ALPHA,
    ):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.history = deque(maxlen=HISTORY_LEN)
        self.min_swipe_distance = min_swipe_distance
        self.cooldown_sec = cooldown_sec
        self.process_every_n = max(1, process_every_n)
        self.pointer_ema_alpha = min(1.0, max(0.05, pointer_ema_alpha))
        self._frame_index = 0
        self._last_event_at = 0.0
        self._misses = 0
        self._ema_x = None
        self._ema_y = None
        self.last_event = None
        self.last_fingertip = None

    def process_frame(self, frame_bgr):
        """Return a new swipe event or None. Pointer is EMA-smoothed every hit."""
        if frame_bgr is None:
            return None

        self._frame_index += 1
        if self._frame_index % self.process_every_n != 0:
            return None

        frame_h, frame_w = frame_bgr.shape[:2]
        roi_h = max(1, int(frame_h * ROI_TOP_FRAC))
        roi = frame_bgr[:roi_h, :]
        infer = self._prepare_infer_image(roi)

        rgb = cv2.cvtColor(infer, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.hands.process(rgb)

        if not result.multi_hand_landmarks:
            return self._on_miss()

        tip = result.multi_hand_landmarks[0].landmark[INDEX_FINGER_TIP]
        px = float(tip.x) * frame_w
        py = float(tip.y) * roi_h
        if py < 0 or py > roi_h:
            return self._on_miss()

        self._misses = 0
        self.last_fingertip = self._smooth_pointer(px, py)
        self.history.append(float(tip.x))
        event = self._detect_swipe()
        if event:
            self.last_event = event
        return event

    def draw_fingertip(self, frame):
        if self.last_fingertip is None:
            return frame
        cv2.circle(frame, self.last_fingertip, 9, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, self.last_fingertip, 12, (0, 200, 0), 2, cv2.LINE_AA)
        return frame

    @staticmethod
    def _prepare_infer_image(roi):
        width = roi.shape[1]
        if width <= HANDS_INFER_MAX_WIDTH:
            return roi
        scale = HANDS_INFER_MAX_WIDTH / float(width)
        return cv2.resize(
            roi,
            (HANDS_INFER_MAX_WIDTH, max(1, int(roi.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )

    def _smooth_pointer(self, x: float, y: float):
        alpha = self.pointer_ema_alpha
        if self._ema_x is None or self._ema_y is None:
            self._ema_x, self._ema_y = x, y
        else:
            self._ema_x = alpha * x + (1.0 - alpha) * self._ema_x
            self._ema_y = alpha * y + (1.0 - alpha) * self._ema_y
        return (int(round(self._ema_x)), int(round(self._ema_y)))

    def _on_miss(self):
        self._misses += 1
        if self._misses > MISS_TOLERANCE:
            self.history.clear()
            self.last_fingertip = None
            self._ema_x = None
            self._ema_y = None
        return None

    def _detect_swipe(self):
        if len(self.history) < MIN_SAMPLES:
            return None
        if time.monotonic() - self._last_event_at < self.cooldown_sec:
            return None

        start_x = self.history[0]
        end_x = self.history[-1]
        delta_x = end_x - start_x

        step_sum = 0.0
        prev = start_x
        for x in list(self.history)[1:]:
            step_sum += x - prev
            prev = x
        if abs(step_sum) < abs(delta_x) * 0.5:
            return None

        event = None
        if delta_x > self.min_swipe_distance:
            event = SWIPE_RIGHT
        elif delta_x < -self.min_swipe_distance:
            event = SWIPE_LEFT

        if event:
            self._last_event_at = time.monotonic()
            self.history.clear()
        return event

    def close(self):
        if self.hands:
            self.hands.close()
            self.hands = None
