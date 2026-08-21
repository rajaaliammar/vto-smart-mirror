import time
from collections import deque

import cv2
import mediapipe as mp

SWIPE_RIGHT = "SWIPE_RIGHT"
SWIPE_LEFT = "SWIPE_LEFT"

INDEX_FINGER_TIP = 8
HISTORY_LEN = 8
MIN_SAMPLES = 5
MIN_SWIPE_DISTANCE = 0.08
COOLDOWN_SEC = 1.5
PROCESS_EVERY_N = 4
ROI_TOP_FRAC = 0.40


class HandGestureDetector:
    """Detect left/right index-finger swipes from MediaPipe Hands landmarks."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        min_swipe_distance: float = MIN_SWIPE_DISTANCE,
        cooldown_sec: float = COOLDOWN_SEC,
        process_every_n: int = PROCESS_EVERY_N,
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
        self._frame_index = 0
        self._last_event_at = 0.0
        self.last_event = None
        self.last_fingertip = None

    def process_frame(self, frame_bgr):
        """Return a new swipe event or None. Hands run only every Nth frame."""
        if frame_bgr is None:
            return None

        self._frame_index += 1
        if self._frame_index % self.process_every_n != 0:
            return None

        frame_h, frame_w = frame_bgr.shape[:2]
        roi_h = max(1, int(frame_h * ROI_TOP_FRAC))
        roi = frame_bgr[:roi_h, :]

        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.hands.process(rgb)

        if not result.multi_hand_landmarks:
            self.history.clear()
            self.last_fingertip = None
            return None

        tip = result.multi_hand_landmarks[0].landmark[INDEX_FINGER_TIP]
        px = int(tip.x * frame_w)
        py = int(tip.y * roi_h)
        if py < 0 or py > roi_h:
            self.history.clear()
            self.last_fingertip = None
            return None

        self.last_fingertip = (px, py)
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
        if abs(step_sum) < abs(delta_x) * 0.7:
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
