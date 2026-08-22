import os
import threading
import time
import logging

import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CameraStream")


class CameraStream:
    """Latest-frame grabber. Capture runs on a daemon thread so pose work cannot stall the device buffer."""

    def __init__(self, device_index: int = 0, width: int = 1280, height: int = 720):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.cap = self._open_capture(device_index)

        if not self.cap.isOpened():
            logger.error(f"Cannot open camera index {self.device_index}")
            raise RuntimeError(f"Failed to open camera index {self.device_index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self._lock = threading.Lock()
        self._latest = None
        self._ok = False
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="CameraGrabber",
            daemon=True,
        )
        self._thread.start()
        self._wait_for_first_frame(timeout_sec=3.0)
        logger.info(f"Camera initialized (non-blocking): {self.width}x{self.height}")

    @staticmethod
    def _open_capture(device_index: int):
        if os.name == "nt":
            cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
            cap.release()
        return cv2.VideoCapture(device_index)

    def _wait_for_first_frame(self, timeout_sec: float) -> None:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            with self._lock:
                if self._latest is not None:
                    return
            time.sleep(0.02)
        logger.warning("Camera opened but no frame arrived within timeout.")

    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            success, frame = self.cap.read()
            if not success or frame is None:
                time.sleep(0.004)
                continue
            frame = cv2.flip(frame, 1)
            with self._lock:
                self._latest = frame
                self._ok = True

    def get_frame(self):
        with self._lock:
            if self._latest is None:
                return False, None
            return True, self._latest.copy()

    def release(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()
            logger.info("Camera device released.")
