import cv2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CameraStream")


class CameraStream:

    def __init__(self, device_index: int = 0, width: int = 1280, height: int = 720):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(self.device_index)

        if not self.cap.isOpened():
            logger.error(f"Cannot open camera index {self.device_index}")
            raise RuntimeError(f"Failed to open camera index {self.device_index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        logger.info(f"Camera initialized: {self.width}x{self.height}")

    def get_frame(self):
        success, frame = self.cap.read()
        if not success:
            logger.warning("Failed to grab frame from camera stream.")
            return False, None

        # Flip horizontally for natural mirror effect
        frame = cv2.flip(frame, 1)
        return True, frame

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            logger.info("Camera device released.")