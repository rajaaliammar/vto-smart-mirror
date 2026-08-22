import cv2
import math
import urllib.request
import os
import logging
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PoseTracker")

MODEL_PATH = "pose_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"

class PoseTracker:
    def __init__(self):
        # Auto-download model file if not present locally
        if not os.path.exists(MODEL_PATH):
            logger.info("Downloading MediaPipe Pose Model (One-time download)...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            logger.info("Model downloaded successfully.")

        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        logger.info("MediaPipe 3.13 Tasks Engine Initialized successfully.")

    def process_frame(self, frame_bgr):
        """Detect pose and return a body_data dict, or None if no pose is found."""
        if frame_bgr is None:
            return None

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = self.detector.detect(mp_image)

        img_height, img_width = frame_bgr.shape[:2]
        return self.get_body_measurements(detection_result, img_width, img_height)

    def get_body_measurements(self, detection_result, img_width, img_height):
        if not detection_result or not detection_result.pose_landmarks:
            return None

        landmarks = detection_result.pose_landmarks[0]
        if len(landmarks) < 13:
            return None

        def _xy(lm, min_visibility: float = 0.35):
            vis = getattr(lm, "visibility", 1.0)
            if vis is not None and vis < min_visibility:
                return None
            return np.array(
                [lm.x * img_width, lm.y * img_height],
                dtype=np.float32,
            )

        # 11/12 = shoulders, 23/24 = hips (MediaPipe Pose).
        left_shoulder = _xy(landmarks[11], min_visibility=0.20)
        right_shoulder = _xy(landmarks[12], min_visibility=0.20)
        if left_shoulder is None or right_shoulder is None:
            return None

        left_hip = _xy(landmarks[23]) if len(landmarks) > 24 else None
        right_hip = _xy(landmarks[24]) if len(landmarks) > 24 else None

        shoulder_width = float(np.linalg.norm(left_shoulder - right_shoulder))
        center = (left_shoulder + right_shoulder) * 0.5

        torso_length = None
        hip_center = None
        if left_hip is not None and right_hip is not None:
            hip_center = (left_hip + right_hip) * 0.5
            torso_length = float(abs(hip_center[1] - center[1]))

        angle_rad = math.atan2(
            right_shoulder[1] - left_shoulder[1],
            right_shoulder[0] - left_shoulder[0],
        )

        return {
            "left_shoulder": (int(left_shoulder[0]), int(left_shoulder[1])),
            "right_shoulder": (int(right_shoulder[0]), int(right_shoulder[1])),
            "left_hip": None if left_hip is None else (int(left_hip[0]), int(left_hip[1])),
            "right_hip": None if right_hip is None else (int(right_hip[0]), int(right_hip[1])),
            "shoulder_width": shoulder_width,
            "torso_length": torso_length,
            "chest_center": (int(center[0]), int(center[1])),
            "shoulder_center": (float(center[0]), float(center[1])),
            "hip_center": None if hip_center is None else (float(hip_center[0]), float(hip_center[1])),
            "angle_deg": math.degrees(angle_rad),
        }

    def draw_landmarks(self, frame_bgr, detection_result):
        if detection_result and detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0]
            for lm in landmarks:
                cx, cy = int(lm.x * frame_bgr.shape[1]), int(lm.y * frame_bgr.shape[0])
                cv2.circle(frame_bgr, (cx, cy), 3, (0, 255, 0), -1)
        return frame_bgr

    def close(self):
        if hasattr(self, 'detector'):
            self.detector.close()
        logger.info("Pose Engine Closed.")