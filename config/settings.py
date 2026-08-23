import os
from pydantic import BaseModel


class CameraConfig(BaseModel):
    DEVICE_INDEX: int = 0
    WIDTH: int = 1280
    HEIGHT: int = 720
    TARGET_FPS: int = 30


class PoseConfig(BaseModel):
    STATIC_IMAGE_MODE: bool = False
    MODEL_COMPLEXITY: int = 1  # 0=Fast, 1=Balanced, 2=Heavy
    SMOOTH_LANDMARKS: bool = True
    MIN_DETECTION_CONFIDENCE: float = 0.4
    MIN_TRACKING_CONFIDENCE: float = 0.4


camera_settings = CameraConfig()
pose_settings = PoseConfig()