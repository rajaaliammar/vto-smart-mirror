import cv2
import time


class FPSCounter:

    def __init__(self):
        self.prev_time = time.time()
        self.fps = 0

    def update(self):
        current_time = time.time()
        self.fps = 1 / (current_time - self.prev_time + 1e-6)
        self.prev_time = current_time
        return int(self.fps)


def draw_hud(frame, fps: int, tracking_active: bool, body_data=None):
    # Draw FPS Status
    cv2.putText(
        frame,
        f"FPS: {fps}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # Draw Tracking Status
    status_text = "Tracking: ACTIVE" if tracking_active else "Tracking: SEARCHING"
    status_color = (0, 255, 0) if tracking_active else (0, 0, 255)
    cv2.putText(
        frame,
        status_text,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2,
        cv2.LINE_AA,
    )

    # Draw Body Measurements if detected
    if body_data:
        ls = body_data["left_shoulder"]
        rs = body_data["right_shoulder"]
        center = body_data["chest_center"]
        width = int(body_data["shoulder_width"])
        angle = int(body_data["angle_deg"])

        # Line between shoulders
        cv2.line(frame, ls, rs, (255, 0, 0), 3)
        # Chest center point
        cv2.circle(frame, center, 8, (0, 255, 255), -1)

        # On-screen text for measurements
        cv2.putText(
            frame,
            f"Shoulder Width: {width} px | Angle: {angle} deg",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # System Branding Footer
    h, w, _ = frame.shape
    cv2.putText(
        frame,
        "VTO Smart Mirror Engine v0.1",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return frame