import cv2


class UIOverlay:

    def draw_status(self, frame, body_data, extra_info: str = ""):
        # Tracking Status
        status_text = (
            "Tracking: ACTIVE" if body_data else "Tracking: SEARCHING..."
        )
        color = (0, 255, 0) if body_data else (0, 0, 255)

        cv2.putText(
            frame,
            status_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

        # Garment & Control Info
        if extra_info:
            cv2.putText(
                frame,
                extra_info,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

        # Footer
        cv2.putText(
            frame,
            "VTO Smart Mirror Engine v0.1",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        return frame

    @staticmethod
    def draw_gesture(frame, message: str):
        if not message:
            return frame
        cv2.putText(
            frame,
            message,
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )
        return frame

    @staticmethod
    def draw_countdown(frame, label: str):
        if not label:
            return frame
        h, w = frame.shape[:2]
        scale = 2.6 if label == "CHEESE!" else 4.0
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 8
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        x = (w - tw) // 2
        y = (h + th) // 2
        cv2.putText(
            frame, label, (x, y), font, scale, (0, 0, 0), thickness + 6, cv2.LINE_AA
        )
        cv2.putText(
            frame, label, (x, y), font, scale, (0, 255, 255), thickness, cv2.LINE_AA
        )
        return frame

    @staticmethod
    def draw_saved_banner(frame, message: str = "Photo Saved!"):
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.4
        (tw, th), _ = cv2.getTextSize(message, font, scale, 4)
        x = (w - tw) // 2
        y = 90
        cv2.rectangle(
            frame,
            (x - 24, y - th - 18),
            (x + tw + 24, y + 18),
            (0, 140, 0),
            -1,
        )
        cv2.putText(frame, message, (x, y), font, scale, (255, 255, 255), 4, cv2.LINE_AA)
        return frame