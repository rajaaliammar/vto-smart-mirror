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