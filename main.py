import cv2
import os
import sys
from config.settings import camera_settings
from core.camera import CameraStream
from core.overlay import GarmentOverlay
from core.tracker import PoseTracker
from utils.visualization import FPSCounter, draw_hud


def main():
    print("==================================================")
    print("      VTO SMART MIRROR - PHASE 2 (OVERLAY)        ")
    print("==================================================")

    garment_path = os.path.join(
        "assets", "sample_clothes", "tshirts", "tshirt.png"
    )
    if not os.path.exists(garment_path):
        print(
            f"[ERROR] Please place a PNG image at '{garment_path}' before running."
        )
        sys.exit(1)

    try:
        camera = CameraStream(
            device_index=camera_settings.DEVICE_INDEX,
            width=camera_settings.WIDTH,
            height=camera_settings.HEIGHT,
        )
    except Exception as e:
        print(f"[ERROR] Could not start camera stream: {e}")
        sys.exit(1)

    tracker = PoseTracker()
    overlay = GarmentOverlay(garment_path)
    fps_counter = FPSCounter()

    print("[INFO] Press 'q' key on screen window to Exit Application.")

    while True:
        success, frame = camera.get_frame()
        if not success:
            continue

        h, w, _ = frame.shape
        detection_result = tracker.process_frame(frame)

        body_data = tracker.get_body_measurements(
            detection_result, img_width=w, img_height=h
        )
        is_tracking = body_data is not None

        # Apply Garment Overlay
        if body_data:
            frame = overlay.apply_overlay(frame, body_data)

        fps = fps_counter.update()
        frame = draw_hud(
            frame, fps, tracking_active=is_tracking, body_data=body_data
        )

        cv2.imshow("VTO Smart Mirror - Virtual Try-On", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Shutting down application...")
            break

    camera.release()
    tracker.close()
    cv2.destroyAllWindows()
    print("[INFO] System exited cleanly.")


if __name__ == "__main__":
    main()