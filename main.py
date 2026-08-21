import cv2
from core.api_client import GarmentApiClient
from core.camera import CameraStream
from core.overlay import GarmentOverlay
from core.tracker import PoseTracker
from core.visualization import UIOverlay


def main():
    print("==========================================")
    print(" VTO SMART MIRROR - PHASE 3 (GARMENT SELECTOR)")
    print("==========================================")
    print(" Controls:")
    print("  'N' -> Next Garment")
    print("  'P' -> Previous Garment")
    print("  'Q' -> Exit Application")
    print("==========================================")

    cam = CameraStream(device_index=0, width=1280, height=720)
    tracker = PoseTracker()
    ui = UIOverlay()
    catalog = GarmentApiClient()

    overlay = GarmentOverlay()
    current_path = catalog.get_current_image_path()
    if current_path:
        overlay.load_garment(current_path)
        print(f"[INFO] Active garment: {catalog.get_current_name()}")

    while True:
        success, frame = cam.get_frame()
        if not success or frame is None:
            continue

        body_data = tracker.process_frame(frame)
        frame = overlay.apply_overlay(frame, body_data)

        garment_name = catalog.get_current_name()
        frame = ui.draw_status(
            frame,
            body_data,
            extra_info=f"Garment: {garment_name} (N:Next | P:Prev)",
        )

        cv2.imshow("VTO Smart Mirror - Virtual Try-On", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break
        elif key == ord("n") or key == ord("N"):
            new_path = catalog.next_garment()
            if overlay.load_garment(new_path):
                print(f"[INFO] Switched to: {catalog.get_current_name()}")
        elif key == ord("p") or key == ord("P"):
            new_path = catalog.prev_garment()
            if overlay.load_garment(new_path):
                print(f"[INFO] Switched to: {catalog.get_current_name()}")

    cam.release()
    tracker.close()
    cv2.destroyAllWindows()
    print("[INFO] System exited cleanly.")


if __name__ == "__main__":
    main()
