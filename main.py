import time

import cv2
from core.api_client import GarmentApiClient
from core.camera import CameraStream
from core.gesture_detector import SWIPE_LEFT, SWIPE_RIGHT, HandGestureDetector
from core.overlay import GarmentOverlay
from core.snapshot import save_snapshot_async
from core.tracker import PoseTracker
from core.visualization import UIOverlay

GESTURE_BANNER_SEC = 1.0
POSE_HOLD_FRAMES = 2
COUNTDOWN_SECONDS = 3
CHEESE_HOLD_SEC = 0.45
SAVED_BANNER_SEC = 2.0


def _load_current(overlay, catalog) -> bool:
    path = catalog.get_current_image_path()
    if overlay.load_garment(path):
        print(f"[INFO] Switched to: {catalog.get_current_name()}")
        return True
    return False


def _countdown_label(elapsed: float):
    if elapsed < 1.0:
        return "3"
    if elapsed < 2.0:
        return "2"
    if elapsed < 3.0:
        return "1"
    if elapsed < 3.0 + CHEESE_HOLD_SEC:
        return "CHEESE!"
    return None


def main():
    print("==========================================")
    print(" VTO SMART MIRROR - PHASE 3 (GARMENT SELECTOR)")
    print("==========================================")
    print(" Controls:")
    print("  'N' / Swipe Right -> Next Garment")
    print("  'P' / Swipe Left  -> Previous Garment")
    print("  'S' -> Snapshot (3-2-1 CHEESE!)")
    print("  'Q' -> Exit Application")
    print("==========================================")

    cam = CameraStream(device_index=0, width=1280, height=720)
    tracker = PoseTracker()
    gestures = HandGestureDetector(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        min_swipe_distance=0.08,
        cooldown_sec=1.5,
        process_every_n=4,
    )
    ui = UIOverlay()
    catalog = GarmentApiClient()

    overlay = GarmentOverlay()
    current_path = catalog.get_current_image_path()
    if current_path:
        overlay.load_garment(current_path)
        print(f"[INFO] Active garment: {catalog.get_current_name()}")

    gesture_banner = ""
    gesture_banner_until = 0.0
    last_body_data = None
    missed_pose_frames = 0
    countdown_started_at = None
    snapshot_taken = False
    saved_banner_until = 0.0

    while True:
        success, frame = cam.get_frame()
        if not success or frame is None:
            continue

        body_data = tracker.process_frame(frame)
        if body_data:
            last_body_data = body_data
            missed_pose_frames = 0
            display_body = body_data
        else:
            missed_pose_frames += 1
            if missed_pose_frames <= POSE_HOLD_FRAMES and last_body_data:
                display_body = last_body_data
            else:
                display_body = None

        frame = overlay.apply_overlay(frame, display_body)
        composite = frame.copy()

        swipe = gestures.process_frame(frame)
        if swipe == SWIPE_RIGHT:
            catalog.next_garment()
            if _load_current(overlay, catalog):
                gesture_banner = "Gesture: NEXT ->"
                gesture_banner_until = time.time() + GESTURE_BANNER_SEC
        elif swipe == SWIPE_LEFT:
            catalog.prev_garment()
            if _load_current(overlay, catalog):
                gesture_banner = "Gesture: PREV <-"
                gesture_banner_until = time.time() + GESTURE_BANNER_SEC

        frame = gestures.draw_fingertip(frame)

        garment_name = catalog.get_current_name()
        frame = ui.draw_status(
            frame,
            display_body,
            extra_info=f"Garment: {garment_name} (N:Next | P:Prev | S:Photo)",
        )
        if time.time() < gesture_banner_until:
            frame = ui.draw_gesture(frame, gesture_banner)

        now = time.time()
        if countdown_started_at is not None:
            elapsed = now - countdown_started_at
            label = _countdown_label(elapsed)
            if label:
                frame = ui.draw_countdown(frame, label)
                if label == "CHEESE!" and not snapshot_taken:
                    save_snapshot_async(composite)
                    snapshot_taken = True
                    saved_banner_until = now + CHEESE_HOLD_SEC + SAVED_BANNER_SEC
            else:
                countdown_started_at = None
                snapshot_taken = False

        if now < saved_banner_until and countdown_started_at is None:
            frame = ui.draw_saved_banner(frame)

        cv2.imshow("VTO Smart Mirror - Virtual Try-On", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break
        elif key == ord("n") or key == ord("N"):
            catalog.next_garment()
            _load_current(overlay, catalog)
        elif key == ord("p") or key == ord("P"):
            catalog.prev_garment()
            _load_current(overlay, catalog)
        elif key == ord("s") or key == ord("S"):
            if countdown_started_at is None:
                countdown_started_at = time.time()
                snapshot_taken = False
                print("[INFO] Snapshot countdown started...")

    cam.release()
    tracker.close()
    gestures.close()
    cv2.destroyAllWindows()
    print("[INFO] System exited cleanly.")


if __name__ == "__main__":
    main()
