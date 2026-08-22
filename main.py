import os
import time
from queue import Empty, Queue

import cv2
from core.api_client import GarmentApiClient
from core.camera import CameraStream
from core.gesture_detector import SWIPE_LEFT, SWIPE_RIGHT, HandGestureDetector
from core.overlay import GarmentOverlay
from core.qr_generator import capture_download_url, generate_qr_image, overlay_qr_code
from core.snapshot import save_snapshot_async
from core.tracker import PoseTracker
from core.ui_overlay import MirrorHUD
from core.visualization import UIOverlay

STATE_LIVE = "LIVE"
STATE_SHOW_SNAPSHOT = "SHOW_SNAPSHOT"

GESTURE_BANNER_SEC = 1.0
POSE_HOLD_FRAMES = 2
CHEESE_HOLD_SEC = 0.45
SNAPSHOT_HOLD_SEC = 6.0


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
    print(" VTO SMART MIRROR - PHASE 4 (QR SHARE)")
    print("==========================================")
    print(" Controls:")
    print("  'N' / Swipe Right -> Next Garment")
    print("  'P' / Swipe Left  -> Previous Garment")
    print("  'S' -> Snapshot + QR (3-2-1 CHEESE!)")
    print("  'Q' -> Exit Application")
    print("==========================================")
    print("[INFO] Keep the FastAPI server on :8000 so phones can download captures.")

    cam = CameraStream(device_index=0, width=1280, height=720)
    tracker = PoseTracker()
    gestures = HandGestureDetector(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        min_swipe_distance=0.08,
        cooldown_sec=1.5,
        process_every_n=4,
    )
    hud = MirrorHUD()
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
    saved_queue = Queue()

    state = STATE_LIVE
    frozen_frame = None
    qr_image = None
    snapshot_until = 0.0

    while True:
        now = time.time()

        if state == STATE_SHOW_SNAPSHOT and frozen_frame is not None:
            cam.get_frame()
            display = frozen_frame.copy()
            remaining = max(0, int(snapshot_until - now))
            cv2.putText(
                display,
                "Scan to Download",
                (40, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.15,
                (0, 255, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                f"Resuming in {remaining}s",
                (40, 185),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (220, 220, 220),
                2,
                cv2.LINE_AA,
            )
            display = overlay_qr_code(display, qr_image)
            display = hud.draw(
                display,
                catalog.get_current_name(),
                catalog.get_current_category(),
            )
            cv2.imshow("VTO Smart Mirror - Virtual Try-On", display)
            if now >= snapshot_until:
                state = STATE_LIVE
                frozen_frame = None
                qr_image = None
                print("[INFO] Resuming live mirror.")
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                break
            continue

        try:
            saved_path, freeze = saved_queue.get_nowait()
            filename = os.path.basename(saved_path)
            url = capture_download_url(filename)
            qr_image = generate_qr_image(url)
            frozen_frame = freeze
            state = STATE_SHOW_SNAPSHOT
            snapshot_until = time.time() + SNAPSHOT_HOLD_SEC
            print(f"[INFO] Scan to download: {url}")
            continue
        except Empty:
            pass

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
        frame = hud.draw(
            frame,
            catalog.get_current_name(),
            catalog.get_current_category(),
        )
        if time.time() < gesture_banner_until:
            frame = ui.draw_gesture(frame, gesture_banner)

        if countdown_started_at is not None:
            elapsed = time.time() - countdown_started_at
            label = _countdown_label(elapsed)
            if label:
                frame = ui.draw_countdown(frame, label)
                if label == "CHEESE!" and not snapshot_taken:
                    freeze = composite.copy()
                    save_snapshot_async(
                        composite,
                        on_saved=lambda path, freeze=freeze: saved_queue.put((path, freeze)),
                    )
                    snapshot_taken = True
            else:
                countdown_started_at = None
                snapshot_taken = False

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
            if countdown_started_at is None and state == STATE_LIVE:
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
