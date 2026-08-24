import os
import time
from queue import Empty, Queue

import cv2
from core.api_client import GarmentApiClient
from core.camera import CameraStream
from core.gesture_detector import SWIPE_LEFT, SWIPE_RIGHT, HandGestureDetector
from core.garment_overlay import COLOR_VARIANTS
from core.overlay import GarmentOverlay
from core.qr_generator import capture_download_url, generate_qr_image, overlay_qr_code
from core.snapshot import MirrorVideoRecorder, save_snapshot_async
from core.tracker import PoseTracker
from core.ui_overlay import MirrorHUD, recommend_size
from core.visualization import UIOverlay

STATE_LIVE = "LIVE"
STATE_SHOW_SNAPSHOT = "SHOW_SNAPSHOT"

GESTURE_BANNER_SEC = 0.85
POSE_HOLD_FRAMES = 2
CHEESE_HOLD_SEC = 0.45
SNAPSHOT_HOLD_SEC = 6.0
VIDEO_SAVED_SEC = 2.0
R_TOGGLE_DEBOUNCE_SEC = 0.40
CAMERA_RECORD_FPS = 20.0


def _load_current(overlay, catalog) -> bool:
    path = catalog.get_current_image_path()
    if overlay.load_garment(path):
        print(f"[INFO] Switched to: {catalog.get_current_name()}")
        return True
    return False


def _switch_garment(overlay, catalog, direction: str, banner: str):
    if direction == "next":
        catalog.next_garment()
    else:
        catalog.prev_garment()
    if not _load_current(overlay, catalog):
        return ""
    return banner


def _draw_mirror_hud(
    hud,
    frame,
    catalog,
    live: bool,
    overlay=None,
    fit_advice=None,
    latency_ms=None,
    recorder=None,
    video_saved: bool = False,
):
    color = overlay.current_color() if overlay is not None else None
    index = overlay.color_index if overlay is not None else 0
    outfit_mode = overlay.outfit_mode if overlay is not None else "upper"
    recording = bool(recorder and recorder.is_recording)
    rec_elapsed = recorder.elapsed() if recording else 0.0
    return hud.draw(
        frame,
        catalog.get_current_name(),
        catalog.get_current_category(),
        catalog.get_current_size(),
        catalog.get_current_price_label(),
        live=live,
        color_variant=color,
        color_variants=COLOR_VARIANTS,
        color_index=index,
        fit_advice=fit_advice,
        outfit_mode=outfit_mode,
        latency_ms=latency_ms,
        recording=recording,
        rec_elapsed=rec_elapsed,
        video_saved=video_saved,
    )


def _write_recording(recorder, frame) -> None:
    """Write a clean BGR copy while recording. OpenCV VideoWriter expects BGR."""
    if frame is not None and recorder.is_recording:
        recorder.write(frame.copy())


def _toggle_recording(recorder, frame, key_state):
    """Debounced 'R' toggle. Returns a relative path when a file is saved."""
    now = time.time()
    if now - key_state.get("last_r", 0.0) < R_TOGGLE_DEBOUNCE_SEC:
        return None
    key_state["last_r"] = now
    if recorder.is_recording:
        return recorder.stop()
    recorder.start(frame, fps=CAMERA_RECORD_FPS)
    return None


def _handle_key(key, recorder, frame, key_state):
    """Return (should_quit, saved_video_path_or_None)."""
    if key == ord("q") or key == ord("Q"):
        return True, None
    if key == ord("r") or key == ord("R"):
        return False, _toggle_recording(recorder, frame, key_state)
    return False, None


def _preload_catalog(overlay, catalog):
    shirts = catalog.upper_items or []
    saved_upper = catalog.upper_index
    for index in range(len(shirts)):
        catalog.upper_index = index
        catalog.current_index = index
        path = catalog.get_current_image_path()
        if path:
            overlay.load_garment(path)
    catalog.upper_index = saved_upper
    catalog.current_index = saved_upper
    overlay.clear_lower_garment()
    path = catalog.get_current_image_path()
    if path:
        overlay.load_garment(path)
        print(
            f"[INFO] Active garment: {catalog.get_current_name()} "
            f"({len(shirts)} shirts in catalog)"
        )


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
    print(" VTO SMART MIRROR - PHASE 9 (RECORD + BENCHMARKS)")
    print("==========================================")
    print(" Controls:")
    print("  'N' / Swipe Right -> Next Shirt")
    print("  'P' / Swipe Left  -> Previous Shirt")
    print("  'O' -> Toggle Outfit (Upper Only / Full Outfit)")
    print("  'C' -> Cycle Color (Original / Crimson / Royal / Emerald / Charcoal)")
    print("  'S' -> Snapshot + QR (3-2-1 CHEESE!)")
    print("  'R' -> Start / Stop MP4 recording")
    print("  'Q' -> Exit Application")
    print("==========================================")
    print("[INFO] Keep the FastAPI server on :8000 so phones can download captures.")
    print("[INFO] Recordings save to captures/videos/ as H.264 .mp4")

    cam = CameraStream(device_index=0, width=1280, height=720)
    tracker = PoseTracker()
    gestures = HandGestureDetector(
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
        min_swipe_distance=0.035,
        cooldown_sec=0.45,
        process_every_n=1,
    )
    hud = MirrorHUD()
    ui = UIOverlay()
    catalog = GarmentApiClient()

    overlay = GarmentOverlay()
    overlay.outfit_mode = "upper"
    recorder = MirrorVideoRecorder(fps=CAMERA_RECORD_FPS)
    _preload_catalog(overlay, catalog)
    print("[INFO] Mode: UPPER ONLY (press O for optional full outfit)")

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
    last_latency_ms = 0.0
    video_saved_until = 0.0
    key_state = {"last_r": 0.0}

    while True:
        now = time.time()

        if state == STATE_SHOW_SNAPSHOT and frozen_frame is not None:
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
            display = _draw_mirror_hud(
                hud,
                display,
                catalog,
                live=False,
                overlay=overlay,
                latency_ms=last_latency_ms,
                recorder=recorder,
                video_saved=now < video_saved_until,
            )
            _write_recording(recorder, display)
            cv2.imshow("VTO Smart Mirror - Virtual Try-On", display)
            if now >= snapshot_until:
                state = STATE_LIVE
                frozen_frame = None
                qr_image = None
                print("[INFO] Resuming live mirror.")
            key = cv2.waitKey(1) & 0xFF
            quit_app, saved_video = _handle_key(
                key, recorder, display, key_state
            )
            if saved_video:
                video_saved_until = time.time() + VIDEO_SAVED_SEC
            if quit_app:
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

        infer_t0 = time.perf_counter()
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

        swipe = gestures.process_frame(frame)
        if swipe == SWIPE_RIGHT:
            banner = _switch_garment(
                overlay, catalog, "next", "Gesture: NEXT ->"
            )
            if banner:
                gesture_banner = banner
                gesture_banner_until = time.time() + GESTURE_BANNER_SEC
        elif swipe == SWIPE_LEFT:
            banner = _switch_garment(
                overlay, catalog, "prev", "Gesture: PREV <-"
            )
            if banner:
                gesture_banner = banner
                gesture_banner_until = time.time() + GESTURE_BANNER_SEC

        frame = overlay.apply_overlay(frame, display_body)
        composite = frame.copy()
        last_latency_ms = (time.perf_counter() - infer_t0) * 1000.0

        fit_advice = recommend_size(display_body, frame.shape[:2])
        frame = _draw_mirror_hud(
            hud,
            frame,
            catalog,
            live=True,
            overlay=overlay,
            fit_advice=fit_advice,
            latency_ms=last_latency_ms,
            recorder=recorder,
            video_saved=time.time() < video_saved_until,
        )
        frame = hud.draw_hand_cursor(frame, gestures.last_fingertip)
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

        _write_recording(recorder, frame)
        cv2.imshow("VTO Smart Mirror - Virtual Try-On", frame)

        key = cv2.waitKey(1) & 0xFF
        quit_app, saved_video = _handle_key(key, recorder, frame, key_state)
        if saved_video:
            video_saved_until = time.time() + VIDEO_SAVED_SEC
        if quit_app:
            break
        elif key == ord("n") or key == ord("N"):
            banner = _switch_garment(
                overlay, catalog, "next", "Key: NEXT ->"
            )
            if banner:
                gesture_banner = banner
                gesture_banner_until = time.time() + GESTURE_BANNER_SEC
        elif key == ord("p") or key == ord("P"):
            banner = _switch_garment(
                overlay, catalog, "prev", "Key: PREV <-"
            )
            if banner:
                gesture_banner = banner
                gesture_banner_until = time.time() + GESTURE_BANNER_SEC
        elif key == ord("c") or key == ord("C"):
            variant = overlay.cycle_color()
            print(f"[INFO] Color: {variant['label']}")
        elif key == ord("o") or key == ord("O"):
            if overlay.outfit_mode != "full":
                lower_path = catalog.get_current_lower_path()
                if not lower_path:
                    print("[WARN] No pants/jeans in catalog. Staying on UPPER ONLY.")
                elif overlay.load_lower_garment(lower_path):
                    overlay.outfit_mode = "full"
                    print("[INFO] Outfit mode: FULL (shirt + pants)")
                else:
                    overlay.outfit_mode = "upper"
                    print("[WARN] Pants image failed to load. Staying on UPPER ONLY.")
            else:
                overlay.outfit_mode = "upper"
                overlay.clear_lower_garment()
                print("[INFO] Outfit mode: UPPER ONLY")
        elif key == ord("s") or key == ord("S"):
            if countdown_started_at is None and state == STATE_LIVE:
                countdown_started_at = time.time()
                snapshot_taken = False
                print("[INFO] Snapshot countdown started...")

    recorder.stop()
    cam.release()
    tracker.close()
    gestures.close()
    cv2.destroyAllWindows()
    print("[INFO] System exited cleanly.")


if __name__ == "__main__":
    main()
