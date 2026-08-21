import os
import threading
import time
from datetime import datetime

import cv2

CAPTURES_DIR = "captures"


def save_snapshot(frame, captures_dir: str = CAPTURES_DIR) -> str:
    """Write a rendered frame to captures/VTO_TryOn_YYYYMMDD_HHMMSS.png."""
    if frame is None:
        raise ValueError("Cannot save an empty frame.")

    os.makedirs(captures_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"VTO_TryOn_{stamp}.png"
    path = os.path.join(captures_dir, filename)

    if os.path.exists(path):
        filename = f"VTO_TryOn_{stamp}_{int(time.time() * 1000) % 1000:03d}.png"
        path = os.path.join(captures_dir, filename)

    if not cv2.imwrite(path, frame):
        raise IOError(f"Failed to write snapshot to {path}")
    return os.path.abspath(path)


def save_snapshot_async(frame, on_saved=None, captures_dir: str = CAPTURES_DIR):
    """Save on a background thread so the webcam loop does not stall."""
    clone = frame.copy()

    def _worker():
        try:
            path = save_snapshot(clone, captures_dir=captures_dir)
            print(f"[INFO] Snapshot saved: {path}")
            if on_saved:
                on_saved(path)
        except Exception as exc:
            print(f"[ERROR] Snapshot failed: {exc}")

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread
