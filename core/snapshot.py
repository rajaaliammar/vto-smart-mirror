import bz2
import gc
import os
import shutil
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

CAPTURES_DIR = "captures"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR_REL = os.path.join("captures", "videos")
VIDEO_CAPTURES_DIR = PROJECT_ROOT / "captures" / "videos"
OPENH264_DLL = "openh264-2.5.0-win64.dll"
OPENH264_URLS = (
    "http://ciscobinary.openh264.org/openh264-2.5.0-win64.dll.bz2",
    "https://ciscobinary.openh264.org/openh264-2.5.0-win64.dll.bz2",
)


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


def ensure_openh264() -> None:
    """Place Cisco OpenH264 on PATH so VideoWriter can encode avc1/h264 MP4."""
    if os.name != "nt":
        return
    dest = PROJECT_ROOT / OPENH264_DLL
    if not dest.exists() or dest.stat().st_size < 100_000:
        print("[INFO] Downloading OpenH264 encoder (one-time, for VS Code MP4 playback)...")
        last_error = None
        for url in OPENH264_URLS:
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    compressed = response.read()
                dest.write_bytes(bz2.decompress(compressed))
                print(f"[INFO] OpenH264 saved: {dest}")
                last_error = None
                break
            except Exception as exc:
                last_error = exc
        if last_error is not None and (not dest.exists() or dest.stat().st_size < 100_000):
            print(f"[WARN] OpenH264 download failed ({last_error}). H.264 MP4 may be empty.")
            return
    os.environ["PATH"] = str(PROJECT_ROOT) + os.pathsep + os.environ.get("PATH", "")
    copies = [Path.cwd() / OPENH264_DLL]
    try:
        copies.append(Path(cv2.__file__).resolve().parent / OPENH264_DLL)
    except Exception:
        pass
    for target in copies:
        if dest.exists() and target.resolve() != dest.resolve() and not target.exists():
            try:
                shutil.copy2(dest, target)
            except OSError:
                pass


class MirrorVideoRecorder:
    """Toggleable H.264 MP4 recorder for VS Code / browser playback."""

    def __init__(self, output_dir: Path = VIDEO_CAPTURES_DIR, fps: float = 20.0):
        self.output_dir = Path(output_dir)
        self.default_fps = float(fps) if fps and fps > 1 else 20.0
        self.is_recording = False
        self.writer = None
        self.path = None
        self.rel_path = None
        self.started_at = None
        self._size = None
        self._fps = self.default_fps
        self._frames_written = 0
        ensure_openh264()

    @property
    def recording(self) -> bool:
        return self.is_recording

    def elapsed(self) -> float:
        if not self.is_recording or self.started_at is None:
            return 0.0
        return max(0.0, time.time() - self.started_at)

    def toggle(self, frame, fps: float = None):
        if self.is_recording:
            return self.stop()
        self.start(frame, fps=fps)
        return None

    def start(self, frame, fps: float = None):
        if self.is_recording:
            return self.rel_path
        if frame is None:
            print("[ERROR] Cannot start recording without a frame.")
            return None

        ensure_openh264()
        abs_dir = os.path.abspath(str(self.output_dir))
        os.makedirs(abs_dir, exist_ok=True)

        width, height = int(frame.shape[1]), int(frame.shape[0])
        width -= width % 2
        height -= height % 2
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vto_recording_{stamp}.mp4"
        abs_path = os.path.abspath(os.path.join(abs_dir, filename))
        if os.path.exists(abs_path):
            filename = f"vto_recording_{stamp}_{int(time.time() * 1000) % 1000:03d}.mp4"
            abs_path = os.path.abspath(os.path.join(abs_dir, filename))

        write_fps = float(fps) if fps and fps > 1 else self.default_fps
        writer, used_path, codec = self._open_writer(abs_path, write_fps, (width, height))
        if writer is None:
            print(f"[ERROR] Failed to open H.264 VideoWriter at {abs_path}")
            return None

        self.writer = writer
        self.is_recording = True
        self.path = used_path
        self.rel_path = os.path.join(
            VIDEO_DIR_REL, os.path.basename(used_path)
        ).replace("\\", "/")
        self.started_at = time.time()
        self._size = (width, height)
        self._fps = write_fps
        self._frames_written = 0
        self.write(frame)
        print(
            f"[INFO] Recording started ({codec} {width}x{height} @ {write_fps:.1f} FPS). "
            "Press 'R' again to stop."
        )
        return self.rel_path

    @staticmethod
    def _open_writer(abs_mp4: str, fps: float, size):
        """H.264 MP4 for VS Code: h264, then avc1, then VP80."""
        width, height = int(size[0]), int(size[1])
        frame_size = (width, height)
        for codec in ("h264", "avc1", "VP80"):
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(abs_mp4, fourcc, float(fps), frame_size, True)
            if writer is not None and writer.isOpened():
                return writer, abs_mp4, codec
            if writer is not None:
                writer.release()
        return None, None, None

    @staticmethod
    def _prepare_frame(frame, size):
        """Contiguous BGR uint8 copy. OpenCV VideoWriter expects BGR, not RGB."""
        if frame is None:
            return None
        prepared = np.ascontiguousarray(frame.copy())
        if prepared.ndim == 2:
            prepared = cv2.cvtColor(prepared, cv2.COLOR_GRAY2BGR)
        elif prepared.shape[2] == 4:
            prepared = cv2.cvtColor(prepared, cv2.COLOR_BGRA2BGR)
        elif prepared.shape[2] != 3:
            return None
        height, width = prepared.shape[:2]
        if (width, height) != size:
            prepared = cv2.resize(prepared, size, interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(prepared, dtype=np.uint8)

    def write(self, frame) -> None:
        if frame is None or not self.is_recording or self.writer is None:
            return
        prepared = self._prepare_frame(frame, self._size)
        if prepared is None:
            return
        self.writer.write(prepared.copy())
        self._frames_written += 1

    def stop(self):
        if not self.is_recording:
            self._close_writer()
            return None

        frames = self._frames_written
        rel = self.rel_path
        abs_saved = self.path
        self.is_recording = False
        self._close_writer()
        self.path = None
        self.rel_path = None
        self.started_at = None
        self._size = None
        self._frames_written = 0
        if rel:
            print(f"[INFO] Video saved to {rel} ({frames} frames)")
        elif abs_saved:
            print(f"[INFO] Video saved to {abs_saved} ({frames} frames)")
        return rel or abs_saved

    def _close_writer(self) -> None:
        """Release the encoder and drop the handle so VS Code can open the file."""
        out = self.writer
        self.writer = None
        if out is None:
            return
        out.release()
        del out
        gc.collect()
        time.sleep(0.15)

