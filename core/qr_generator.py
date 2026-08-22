import socket
from pathlib import Path

import cv2
import numpy as np
import qrcode


def get_lan_ip() -> str:
    """Best-effort LAN address so a phone on the same Wi-Fi can open the URL."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except OSError:
        return "127.0.0.1"


def capture_download_url(filename: str, port: int = 8000) -> str:
    name = Path(filename).name
    return f"http://{get_lan_ip()}:{port}/captures/{name}"


def generate_qr_image(url: str, pixel_size: int = 196) -> np.ndarray:
    """Return a BGR OpenCV image of a QR code for `url`."""
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=8,
        border=2,
    )
    qr.add_data(url)
    qr.make(fit=True)
    pil_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    rgb = np.array(pil_img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.resize(bgr, (pixel_size, pixel_size), interpolation=cv2.INTER_NEAREST)


def overlay_qr_code(frame, qr_image, margin: int = 28, label: str = "Scan to Download"):
    """Draw the QR code in the bottom-right corner with a caption."""
    if frame is None or qr_image is None:
        return frame

    frame_h, frame_w = frame.shape[:2]
    qr_h, qr_w = qr_image.shape[:2]
    pad = 12
    x1 = frame_w - qr_w - margin - pad
    y1 = frame_h - qr_h - margin - pad - 36
    x2 = x1 + qr_w + pad * 2
    y2 = y1 + qr_h + pad * 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1 - 8, y1 - 44), (x2 + 8, y2 + 8), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    frame[y1 + pad: y1 + pad + qr_h, x1 + pad: x1 + pad + qr_w] = qr_image

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        frame,
        label,
        (x1, y1 - 12),
        font,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame
