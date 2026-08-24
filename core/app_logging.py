"""File + console logging for the Smart Mirror and FastAPI processes."""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "app.log"

_log_file = None
_configured = False


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
                stream.flush()
            except Exception:
                pass

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass

    def isatty(self):
        return False


def setup_app_logging(name: str = "vto"):
    """Send logging and print output to logs/app.log as well as the console."""
    global _configured, _log_file
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if _log_file is None:
        _log_file = open(LOG_PATH, "a", encoding="utf-8")

    if not _configured:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.StreamHandler(_log_file),
                logging.StreamHandler(sys.__stdout__),
            ],
            force=True,
        )
        sys.stdout = _Tee(sys.__stdout__, _log_file)
        sys.stderr = _Tee(sys.__stderr__, _log_file)
        _configured = True

    logging.getLogger(name).info("Logging to %s", LOG_PATH)
    return logging.getLogger(name)


def close_app_logging():
    global _log_file, _configured
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    if _log_file is not None:
        try:
            _log_file.flush()
            _log_file.close()
        except Exception:
            pass
        _log_file = None
    _configured = False
