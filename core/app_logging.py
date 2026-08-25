"""File + console logging for the Smart Mirror and FastAPI processes."""

import logging
import os
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


class _SafeFileHandler(logging.FileHandler):
    """Append handler that survives a locked log file (FastAPI cmd redirect)."""

    _warned = False

    def emit(self, record):
        try:
            super().emit(record)
        except (PermissionError, OSError):
            self.handleError(record)

    def handleError(self, record):
        if not _SafeFileHandler._warned:
            _SafeFileHandler._warned = True
            try:
                sys.__stderr__.write(
                    "[WARN] Log file locked; writing to console only.\n"
                )
            except Exception:
                pass
        try:
            self.close()
        except Exception:
            pass
        self.stream = None


def _open_shared_append(path: Path):
    """Open a log file with Windows share-write flags when possible."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        try:
            flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
            if hasattr(os, "O_NOINHERIT"):
                flags |= os.O_NOINHERIT
            fd = os.open(str(path), flags)
            return os.fdopen(fd, "a", encoding="utf-8", buffering=1)
        except OSError:
            pass
    return open(path, "a", encoding="utf-8", buffering=1)


def setup_app_logging(name: str = "vto"):
    """Send logging and print output to logs/app.log as well as the console."""
    global _configured, _log_file
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    handlers = [logging.StreamHandler(sys.__stdout__)]
    file_ok = False

    if _log_file is None:
        try:
            _log_file = _open_shared_append(LOG_PATH)
            file_ok = True
        except (PermissionError, OSError) as exc:
            _log_file = None
            print(
                f"[WARN] Could not open {LOG_PATH} ({exc}). Logging to stdout.",
                file=sys.__stderr__,
            )

    if not _configured:
        if file_ok and _log_file is not None:
            handlers.append(logging.StreamHandler(_log_file))
        else:
            try:
                file_handler = _SafeFileHandler(
                    str(LOG_PATH),
                    mode="a",
                    encoding="utf-8",
                    delay=True,
                )
                handlers.append(file_handler)
            except (PermissionError, OSError) as exc:
                print(
                    f"[WARN] FileHandler unavailable for {LOG_PATH} ({exc}).",
                    file=sys.__stderr__,
                )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=handlers,
            force=True,
        )
        if _log_file is not None:
            sys.stdout = _Tee(sys.__stdout__, _log_file)
            sys.stderr = _Tee(sys.__stderr__, _log_file)
        _configured = True

    logger = logging.getLogger(name)
    logger.info("Logging to %s", LOG_PATH if _log_file else "stdout")
    return logger


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
