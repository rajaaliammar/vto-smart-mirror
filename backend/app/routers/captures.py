from pathlib import Path

from fastapi import APIRouter, Request

from app.schemas.garment import CaptureItem, CaptureListResponse

router = APIRouter(prefix="/captures", tags=["captures"])

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent
CAPTURES_DIR = PROJECT_ROOT / "captures"
VIDEO_DIR = CAPTURES_DIR / "videos"
SNAPSHOT_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".webm", ".mov"}


def _public_url(request: Request, relative: str) -> str:
    return str(request.base_url).rstrip("/") + "/" + relative.lstrip("/")


@router.get("", response_model=CaptureListResponse)
def list_captures(request: Request):
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    snapshots = []
    for path in sorted(CAPTURES_DIR.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file() or path.suffix.lower() not in SNAPSHOT_EXTS:
            continue
        snapshots.append(
            CaptureItem(
                filename=path.name,
                kind="snapshot",
                url=_public_url(request, f"captures/{path.name}"),
            )
        )

    videos = []
    if VIDEO_DIR.is_dir():
        for path in sorted(VIDEO_DIR.iterdir(), key=lambda item: item.name.lower()):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
                continue
            videos.append(
                CaptureItem(
                    filename=path.name,
                    kind="video",
                    url=_public_url(request, f"captures/videos/{path.name}"),
                )
            )

    return CaptureListResponse(
        count=len(snapshots) + len(videos),
        snapshots=snapshots,
        videos=videos,
    )
