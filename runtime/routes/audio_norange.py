from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

audio_router = APIRouter(tags=["audio"])


def _audio_dir() -> str:
    # Keep consistent with your config/mount. If you already have a config helper, swap it in.
    return os.getenv("AUDIO_DIR", "data/audio")


@audio_router.get("/audio_norange/{name}")
def audio_norange(name: str):
    # Basic safety
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="Invalid filename")

    path = os.path.abspath(os.path.join(_audio_dir(), name))
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Audio file not found")

    # Force full-file behavior for browsers (avoid 206/range weirdness)
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
        "Accept-Ranges": "none",
    }

    return FileResponse(
        path=path,
        media_type="audio/wav",
        filename=name,
        headers=headers,
    )
