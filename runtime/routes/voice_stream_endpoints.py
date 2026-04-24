"""
Voice Stream Endpoints
- POST /v1/voice/stream/say : stream TTS audio as it is produced

This router defers to an implementation function named `stream_say`
which should live in one of these locations (first match wins):

  1) runtime/services/voice_stream.py      (recommended)
  2) runtime/routes/voice_stream.py
  3) runtime/voice_stream.py
  4) voice_stream.py                        (project root)

The implementation may return:
  • fastapi.responses.StreamingResponse
  • an iterator/generator yielding audio bytes
  • raw bytes (mp3) — this router will wrap it
"""

from __future__ import annotations

import importlib
import io
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

router = APIRouter(prefix="/v1/voice/stream", tags=["voice-stream"])

# ---- Input schema -----------------------------------------------------------


class StreamSayIn(BaseModel):
    text: str
    voice: str | None = None
    rate: str | None = None  # e.g. "slow", "medium", "fast" or "0%/+10%/-10%"
    tone: str | None = None  # e.g. "neutral", "soft", "warm", "bright"


# ---- Implementation loader --------------------------------------------------

_IMPL_CANDIDATES = [
    "runtime.services.voice_stream",
    "runtime.routes.voice_stream",
    "runtime.voice_stream",
    "voice_stream",
]


def _load_stream_say() -> Any:
    """
    Find and return a callable named `stream_say` from the first available
    implementation module. Raises ImportError with a helpful message if none
    can be located.
    """
    errors: list[str] = []
    for mod_name in _IMPL_CANDIDATES:
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as e:
            errors.append(f"{mod_name!r}: {e.__class__.__name__}: {e}")
            continue

        func = getattr(mod, "stream_say", None)
        if callable(func):
            return func
        errors.append(f"{mod_name!r}: missing callable `stream_say`")

    raise ImportError(
        "Could not locate a `stream_say` implementation. " "Tried:\n- " + "\n- ".join(errors)
    )


# ---- Routes -----------------------------------------------------------------


@router.get("/health")
def health() -> JSONResponse:
    """Lightweight health check for this router."""
    return JSONResponse({"ok": True, "router": "voice-stream"})


@router.post("/say")
def stream_say_endpoint(body: StreamSayIn):
    """
    Stream synthesized speech (MP3 by default).

    Delegates to your `stream_say(text, voice=None, rate=None, tone=None)`.
    That function may return:
      - StreamingResponse (returned as-is)
      - iterator/generator yielding bytes (wrapped as StreamingResponse)
      - raw bytes (wrapped as StreamingResponse)
    """
    try:
        impl = _load_stream_say()
    except ImportError as e:
        # Clear and actionable error for misconfiguration
        raise HTTPException(status_code=500, detail=str(e))

    try:
        result = impl(body.text, voice=body.voice, rate=body.rate, tone=body.tone)
    except Exception as e:  # surface implementation errors nicely
        raise HTTPException(status_code=500, detail=f"stream_say failed: {e}")

    # Case 1: implementation gave us a proper StreamingResponse
    if isinstance(result, StreamingResponse):
        return result

    # Case 2: implementation yielded/iterated bytes
    if hasattr(result, "__iter__") and not isinstance(result, (bytes, bytearray)):
        return StreamingResponse(result, media_type="audio/mpeg")

    # Case 3: implementation returned raw bytes
    if isinstance(result, (bytes, bytearray)):
        return StreamingResponse(io.BytesIO(result), media_type="audio/mpeg")

    # Unexpected return type
    raise HTTPException(
        status_code=500,
        detail=(
            "stream_say returned an unsupported type. "
            "Return a StreamingResponse, an iterator of bytes, or raw bytes."
        ),
    )
