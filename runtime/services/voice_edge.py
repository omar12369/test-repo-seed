from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Tuple

import edge_tts

# -----------------------------------------------------------------------------
# ENV / PATHS
# -----------------------------------------------------------------------------
AUDIO_DIR = Path(os.getenv("AUDIO_DIR", "data/audio")).resolve()
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

VOICE_EDGE_VOICE = os.getenv("VOICE_EDGE_VOICE", "en-US-JennyNeural").strip()

# Slight natural adjustments
EDGE_RATE = os.getenv("VOICE_EDGE_RATE", "-5%")  # slower = more natural
EDGE_VOLUME = os.getenv("VOICE_EDGE_VOLUME", "+0%")


# -----------------------------------------------------------------------------
# CORE
# -----------------------------------------------------------------------------
async def _synthesize_edge_async(text: str, voice: str) -> bytes:
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=EDGE_RATE,
        volume=EDGE_VOLUME,
    )

    audio_bytes = b""

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes += chunk["data"]

    return audio_bytes


def synthesize_edge_to_wav(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Fully isolated Edge-TTS synthesis.

    Returns:
        (filename, meta)
    """
    cleaned = (text or "").strip()
    if not cleaned:
        raise RuntimeError("Nothing to synthesize.")

    try:
        audio_bytes = asyncio.run(_synthesize_edge_async(cleaned, VOICE_EDGE_VOICE))
    except RuntimeError:
        # Fallback for already running loop (safety)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        audio_bytes = loop.run_until_complete(_synthesize_edge_async(cleaned, VOICE_EDGE_VOICE))
        loop.close()

    name = f"{uuid.uuid4().hex}.mp3"
    out_path = AUDIO_DIR / name

    out_path.write_bytes(audio_bytes)

    meta: Dict[str, Any] = {
        "engine": "edge-tts",
        "voice": VOICE_EDGE_VOICE,
        "rate": EDGE_RATE,
        "volume": EDGE_VOLUME,
        "bytes": len(audio_bytes),
    }

    return name, meta
