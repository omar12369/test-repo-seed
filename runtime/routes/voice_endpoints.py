# runtime/routes/voice_endpoints.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# --- locate your project paths ---
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_FILE = DATA_DIR / os.getenv("VOICE_HISTORY_FILE", "voice_history.json")
HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

DEFAULT_ENGINE = os.getenv("VOICE_DEFAULT_ENGINE", "pyttsx3")
DEFAULT_VOICE = os.getenv("VOICE_DEFAULT_VOICE", "")
ALLOW_OVERRIDE = os.getenv("VOICE_ALLOW_OVERRIDE", "false").lower() == "true"

router = APIRouter(prefix="/v1/voice", tags=["voice"])


# ---- models ----
class SpeakIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    rate: Optional[str] = None
    pitch: Optional[str] = None
    volume: Optional[float] = None
    voice: Optional[str] = None
    tone: Optional[str] = None


class SpeakOut(BaseModel):
    ok: bool
    engine: str
    filename: str
    url: str


# ---- helpers ----
def _read_history() -> List[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _write_history(items: List[dict]) -> None:
    HISTORY_FILE.write_text(json.dumps(items, indent=2), encoding="utf-8")


def _append_history(entry: dict) -> None:
    h = _read_history()
    h.append(entry)
    _write_history(h)


def _first_callable(mod, names: list[str]) -> Optional[Callable]:
    for n in names:
        f = getattr(mod, n, None)
        if callable(f):
            return f
    return None


# Import your existing voice module (DON'T delete it)
# Try a few common locations/names; adjust if yours differs.
_voice_mod = None
for cand in (
    "runtime.services.voice_service",
    "runtime.voice",
    "voice",
):
    try:
        _voice_mod = __import__(cand, fromlist=["*"])
        break
    except Exception:
        pass
if _voice_mod is None:
    raise RuntimeError("Could not import your voice module; update the import list.")

_speak = _first_callable(
    _voice_mod,
    ["speak", "say", "tts_speak", "synthesize", "synthesize_speech"],
)
_list_voices = _first_callable(
    _voice_mod, ["list_voices", "voices", "available_voices", "get_voices"]
)


# ---- endpoints ----
@router.get("/version")
def voice_version():
    return {
        "engine": DEFAULT_ENGINE,
        "voice": DEFAULT_VOICE,
        "ffmpeg_enabled": os.getenv("VOICE_FFMPEG_ENABLE", "0"),
        "allow_override": ALLOW_OVERRIDE,
    }


@router.get("/status")
def voice_status():
    return {"ok": True, "engine": DEFAULT_ENGINE, "audio_dir": str(AUDIO_DIR)}


@router.get("/voices")
def voices():
    if _list_voices:
        try:
            v = _list_voices()
            return {"voices": v}
        except Exception:
            pass
    # fallback
    return {"voices": [DEFAULT_VOICE or "en-US-JennyNeural", "en-US-GuyNeural"]}


@router.post("/debug/clean")
def debug_clean():
    deleted = 0
    for p in AUDIO_DIR.glob("*"):
        try:
            p.unlink()
            deleted += 1
        except Exception:
            pass
    return {"ok": True, "deleted": deleted}


@router.get("/history")
def history_get():
    return {"items": _read_history()}


@router.post("/history/clear")
def history_clear():
    _write_history([])
    return {"ok": True}


@router.post("/speak", response_model=SpeakOut)
def speak(inp: SpeakIn):
    if not _speak:
        raise HTTPException(status_code=500, detail="No speak() function found.")
    try:
        out_path = _speak(
            text=inp.text,
            rate=inp.rate,
            pitch=inp.pitch,
            volume=inp.volume,
            voice=inp.voice,
            tone=inp.tone,
            allow_override=ALLOW_OVERRIDE,
            audio_dir=str(AUDIO_DIR),
        )
        filename = Path(out_path).name
    except TypeError:
        # If your speak() has a different signature, try minimal call
        out_path = _speak(inp.text)  # type: ignore[arg-type]
        filename = Path(out_path).name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"speak failed: {e}")

    _append_history({"file": filename, "text": inp.text[:2000], "engine": DEFAULT_ENGINE})
    return SpeakOut(
        ok=True, engine=DEFAULT_ENGINE, filename=filename, url=f"/v1/voice/file/{filename}"
    )


@router.get("/file/{name}")
def voice_file(name: str):
    path = (AUDIO_DIR / name).resolve()
    if not str(path).startswith(str(AUDIO_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)
