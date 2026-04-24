# runtime/routes/voice_voices.py
from __future__ import annotations

import os

import pyttsx3
from fastapi import APIRouter
from fastapi.responses import JSONResponse

voices_router = APIRouter(tags=["voice-voices"])


def _engine() -> pyttsx3.Engine:
    # create/dispose each call to avoid cross-thread surprises on reload
    eng = pyttsx3.init()  # SAPI5 on Windows
    return eng


@voices_router.get("/voices", summary="List installed TTS voices")
def list_voices() -> JSONResponse:
    eng = _engine()
    items = []
    try:
        for v in eng.getProperty("voices"):
            items.append(
                {
                    "id": getattr(v, "id", ""),
                    "name": getattr(v, "name", ""),
                    "languages": (
                        [str(x) for x in getattr(v, "languages", [])]
                        if hasattr(v, "languages")
                        else []
                    ),
                    "gender": getattr(
                        v, "gender", ""
                    ),  # often empty on SAPI5, but return if present
                    "age": getattr(v, "age", ""),
                }
            )
    finally:
        try:
            eng.stop()
        except Exception:
            pass
        del eng
    return JSONResponse({"count": len(items), "voices": items})


@voices_router.get("/voice/default", summary="Show current default engine/voice")
def show_default() -> JSONResponse:
    engine = os.getenv("VOICE_ENGINE", "pyttsx3")
    default_voice = os.getenv(
        "VOICE_DEFAULT_VOICE",
        os.getenv(
            "VOICE_EDGE_VOICE",
            "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_ZIRA_11.0",
        ),
    )
    return JSONResponse({"engine": engine, "default_voice": default_voice})
