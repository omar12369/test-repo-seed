from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv
from fastapi import APIRouter, WebSocket
from fastapi.responses import HTMLResponse
from starlette.websockets import WebSocketDisconnect

from runtime.services import get_audio_path, synthesize_to_wav

# Safe profile lookup only
try:
    from runtime.services.voice_profiles import get_profile  # type: ignore
except Exception:
    get_profile = None  # type: ignore

# Phase 5C: feedback logging (safe import)
try:
    from runtime.routes.feedback import log_feedback_internal  # type: ignore
except Exception:
    log_feedback_internal = None  # type: ignore

# Load .env
load_dotenv(override=True)

realtime_router = APIRouter(prefix="/v1/voice/realtime", tags=["voice-realtime"])

VOICE_ENGINE = os.getenv("VOICE_ENGINE", "pyttsx3").strip().lower()
DEFAULT_PROFILE = os.getenv("VOICE_DEFAULT_PROFILE", "auren_balanced").strip().lower()


# -----------------------------
# Helpers
# -----------------------------
async def _send(ws: WebSocket, msg: Dict[str, Any]) -> None:
    await ws.send_text(json.dumps(msg, ensure_ascii=False))


def _shape_text_basic(text: str, pause_ms: int, warmth: float) -> str:
    t = (text or "").strip()
    if not t:
        return ""

    if pause_ms and pause_ms > 0:
        t = t.replace(". ", ".  ").replace("! ", "!  ").replace("? ", "?  ")

    _ = warmth
    return t


def _apply_profile_fallback(
    text: str,
    profile: str,
    mood: Optional[str],
    rate_override: Optional[float],
    volume_override: Optional[float],
) -> Tuple[str, float, float]:
    rate_mult = 1.0
    vol = 1.0
    pause_ms = 90
    warmth = 0.35

    if get_profile is not None:
        try:
            p = get_profile(profile)  # type: ignore
            rate_mult = float(getattr(p, "rate", rate_mult))
            vol = float(getattr(p, "volume", vol))
            pause_ms = int(getattr(p, "pause_ms", pause_ms))
            warmth = float(getattr(p, "warmth", warmth))
        except Exception:
            pass

    if rate_override is not None:
        try:
            rate_mult = float(rate_override)
        except Exception:
            pass

    if volume_override is not None:
        try:
            vol = float(volume_override)
        except Exception:
            pass

    shaped = _shape_text_basic(text, pause_ms=pause_ms, warmth=warmth)

    _ = mood
    return shaped, rate_mult, vol


# -----------------------------
# Demo Page
# -----------------------------
@realtime_router.get("/demo", response_class=HTMLResponse)
def demo_page() -> str:
    return """
<!doctype html>
<html>
  <head><meta charset="utf-8" /><title>Auren Realtime</title></head>
  <body>
    <h2>Auren Realtime</h2>
    <input id="text" style="width:80%" placeholder="Type..." />
    <button onclick="speak()">Speak</button>
    <p id="status">Ready</p>
    <audio id="player" controls autoplay style="width:100%"></audio>

    <script>
      const wsUrl = (location.protocol === "https:" ? "wss://" : "ws://")
        + location.host + "/v1/voice/realtime/ws";

      function speak() {
        const ws = new WebSocket(wsUrl);
        const text = document.getElementById("text").value;

        ws.onopen = () => {
          ws.send(JSON.stringify({ type: "speak", text }));
        };

        ws.onmessage = (e) => {
          const msg = JSON.parse(e.data);

          if (msg.type === "speaking") {
            const player = document.getElementById("player");
            player.src = msg.abs_url || msg.url;
            player.play();
            ws.close();
          }
        };
      }
    </script>
  </body>
</html>
"""


# -----------------------------
# WebSocket
# -----------------------------
@realtime_router.websocket("/ws")
async def ws_voice(ws: WebSocket) -> None:
    import asyncio

    await ws.accept()
    await _send(ws, {"type": "info", "text": "Realtime ready"})

    try:
        while True:
            raw = await ws.receive_text()

            try:
                obj = json.loads(raw)
            except Exception:
                await _send(ws, {"type": "error", "detail": "Invalid JSON"})
                continue

            if obj.get("type") != "speak":
                await _send(ws, {"type": "error", "detail": "Invalid type"})
                continue

            text = (obj.get("text") or "").strip()
            profile = (obj.get("profile") or DEFAULT_PROFILE).strip().lower()

            if not text:
                await _send(ws, {"type": "error", "detail": "Empty text"})
                continue

            shaped, rmult, vol = _apply_profile_fallback(
                text=text,
                profile=profile,
                mood=None,
                rate_override=None,
                volume_override=None,
            )

            await _send(ws, {"type": "generating"})

            try:
                name, synth_meta = await asyncio.to_thread(
                    synthesize_to_wav,
                    shaped,
                    rate=rmult,
                    volume=vol,
                )

                path = get_audio_path(name)
                if not path.is_file():
                    raise RuntimeError("File missing")

            except Exception as e:
                await _send(ws, {"type": "error", "detail": str(e)})
                continue

            rel = f"/v1/voice/file/{name}"
            base = str(ws.url).replace("ws://", "http://").replace("wss://", "https://")
            base = base.split("/v1/voice/realtime/ws")[0].rstrip("/")
            abs_url = base + rel

            await _send(
                ws,
                {
                    "type": "speaking",
                    "engine": VOICE_ENGINE,
                    "name": name,
                    "url": rel,
                    "abs_url": abs_url,
                    "profile": profile,
                    "rate": rmult,
                    "volume": vol,
                    "ts": int(time.time()),
                },
            )

            # -----------------------------
            # Phase 5C: Feedback Logging
            # -----------------------------
            if log_feedback_internal:
                try:
                    log_feedback_internal(
                        user_input=text,
                        system_output=f"[audio:{name}]",
                        channel="voice",
                        input_mode="text",
                        output_mode="voice",
                        engine=VOICE_ENGINE,
                        profile=profile,
                        success=True,
                        latency_ms=None,
                        playback_ok=True,
                        user_feedback=None,
                        feedback_tags=["auto_logged", "realtime"],
                        notes=None,
                    )
                except Exception:
                    pass

    except WebSocketDisconnect:
        return
