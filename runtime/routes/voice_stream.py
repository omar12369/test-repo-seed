# runtime/routes/voice_stream.py
from __future__ import annotations

from typing import Iterator, Literal
from urllib.parse import quote_plus

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse

from runtime.services import get_audio_path, speak_to_wav
from runtime.services.text_preprocessor import clean_text, segment_for_tts

stream_router = APIRouter(tags=["voice-stream"])


# ---------------------------- helpers ---------------------------------
def _file_chunker(path_str: str, chunk_size: int = 64 * 1024) -> Iterator[bytes]:
    with open(path_str, "rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            yield data


def _prepare_text(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        raise HTTPException(status_code=400, detail="Missing 'text'.")
    cln = clean_text(s)
    if not cln:
        raise HTTPException(status_code=400, detail="Nothing to speak after cleaning.")
    segs = segment_for_tts(cln)
    return " ... ".join(segs) if segs else cln


# ----------------------------- routes ---------------------------------
@stream_router.get("/version")
def version():
    return {"voice_stream_routes_version": "phase3.2-json-or-stream"}


@stream_router.post("/say")
def say_post(
    payload: dict = Body(
        ...,
        description='{"text":"...","rate":1.0,"volume":1.0} (rate/volume optional)',
    ),
    mode: Literal["json", "stream"] = Query(
        "json",
        description="json (Swagger-friendly) | stream (audio/wav bytes)",
    ),
):
    text = _prepare_text(str(payload.get("text", "")))
    rate = payload.get("rate")
    volume = payload.get("volume")

    name = speak_to_wav(text, rate=rate, volume=volume)
    path = get_audio_path(name)
    if not path.is_file():
        raise HTTPException(status_code=500, detail="Audio generation failed.")

    if mode == "stream":
        return StreamingResponse(
            _file_chunker(str(path)),
            media_type="audio/wav",
            headers={"Content-Disposition": f'inline; filename="{name}"'},
        )

    # Default: JSON so Swagger doesn't hang on binary streams
    return {
        "ok": True,
        "file": name,
        "url": f"/audio/{name}",
        "abs_url": f"http://127.0.0.1:8000/audio/{name}",
        "mime": "audio/wav",
    }


@stream_router.get("/say")
def say_get(
    text: str = Query(..., description="Text to speak"),
    rate: float | None = Query(None, description="Rate multiplier (e.g., 0.9..1.2)"),
    volume: float | None = Query(None, description="Volume 0.0..1.0"),
):
    final_text = _prepare_text(text)

    name = speak_to_wav(final_text, rate=rate, volume=volume)
    path = get_audio_path(name)
    if not path.is_file():
        raise HTTPException(status_code=500, detail="Audio generation failed.")

    return RedirectResponse(url=f"/audio/{name}", status_code=302)


@stream_router.get("/demo", response_class=HTMLResponse)
def demo(text: str = "Streaming demo — Omar, we are live."):
    q = quote_plus(text)
    src = f"/v1/voice/stream/say?text={q}"

    return f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <title>Voice Stream Demo</title>
        <style>
          body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto; padding: 24px; }}
          button {{ font-size: 16px; padding: 10px 16px; margin-right: 8px; }}
          .row {{ margin: 12px 0; }}
          #err {{ color: #b00; white-space: pre-wrap; }}
          audio {{ width: 100%; }}
        </style>
      </head>
      <body>
        <h3>Streaming Demo</h3>
        <p>Text: <code>{text}</code></p>

        <div class="row">
          <audio id="player" controls preload="none"></audio>
        </div>

        <div class="row">
          <button onclick="playNow()">▶ Play</button>
          <button onclick="regenerate()">↻ Regenerate</button>
        </div>

        <div id="err"></div>

        <script>
          const src = "{src}";
          const player = document.getElementById("player");
          const errBox = document.getElementById("err");

          function logErr(e) {{
            errBox.textContent = (e && e.message) ? e.message : String(e || "");
          }}

          function loadSource() {{
            player.src = src;
          }}

          async function tryAutoplay() {{
            try {{
              await player.play();
            }} catch (e) {{
              logErr("Autoplay blocked by the browser. Click Play to start.");
            }}
          }}

          function playNow() {{
            loadSource();
            player.play().catch(logErr);
          }}

          function regenerate() {{
            location.reload();
          }}

          loadSource();
          tryAutoplay();
        </script>
      </body>
    </html>
    """
