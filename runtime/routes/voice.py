from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from runtime.services import get_audio_path, synthesize_to_wav
from runtime.services.feedback import log_feedback
from runtime.services.voice_context import build_delivery_style
from runtime.services.voice_edge import synthesize_edge_to_wav
from runtime.services.voice_history import append_history, get_history
from runtime.services.voice_policy import build_voice_policy
from runtime.services.voice_profiles import get_profile, list_profiles

# -----------------------------------------------------------------------------
# Phase lock / config
# -----------------------------------------------------------------------------
ENGINE_LOCK = "pyttsx3"
EDGE_PREVIEW_ENGINE = "edge-tts"

AUDIO_DIR = Path(os.getenv("AUDIO_DIR", "data/audio")).resolve()
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PROFILE = (os.getenv("VOICE_DEFAULT_PROFILE", "auren_balanced") or "auren_balanced").strip()
SERVED_URL_HINT = (
    os.getenv("VOICE_SERVED_URL_HINT", "/audio/{filename}") or "/audio/{filename}"
).strip()

EDGE_PREVIEW_ENABLED = os.getenv("VOICE_EDGE_PREVIEW_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

VOICE_AUTO_ROUTE = os.getenv("VOICE_AUTO_ROUTE", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
VOICE_POLICY_LIMIT = int(os.getenv("VOICE_POLICY_LIMIT", "50"))

voice_router = APIRouter(prefix="/v1/voice", tags=["voice"])


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class SpeakIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    profile: Optional[str] = None


class SpeakOut(BaseModel):
    ok: bool
    engine: str
    name: str
    url: str
    abs_url: str
    meta: Dict[str, Any]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _safe_audio_path(name: str) -> Path:
    candidate = get_audio_path(name).resolve()
    if candidate.parent != AUDIO_DIR:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    return candidate


def _base_url(request: Request) -> str:
    proto = request.headers.get("x-forwarded-proto")
    host = request.headers.get("x-forwarded-host")
    if proto and host:
        return f"{proto}://{host}".rstrip("/")
    return str(request.base_url).rstrip("/")


def _join(base: str, path: str) -> str:
    if not path.startswith("/"):
        path = "/" + path
    return base + path


def _engine_env_requested() -> str:
    return (os.getenv("VOICE_ENGINE") or os.getenv("VOICE_DEFAULT_ENGINE") or "").strip().lower()


def _build_urls(request: Request, filename: str) -> tuple[str, str]:
    url = SERVED_URL_HINT.format(filename=filename)
    base = _base_url(request)
    abs_url = _join(base, url)
    return url, abs_url


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _history_volume_value(meta: Dict[str, Any]) -> float:
    raw = meta.get("volume", 1.0)

    if isinstance(raw, (int, float)):
        return float(raw)

    if isinstance(raw, str):
        s = raw.strip()
        if s.endswith("%"):
            return 1.0
        try:
            return float(s)
        except Exception:
            return 1.0

    return 1.0


def _append_voice_history(
    *,
    engine: str,
    profile_name: str,
    text: str,
    filename: str,
    meta: Dict[str, Any],
) -> None:
    append_history(
        {
            "ts": int(time.time()),
            "engine": engine,
            "profile": profile_name,
            "file": filename,
            "chars": len((text or "").strip()),
            "bytes": _safe_int(meta.get("bytes", 0), 0),
            "voice_selected": meta.get("voice_selected", meta.get("voice", "")),
            "rate_wpm": _safe_int(meta.get("rate_wpm", 0), 0),
            "volume": _history_volume_value(meta),
            "leadin_ms": _safe_int(meta.get("leadin_ms", 0), 0),
            "chunks": _safe_int(meta.get("chunks", 0), 0),
        }
    )


def _try_auto_feedback_log(
    *,
    user_input: str,
    audio_name: str,
    engine: str,
    profile: str,
    latency_ms: int,
    tags: list[str],
) -> None:
    try:
        log_feedback(
            {
                "ts": int(time.time()),
                "user_input": user_input,
                "system_output": f"[audio:{audio_name}]",
                "channel": "voice",
                "input_mode": "text",
                "output_mode": "voice",
                "engine": engine,
                "profile": profile,
                "success": True,
                "latency_ms": latency_ms,
                "playback_ok": True,
                "feedback_tags": tags,
            }
        )
    except Exception:
        pass


def _synthesize_with_profile(
    text: str,
    profile_name: Optional[str],
    *,
    rate_scale: float = 1.0,
) -> tuple[str, Dict[str, Any], Any]:
    shaped = (text or "").strip()
    if not shaped:
        raise HTTPException(status_code=422, detail="text is required")

    prof = get_profile((profile_name or DEFAULT_PROFILE).strip())

    adjusted_rate = float(getattr(prof, "rate", 1.0)) * float(rate_scale)

    try:
        name, tts_meta = synthesize_to_wav(
            shaped,
            rate=adjusted_rate,
            volume=prof.volume,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {exc}") from exc

    tts_meta = dict(tts_meta)
    tts_meta["context_rate_scale"] = float(rate_scale)
    return name, tts_meta, prof


def _synthesize_with_edge(
    text: str,
    *,
    rate_scale: float = 1.0,
) -> tuple[str, Dict[str, Any]]:
    shaped = (text or "").strip()
    if not shaped:
        raise HTTPException(status_code=422, detail="text is required")

    if not EDGE_PREVIEW_ENABLED:
        raise HTTPException(status_code=503, detail="Edge voice preview is disabled.")

    try:
        name, tts_meta = synthesize_edge_to_wav(shaped)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Edge TTS synthesis failed: {exc}") from exc

    tts_meta = dict(tts_meta)
    tts_meta["context_rate_scale"] = float(rate_scale)
    return name, tts_meta


def _resolve_adaptive_plan(requested_profile: Optional[str]) -> Dict[str, str]:
    """
    Safe policy resolution for /speak.

    Rules:
    - If auto-route is off, stay on pyttsx3 baseline.
    - If caller provided an explicit profile, honor manual control and stay on pyttsx3.
    - If auto-route is on and no explicit profile was provided, consult policy.
    - If policy recommends edge-tts and edge preview is enabled, route to edge preview.
    - Otherwise, use pyttsx3 with the recommended/default profile.
    """
    explicit_profile = (requested_profile or "").strip()

    if explicit_profile:
        return {
            "engine": ENGINE_LOCK,
            "profile": explicit_profile,
            "source": "manual_profile",
        }

    if not VOICE_AUTO_ROUTE:
        return {
            "engine": ENGINE_LOCK,
            "profile": DEFAULT_PROFILE,
            "source": "baseline_default",
        }

    try:
        policy = build_voice_policy(limit=VOICE_POLICY_LIMIT)
    except Exception:
        return {
            "engine": ENGINE_LOCK,
            "profile": DEFAULT_PROFILE,
            "source": "policy_error_fallback",
        }

    recommended_engine = str(policy.get("recommended_engine", ENGINE_LOCK)).strip() or ENGINE_LOCK
    recommended_profile = (
        str(policy.get("recommended_profile", DEFAULT_PROFILE)).strip() or DEFAULT_PROFILE
    )

    if recommended_engine == EDGE_PREVIEW_ENGINE and EDGE_PREVIEW_ENABLED:
        return {
            "engine": EDGE_PREVIEW_ENGINE,
            "profile": "edge_preview",
            "source": "policy_edge",
        }

    return {
        "engine": ENGINE_LOCK,
        "profile": recommended_profile,
        "source": "policy_pyttsx3",
    }


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@voice_router.get("/presets")
def presets() -> JSONResponse:
    return JSONResponse({"ok": True, "default": DEFAULT_PROFILE, "profiles": list_profiles()})


@voice_router.get("/status")
def status(request: Request) -> JSONResponse:
    base = _base_url(request)
    req = _engine_env_requested()

    return JSONResponse(
        {
            "ok": True,
            "engine": ENGINE_LOCK,
            "engine_env_requested": req or None,
            "engine_note": "Voice generation is unified through runtime.services.voice using pyttsx3.",
            "edge_preview": {
                "enabled": EDGE_PREVIEW_ENABLED,
                "engine": EDGE_PREVIEW_ENGINE,
                "endpoint": _join(base, "/v1/voice/speak_edge"),
            },
            "adaptive_policy": {
                "auto_route_enabled": VOICE_AUTO_ROUTE,
                "policy_limit": VOICE_POLICY_LIMIT,
                "endpoint": _join(base, "/v1/voice/policy"),
            },
            "audio_dir": str(AUDIO_DIR),
            "default_profile": DEFAULT_PROFILE,
            "served_url_hint": SERVED_URL_HINT,
            "served_url_example": _join(
                base, SERVED_URL_HINT.format(filename="example.wav").lstrip("/")
            ),
            "env": os.getenv("APP_ENV", "dev"),
        }
    )


@voice_router.get("/history")
def history() -> JSONResponse:
    h = get_history()
    return JSONResponse({"ok": True, "size": h.get("size", 0), "items": h.get("items", [])})


@voice_router.post("/speak", response_model=SpeakOut)
def speak(payload: SpeakIn, request: Request) -> SpeakOut:
    start_time = time.time()

    plan = _resolve_adaptive_plan(payload.profile)
    delivery = build_delivery_style(payload.text)
    rate_scale = _safe_float(delivery.get("rate_scale"), 1.0)
    pause_scale = _safe_float(delivery.get("pause_scale"), 1.0)

    if plan["engine"] == EDGE_PREVIEW_ENGINE:
        name, tts_meta = _synthesize_with_edge(payload.text, rate_scale=rate_scale)
        active_engine = EDGE_PREVIEW_ENGINE
        active_profile = "edge_preview"

        _append_voice_history(
            engine=active_engine,
            profile_name=active_profile,
            text=payload.text,
            filename=name,
            meta=tts_meta,
        )

        _try_auto_feedback_log(
            user_input=payload.text,
            audio_name=name,
            engine=active_engine,
            profile=active_profile,
            latency_ms=int((time.time() - start_time) * 1000),
            tags=["auto_logged", "edge", "policy_selected"],
        )

        url, abs_url = _build_urls(request, name)

        return SpeakOut(
            ok=True,
            engine=active_engine,
            name=name,
            url=url,
            abs_url=abs_url,
            meta={
                "profile": active_profile,
                "voice": tts_meta.get("voice", ""),
                "rate": tts_meta.get("rate", ""),
                "volume": tts_meta.get("volume", ""),
                "bytes": _safe_int(tts_meta.get("bytes", 0), 0),
                "policy_source": plan["source"],
                "auto_routed": VOICE_AUTO_ROUTE and not bool((payload.profile or "").strip()),
                "context_rate_scale": rate_scale,
                "context_pause_scale": pause_scale,
            },
        )

    name, tts_meta, prof = _synthesize_with_profile(
        payload.text,
        plan["profile"],
        rate_scale=rate_scale,
    )
    url, abs_url = _build_urls(request, name)

    _append_voice_history(
        engine=ENGINE_LOCK,
        profile_name=prof.name,
        text=payload.text,
        filename=name,
        meta=tts_meta,
    )

    tags = ["auto_logged"]
    if VOICE_AUTO_ROUTE and not bool((payload.profile or "").strip()):
        tags.append("policy_selected")

    _try_auto_feedback_log(
        user_input=payload.text,
        audio_name=name,
        engine=ENGINE_LOCK,
        profile=prof.name,
        latency_ms=int((time.time() - start_time) * 1000),
        tags=tags,
    )

    return SpeakOut(
        ok=True,
        engine=ENGINE_LOCK,
        name=name,
        url=url,
        abs_url=abs_url,
        meta={
            "profile": prof.name,
            "rate_mul": float(tts_meta.get("rate_mul", 1.0)),
            "rate_wpm": _safe_int(tts_meta.get("rate_wpm", 0), 0),
            "volume": _safe_float(tts_meta.get("volume", 1.0), 1.0),
            "leadin_ms": _safe_int(tts_meta.get("leadin_ms", 0), 0),
            "bytes": _safe_int(tts_meta.get("bytes", 0), 0),
            "chunks": _safe_int(tts_meta.get("chunks", 0), 0),
            "voice_request": tts_meta.get("voice_request", ""),
            "voice_selected": tts_meta.get("voice_selected", ""),
            "wpm_bounds": tts_meta.get("wpm_bounds", {}),
            "policy_source": plan["source"],
            "auto_routed": VOICE_AUTO_ROUTE and not bool((payload.profile or "").strip()),
            "context_rate_scale": rate_scale,
            "context_pause_scale": pause_scale,
        },
    )


@voice_router.post("/speak_edge", response_model=SpeakOut)
def speak_edge(payload: SpeakIn, request: Request) -> SpeakOut:
    start_time = time.time()
    delivery = build_delivery_style(payload.text)
    rate_scale = _safe_float(delivery.get("rate_scale"), 1.0)
    pause_scale = _safe_float(delivery.get("pause_scale"), 1.0)

    name, tts_meta = _synthesize_with_edge(payload.text, rate_scale=rate_scale)
    url, abs_url = _build_urls(request, name)

    _append_voice_history(
        engine=EDGE_PREVIEW_ENGINE,
        profile_name="edge_preview",
        text=payload.text,
        filename=name,
        meta=tts_meta,
    )

    _try_auto_feedback_log(
        user_input=payload.text,
        audio_name=name,
        engine=EDGE_PREVIEW_ENGINE,
        profile="edge_preview",
        latency_ms=int((time.time() - start_time) * 1000),
        tags=["auto_logged", "edge"],
    )

    return SpeakOut(
        ok=True,
        engine=EDGE_PREVIEW_ENGINE,
        name=name,
        url=url,
        abs_url=abs_url,
        meta={
            "profile": "edge_preview",
            "voice": tts_meta.get("voice", ""),
            "rate": tts_meta.get("rate", ""),
            "volume": tts_meta.get("volume", ""),
            "bytes": _safe_int(tts_meta.get("bytes", 0), 0),
            "context_rate_scale": rate_scale,
            "context_pause_scale": pause_scale,
        },
    )


@voice_router.get("/selftest")
def selftest(request: Request) -> JSONResponse:
    sample = "SEED voice self test. Omar, if you hear this clearly, the pipeline is healthy."
    name, tts_meta, prof = _synthesize_with_profile(sample, DEFAULT_PROFILE)

    out_path = _safe_audio_path(name)
    exists = out_path.exists()
    size_bytes = out_path.stat().st_size if exists else 0
    ok = bool(exists and size_bytes > 0)

    _url, abs_url = _build_urls(request, name)

    return JSONResponse(
        {
            "ok": ok,
            "engine": ENGINE_LOCK,
            "profile": prof.name,
            "file": name,
            "disk": {"exists": exists, "bytes": size_bytes, "path": str(out_path)},
            "http": {"url": abs_url},
            "applied": {
                "rate_mul": float(tts_meta.get("rate_mul", 1.0)),
                "rate_wpm": _safe_int(tts_meta.get("rate_wpm", 0), 0),
                "volume": _safe_float(tts_meta.get("volume", 1.0), 1.0),
                "leadin_ms": _safe_int(tts_meta.get("leadin_ms", 0), 0),
                "chunks": _safe_int(tts_meta.get("chunks", 0), 0),
                "wpm_bounds": tts_meta.get("wpm_bounds", {}),
            },
            "voice": {
                "voice_request": tts_meta.get("voice_request", ""),
                "voice_selected": tts_meta.get("voice_selected", ""),
            },
        },
        status_code=200 if ok else 500,
    )


@voice_router.get("/selftest_edge")
def selftest_edge(request: Request) -> JSONResponse:
    if not EDGE_PREVIEW_ENABLED:
        raise HTTPException(status_code=503, detail="Edge voice preview is disabled.")

    sample = "SEED edge voice self test. Omar, this is the experimental voice preview."
    name, tts_meta = _synthesize_with_edge(sample)

    out_path = _safe_audio_path(name)
    exists = out_path.exists()
    size_bytes = out_path.stat().st_size if exists else 0
    ok = bool(exists and size_bytes > 0)

    _url, abs_url = _build_urls(request, name)

    return JSONResponse(
        {
            "ok": ok,
            "engine": EDGE_PREVIEW_ENGINE,
            "profile": "edge_preview",
            "file": name,
            "disk": {"exists": exists, "bytes": size_bytes, "path": str(out_path)},
            "http": {"url": abs_url},
            "applied": {
                "voice": tts_meta.get("voice", ""),
                "rate": tts_meta.get("rate", ""),
                "volume": tts_meta.get("volume", ""),
                "bytes": _safe_int(tts_meta.get("bytes", 0), 0),
            },
        },
        status_code=200 if ok else 500,
    )


@voice_router.get("/file/{name}")
def file(name: str) -> FileResponse:
    p = _safe_audio_path(name)
    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    media_type = "audio/mpeg" if p.suffix.lower() == ".mp3" else "audio/wav"
    return FileResponse(path=str(p), media_type=media_type, filename=p.name)


@voice_router.get("/play/{name}", response_class=HTMLResponse)
def play(name: str) -> HTMLResponse:
    suffix = Path(name).suffix.lower()
    mime = "audio/mpeg" if suffix == ".mp3" else "audio/wav"

    src_static = f"/audio/{name}"
    src_fallback = f"/v1/voice/file/{name}"

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Play {name}</title>
</head>
<body style="font-family: system-ui, -apple-system, Segoe UI, Roboto; padding: 24px;">
  <h3>SEED Voice Playback</h3>
  <p><code>{name}</code></p>
  <audio controls autoplay style="width: 100%;">
    <source src="{src_static}" type="{mime}" />
    <source src="{src_fallback}" type="{mime}" />
    Your browser does not support audio playback.
  </audio>
  <p style="opacity: 0.7; margin-top: 12px;">
    Static: <a href="{src_static}">{src_static}</a> |
    Fallback: <a href="{src_fallback}">{src_fallback}</a>
  </p>
</body>
</html>"""
    return HTMLResponse(content=html)
