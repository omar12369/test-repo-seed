from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles


# -----------------------------
# Load .env BEFORE reading env vars
# -----------------------------
def _load_env() -> Optional[str]:
    """
    Load .env from repo root (preferred), then fallback to CWD.

    Returns the path used (string) or None if python-dotenv isn't installed
    or no .env was found/loaded.
    """
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return None

    repo_root = Path(__file__).resolve().parent.parent
    env_path = repo_root / ".env"

    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
        return str(env_path)

    # Fallback: attempt CWD (useful if running from repo root already)
    load_dotenv(override=False)
    return None


ENV_LOADED_FROM = _load_env()


# -----------------------------
# Helpers
# -----------------------------
def _audio_dir() -> str:
    return os.getenv("AUDIO_DIR", "data/audio")


def _web_dir() -> str:
    return os.getenv("WEB_DIR", "web_client")


def _include_router_safe(app: FastAPI, import_path: str, attr: str) -> Tuple[bool, Optional[str]]:
    """
    Import router defensively so the app still boots even if a module is missing.
    """
    try:
        module = __import__(import_path, fromlist=[attr])
        router = getattr(module, attr)
        app.include_router(router)
        return True, None
    except Exception as e:
        return False, f"{import_path}.{attr} -> {e!r}"


# -----------------------------
# App
# -----------------------------
app = FastAPI(
    title="SEED Runtime",
    version=os.getenv("SEED_VERSION", "0.1.0"),
)


# -----------------------------
# Mount static directories
# -----------------------------
audio_dir = _audio_dir()
os.makedirs(audio_dir, exist_ok=True)
app.mount("/audio", StaticFiles(directory=audio_dir), name="audio")

web_dir = _web_dir()
if os.path.isdir(web_dir):
    app.mount("/web", StaticFiles(directory=web_dir, html=True), name="web")


# -----------------------------
# Routers (safe include)
# -----------------------------
router_errors: list[str] = []

# Phase 3 baseline
ok, err = _include_router_safe(app, "runtime.routes.voice", "voice_router")
if not ok and err:
    router_errors.append(err)

ok, err = _include_router_safe(app, "runtime.routes.voice_profiles", "profile_router")
if not ok and err:
    router_errors.append(err)

ok, err = _include_router_safe(app, "runtime.routes.voice_stream", "stream_router")
if not ok and err:
    router_errors.append(err)

ok, err = _include_router_safe(app, "runtime.routes.audio_norange", "audio_router")
if not ok and err:
    router_errors.append(err)

# Phase 4: realtime WS demo + ws endpoint
ok, err = _include_router_safe(app, "runtime.routes.voice_realtime", "realtime_router")
if not ok and err:
    router_errors.append(err)

# Phase 4: realtime chat -> voice (TTS)
ok, err = _include_router_safe(app, "runtime.routes.voice_realtime_chat", "chat_router")
if not ok and err:
    router_errors.append(err)

# Phase 5A / 6A: feedback loop + summary
ok, err = _include_router_safe(app, "runtime.routes.feedback", "feedback_router")
if not ok and err:
    router_errors.append(err)

# Phase 6B: voice policy layer
ok, err = _include_router_safe(app, "runtime.routes.voice_policy", "policy_router")
if not ok and err:
    router_errors.append(err)


# -----------------------------
# Basic health/status
# -----------------------------
@app.get("/health", tags=["system"])
def health():
    return {"ok": True}


@app.get("/healthz", tags=["system"])
def healthz():
    return {"ok": True}


@app.get("/v1/ping", tags=["system"])
def ping(msg: str = "seed"):
    return {"pong": msg}


@app.get("/status", tags=["system"])
def status():
    return {
        "ok": True,
        "version": os.getenv("SEED_VERSION", "0.1.0"),
        "env_loaded_from": ENV_LOADED_FROM,
        "audio_dir": os.path.abspath(audio_dir),
        "web_dir": os.path.abspath(web_dir) if os.path.isdir(web_dir) else None,
        "router_import_errors": router_errors,
    }
