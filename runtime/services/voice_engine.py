# runtime/services/voice_engine.py
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from runtime.services.voice_profiles import VoiceProfile


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)).strip())
    except Exception:
        return default


def _pick_voice(engine: Any, match: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Bind a voice by matching name OR id (case-insensitive substring).
    Returns (chosen_voice_id, chosen_voice_name).
    """
    match_l = (match or "").strip().lower()
    if not match_l:
        return None, None

    voices = engine.getProperty("voices") or []
    for v in voices:
        vid = getattr(v, "id", "") or ""
        vname = getattr(v, "name", "") or ""
        if match_l in vid.lower() or match_l in vname.lower():
            try:
                engine.setProperty("voice", vid)
                return vid, vname
            except Exception:
                continue

    return None, None


def _apply_rate_volume(engine: Any, profile: VoiceProfile) -> Tuple[int, float]:
    """
    Map profile.rate (multiplier) onto pyttsx3 integer rate.
    """
    base_rate = _env_int("VOICE_PYTTSX3_BASE_RATE", 185)
    min_rate = _env_int("VOICE_PYTTSX3_MIN_RATE", 80)
    max_rate = _env_int("VOICE_PYTTSX3_MAX_RATE", 350)

    applied_rate = int(round(base_rate * float(profile.rate)))
    if applied_rate < min_rate:
        applied_rate = min_rate
    if applied_rate > max_rate:
        applied_rate = max_rate

    try:
        engine.setProperty("rate", applied_rate)
    except Exception:
        pass

    applied_volume = float(profile.volume)
    if applied_volume < 0.0:
        applied_volume = 0.0
    if applied_volume > 1.0:
        applied_volume = 1.0

    try:
        engine.setProperty("volume", applied_volume)
    except Exception:
        pass

    return applied_rate, applied_volume


def _wait_for_file(path: Path) -> None:
    """
    Prevents missing/silent WAV files on Windows by waiting for existence and size.
    """
    wait_ms = _env_int("VOICE_WRITE_WAIT_MS", 2500)
    poll_ms = _env_int("VOICE_WRITE_POLL_MS", 50)
    min_bytes = _env_int("VOICE_MIN_BYTES", 800)

    deadline = time.time() + (wait_ms / 1000.0)
    while time.time() < deadline:
        try:
            if path.exists() and path.stat().st_size >= min_bytes:
                return
        except Exception:
            pass
        time.sleep(poll_ms / 1000.0)


def synthesize_wav(text: str, profile: VoiceProfile, out_path: Path) -> Dict[str, Any]:
    """
    Synthesize WAV via pyttsx3 and return metadata.
    """
    import pyttsx3  # local import keeps startup light

    voice_match = os.getenv("VOICE_PYTTSX3_VOICE", "").strip()

    engine = pyttsx3.init()

    chosen_voice_id: Optional[str] = None
    chosen_voice_name: Optional[str] = None
    if voice_match:
        chosen_voice_id, chosen_voice_name = _pick_voice(engine, voice_match)

    applied_rate, applied_volume = _apply_rate_volume(engine, profile)

    engine.save_to_file(text, str(out_path))
    engine.runAndWait()

    _wait_for_file(out_path)

    return {
        "engine": "pyttsx3",
        "voice_match": voice_match or None,
        "chosen_voice_id": chosen_voice_id,
        "chosen_voice_name": chosen_voice_name,
        "rate_multiplier": float(profile.rate),
        "applied_rate": int(applied_rate),
        "applied_volume": float(applied_volume),
        "file_bytes": out_path.stat().st_size if out_path.exists() else 0,
    }
