# runtime/services/voice_profiles.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


# ----------------------------
# Profile model
# ----------------------------
@dataclass(frozen=True)
class VoiceProfile:
    name: str
    rate: float = 1.0  # multiplier (0.85..1.15 typical)
    volume: float = 1.0  # 0.0..1.0
    pause_ms: int = 90  # 0..250 (micro-pauses after . ! ?)
    warmth: float = 0.35  # 0.0..1.0 (text shaping intensity)


# ----------------------------
# Presets (tune these anytime)
# ----------------------------
_PROFILES: Dict[str, VoiceProfile] = {
    "auren_balanced": VoiceProfile(
        name="auren_balanced",
        rate=0.95,
        volume=1.00,
        pause_ms=90,
        warmth=0.35,
    ),
    "auren_intimate": VoiceProfile(
        name="auren_intimate",
        rate=0.90,
        volume=0.92,
        pause_ms=140,
        warmth=0.60,
    ),
    "auren_clear": VoiceProfile(
        name="auren_clear",
        rate=0.98,
        volume=1.00,
        pause_ms=70,
        warmth=0.20,
    ),
}

DEFAULT_PROFILE_NAME = "auren_balanced"


# ----------------------------
# Public helpers used by routes
# ----------------------------
def list_profiles() -> List[str]:
    return sorted(_PROFILES.keys())


def has_profile(name: str) -> bool:
    return (name or "").strip().lower() in _PROFILES


def get_profile(name: str) -> VoiceProfile:
    key = (name or "").strip().lower()
    if not key:
        key = DEFAULT_PROFILE_NAME
    return _PROFILES.get(key, _PROFILES[DEFAULT_PROFILE_NAME])


def profile_dict(name: str) -> dict:
    p = get_profile(name)
    return {
        "name": p.name,
        "rate": p.rate,
        "volume": p.volume,
        "pause_ms": p.pause_ms,
        "warmth": p.warmth,
    }


def all_profiles_dict() -> dict:
    return {k: profile_dict(k) for k in _PROFILES.keys()}
