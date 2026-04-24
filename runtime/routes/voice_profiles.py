# runtime/routes/voice_profiles.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from runtime.services.voice_profiles import (
    DEFAULT_PROFILE_NAME,
    all_profiles_dict,
    has_profile,
    list_profiles,
    profile_dict,
)

profile_router = APIRouter(prefix="/v1/voice", tags=["voice-profiles"])


@profile_router.get(
    "/presets",
    operation_id="v1_voice_presets_list",
)
def presets() -> dict:
    """
    List available voice presets (profiles) and return full details.
    """
    return {
        "default": DEFAULT_PROFILE_NAME,
        "profiles": list_profiles(),
        "details": all_profiles_dict(),
    }


@profile_router.get(
    "/presets/{name}",
    operation_id="v1_voice_presets_get",
)
def read_preset(name: str) -> dict:
    """
    Return a single preset by name.
    """
    key = name.strip().lower()
    if not has_profile(key):
        raise HTTPException(status_code=404, detail="Preset not found")

    return {
        "default": DEFAULT_PROFILE_NAME,
        "name": key,
        "profile": profile_dict(key),
        "is_default": (key == DEFAULT_PROFILE_NAME),
    }


__all__ = ["profile_router"]
