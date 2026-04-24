from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from runtime.services.voice_policy import (
    build_contextual_voice_policy,
    build_voice_policy,
)

policy_router = APIRouter(prefix="/v1/voice/policy", tags=["voice-policy"])


@policy_router.get("")
def get_voice_policy(limit: int = Query(default=50, ge=1, le=200)) -> JSONResponse:
    policy = build_voice_policy(limit=limit)

    return JSONResponse(
        {
            "ok": True,
            "policy": policy,
        }
    )


@policy_router.get("/context")
def get_contextual_voice_policy(
    text: str = Query(..., min_length=1),
    limit: int = Query(default=50, ge=1, le=200),
) -> JSONResponse:
    policy = build_contextual_voice_policy(text=text, limit=limit)

    return JSONResponse(
        {
            "ok": True,
            "policy": policy,
        }
    )
