from __future__ import annotations

import time
from collections import Counter
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from runtime.services.feedback import (
    feedback_storage_status,
    log_feedback,
    read_recent_feedback,
)

feedback_router = APIRouter(prefix="/v1/feedback", tags=["feedback"])


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class FeedbackIn(BaseModel):
    ts: Optional[int] = None
    session_id: Optional[str] = None

    user_input: str = Field(..., min_length=1, max_length=10000)
    system_output: str = Field(..., min_length=1, max_length=20000)

    channel: str = Field(..., min_length=1, max_length=100)
    input_mode: Optional[str] = Field(default=None, max_length=100)
    output_mode: Optional[str] = Field(default=None, max_length=100)

    engine: str = Field(..., min_length=1, max_length=100)
    profile: Optional[str] = Field(default=None, max_length=100)

    success: bool

    latency_ms: Optional[int] = Field(default=None, ge=0)
    playback_ok: Optional[bool] = None

    user_feedback: Optional[str] = Field(default=None, max_length=100)
    feedback_tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = Field(default=None, max_length=5000)


class FeedbackOut(BaseModel):
    ok: bool
    item: Dict[str, Any]


# -----------------------------------------------------------------------------
# Safe helpers
# -----------------------------------------------------------------------------
def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "on"}:
            return True
        if v in {"false", "0", "no", "off"}:
            return False
    try:
        return bool(value)
    except Exception:
        return default


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        return str(value)
    except Exception:
        return default


def _safe_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            s = _safe_str(item).strip()
            if s:
                out.append(s)
        return out
    s = _safe_str(value).strip()
    return [s] if s else []


def _avg(values: List[int]) -> Optional[int]:
    if not values:
        return None
    return int(sum(values) / len(values))


def _quality_trend(
    *,
    total: int,
    successes: int,
    positive_feedback: int,
    negative_feedback: int,
) -> str:
    if total == 0:
        return "no_data"

    success_ratio = successes / total

    if negative_feedback > positive_feedback and negative_feedback >= 2:
        return "needs_attention"

    if success_ratio >= 0.9 and positive_feedback >= negative_feedback:
        return "improving"

    if success_ratio >= 0.75:
        return "stable"

    return "needs_attention"


def _build_recommendations(
    *,
    total: int,
    successes: int,
    failures: int,
    avg_latency_ms: Optional[int],
    engine_counts: Dict[str, int],
    positive_feedback: int,
    negative_feedback: int,
    tag_counter: Counter[str],
) -> List[str]:
    recs: List[str] = []

    if total == 0:
        return ["Collect more interactions before generating recommendations."]

    if engine_counts.get("edge-tts", 0) > engine_counts.get("pyttsx3", 0):
        recs.append("Edge voice is currently the stronger user-facing output path.")
    elif engine_counts.get("pyttsx3", 0) > 0 and engine_counts.get("edge-tts", 0) == 0:
        recs.append("Collect more edge-tts samples before comparing engine quality.")

    if avg_latency_ms is not None:
        if avg_latency_ms > 2000:
            recs.append("Average latency is high; review synthesis path and response timing.")
        elif avg_latency_ms < 1200:
            recs.append("Latency is in a healthy range for current voice interactions.")

    if failures > 0:
        recs.append("Review failed interactions and identify repeated error conditions.")

    if positive_feedback > negative_feedback and positive_feedback > 0:
        recs.append("Recent feedback is trending positive; preserve current successful patterns.")
    elif negative_feedback > positive_feedback:
        recs.append(
            "Negative feedback is outweighing positive feedback; inspect tone, timing, and output quality."
        )

    if tag_counter.get("great_tone", 0) > 0:
        recs.append("Tone quality is being recognized positively; keep using that output style.")
    if tag_counter.get("too_robotic", 0) > 0:
        recs.append("Robotic voice feedback is present; prefer edge voice where possible.")
    if tag_counter.get("realtime", 0) > 0:
        recs.append("Realtime interactions are now entering the learning stream successfully.")

    if not recs:
        recs.append(
            "System is stable; continue collecting more feedback before making larger adjustments."
        )

    return recs


# -----------------------------------------------------------------------------
# Internal helper for other routes/services
# -----------------------------------------------------------------------------
def log_feedback_internal(
    *,
    user_input: str,
    system_output: str,
    channel: str,
    input_mode: Optional[str],
    output_mode: Optional[str],
    engine: str,
    profile: Optional[str],
    success: bool,
    latency_ms: Optional[int],
    playback_ok: Optional[bool],
    user_feedback: Optional[str],
    feedback_tags: Optional[List[str]],
    notes: Optional[str],
    session_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Safe internal helper for other runtime modules.

    Returns the logged item if successful, otherwise None.
    Never raises.
    """
    payload = {
        "ts": int(time.time()),
        "session_id": session_id,
        "user_input": user_input,
        "system_output": system_output,
        "channel": channel,
        "input_mode": input_mode,
        "output_mode": output_mode,
        "engine": engine,
        "profile": profile,
        "success": success,
        "latency_ms": latency_ms,
        "playback_ok": playback_ok,
        "user_feedback": user_feedback,
        "feedback_tags": feedback_tags or [],
        "notes": notes,
    }

    try:
        return log_feedback(payload)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@feedback_router.get("/status")
def feedback_status() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "storage": feedback_storage_status(),
        }
    )


@feedback_router.post("/log", response_model=FeedbackOut)
def log_feedback_endpoint(payload: FeedbackIn) -> FeedbackOut:
    raw = payload.model_dump()

    if raw.get("ts") is None:
        raw["ts"] = int(time.time())

    try:
        item = log_feedback(raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Feedback log failed: {exc}") from exc

    return FeedbackOut(ok=True, item=item)


@feedback_router.get("/recent")
def recent_feedback(limit: int = Query(default=20, ge=1, le=200)) -> JSONResponse:
    try:
        items = read_recent_feedback(limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Feedback read failed: {exc}") from exc

    return JSONResponse(
        {
            "ok": True,
            "count": len(items),
            "items": items,
        }
    )


@feedback_router.get("/summary")
def feedback_summary(limit: int = Query(default=50, ge=1, le=500)) -> JSONResponse:
    """
    Phase 6A: first intelligence summary layer built on top of feedback data.
    """
    try:
        items = read_recent_feedback(limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Feedback summary failed: {exc}") from exc

    total = len(items)
    success_items = [item for item in items if _safe_bool(item.get("success"), False)]
    failure_items = [item for item in items if not _safe_bool(item.get("success"), False)]

    successes = len(success_items)
    failures = len(failure_items)

    latency_values = [
        _safe_int(item.get("latency_ms"), 0) for item in items if item.get("latency_ms") is not None
    ]
    latency_values = [v for v in latency_values if v >= 0]
    avg_latency_ms = _avg(latency_values)

    engine_counter: Counter[str] = Counter()
    profile_counter: Counter[str] = Counter()
    tag_counter: Counter[str] = Counter()

    positive_feedback = 0
    negative_feedback = 0

    for item in items:
        engine = _safe_str(item.get("engine")).strip()
        if engine:
            engine_counter[engine] += 1

        profile = _safe_str(item.get("profile")).strip()
        if profile:
            profile_counter[profile] += 1

        tags = _safe_str_list(item.get("feedback_tags"))
        for tag in tags:
            tag_counter[tag] += 1

        user_feedback = _safe_str(item.get("user_feedback")).strip().lower()
        if user_feedback == "positive":
            positive_feedback += 1
        elif user_feedback == "negative":
            negative_feedback += 1

    preferred_engine = None
    if engine_counter:
        preferred_engine = engine_counter.most_common(1)[0][0]

    preferred_profile = None
    if profile_counter:
        preferred_profile = profile_counter.most_common(1)[0][0]

    trend = _quality_trend(
        total=total,
        successes=successes,
        positive_feedback=positive_feedback,
        negative_feedback=negative_feedback,
    )

    recommendations = _build_recommendations(
        total=total,
        successes=successes,
        failures=failures,
        avg_latency_ms=avg_latency_ms,
        engine_counts=dict(engine_counter),
        positive_feedback=positive_feedback,
        negative_feedback=negative_feedback,
        tag_counter=tag_counter,
    )

    return JSONResponse(
        {
            "ok": True,
            "window": {
                "requested_limit": limit,
                "actual_count": total,
            },
            "summary": {
                "preferred_engine": preferred_engine,
                "preferred_profile": preferred_profile,
                "avg_latency_ms": avg_latency_ms,
                "success_count": successes,
                "failure_count": failures,
                "quality_trend": trend,
                "positive_feedback_count": positive_feedback,
                "negative_feedback_count": negative_feedback,
                "engine_counts": dict(engine_counter),
                "profile_counts": dict(profile_counter),
                "top_feedback_tags": dict(tag_counter.most_common(10)),
                "recommendations": recommendations,
            },
        }
    )
