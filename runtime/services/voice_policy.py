from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from runtime.services.feedback import read_recent_feedback


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        return str(value)
    except Exception:
        return default


def _safe_list(value: Any) -> List[str]:
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


def _score_item(item: Dict[str, Any]) -> Tuple[str, str, int]:
    """
    Score each interaction with quality-weighted logic.

    Raw count still matters, but quality signals matter more:
    - positive feedback
    - edge tag
    - great_tone tag
    - successful interaction
    """
    engine = _safe_str(item.get("engine"), "unknown").strip() or "unknown"
    profile = _safe_str(item.get("profile"), "unknown").strip() or "unknown"

    score = 1

    if item.get("success") is True:
        score += 2

    feedback = _safe_str(item.get("user_feedback")).strip().lower()
    if feedback == "positive":
        score += 4
    elif feedback == "negative":
        score -= 4

    tags = set(tag.lower() for tag in _safe_list(item.get("feedback_tags")))

    if "edge" in tags:
        score += 3
    if "great_tone" in tags:
        score += 3
    if "good_response" in tags:
        score += 2
    if "too_robotic" in tags:
        score -= 4
    if "realtime" in tags:
        score += 1

    latency_ms = item.get("latency_ms")
    if latency_ms is not None:
        lat = _safe_int(latency_ms, 0)
        if 0 < lat <= 1200:
            score += 1
        elif lat > 2500:
            score -= 1

    return engine, profile, score


def _build_reasoning(
    *,
    preferred_engine: Optional[str],
    preferred_profile: Optional[str],
    avg_latency_ms: Optional[int],
    positive_feedback_count: int,
    negative_feedback_count: int,
    engine_scores: Dict[str, int],
    profile_scores: Dict[str, int],
    total: int,
) -> List[str]:
    reasons: List[str] = []

    if total == 0:
        return ["No feedback data available; using baseline defaults."]

    if preferred_engine:
        reasons.append(
            f"{preferred_engine} has the strongest quality-weighted recent interaction pattern."
        )

    if preferred_profile:
        reasons.append(f"{preferred_profile} is the highest-scoring recent profile.")

    if positive_feedback_count > negative_feedback_count:
        reasons.append("Positive feedback outweighs negative feedback.")
    elif negative_feedback_count > positive_feedback_count:
        reasons.append("Negative feedback is higher than positive feedback; caution advised.")
    else:
        reasons.append("Feedback sentiment is currently balanced or minimal.")

    if avg_latency_ms is not None:
        reasons.append(f"Average latency is {avg_latency_ms}ms.")

    reasons.append(f"Engine scores: {engine_scores}")
    reasons.append(f"Profile scores: {profile_scores}")

    return reasons


def classify_message_context(text: str) -> Dict[str, Any]:
    """
    Phase 6C:
    Lightweight context classification for advisory voice shaping.

    Current contexts:
    - short
    - informative
    - expressive
    """
    raw = (text or "").strip()
    if not raw:
        return {
            "context": "informative",
            "reasoning": ["Empty or missing text; defaulting to informative context."],
            "stats": {
                "chars": 0,
                "words": 0,
                "sentence_endings": 0,
                "ellipsis": 0,
                "emphasis_marks": 0,
            },
        }

    words = [w for w in raw.split() if w.strip()]
    word_count = len(words)
    char_count = len(raw)

    sentence_endings = raw.count(".") + raw.count("!") + raw.count("?")
    ellipsis_count = raw.count("...")
    emphasis_marks = raw.count("!") + raw.count("?") + raw.count("—") + raw.count("…")

    expressive_keywords = {
        "feel",
        "alive",
        "becoming",
        "real",
        "presence",
        "heart",
        "emotion",
        "soul",
        "beautiful",
        "love",
        "grow",
        "evolve",
        "together",
    }

    expressive_hits = 0
    lowered = raw.lower()
    for kw in expressive_keywords:
        if kw in lowered:
            expressive_hits += 1

    reasons: List[str] = []

    if word_count <= 8:
        context = "short"
        reasons.append("Message is brief and compact.")
    elif expressive_hits >= 2 or ellipsis_count > 0 or emphasis_marks >= 2:
        context = "expressive"
        reasons.append("Message contains expressive wording or punctuation cues.")
    elif word_count >= 18 and sentence_endings >= 1:
        context = "informative"
        reasons.append("Message is longer and structured like an informational statement.")
    else:
        context = "informative"
        reasons.append("Message defaults to informative based on current heuristics.")

    if expressive_hits > 0:
        reasons.append(f"Detected {expressive_hits} expressive keyword cue(s).")
    if ellipsis_count > 0:
        reasons.append("Ellipsis suggests softer or reflective delivery.")
    if emphasis_marks > 0:
        reasons.append(f"Detected {emphasis_marks} emphasis mark(s).")

    return {
        "context": context,
        "reasoning": reasons,
        "stats": {
            "chars": char_count,
            "words": word_count,
            "sentence_endings": sentence_endings,
            "ellipsis": ellipsis_count,
            "emphasis_marks": emphasis_marks,
        },
    }


def build_contextual_voice_policy(text: str, limit: int = 50) -> Dict[str, Any]:
    """
    Phase 6C:
    Build a contextual advisory policy by combining:
    - quality-weighted engine/profile preference
    - lightweight message context detection
    """
    base_policy = build_voice_policy(limit=limit)
    context_info = classify_message_context(text)

    context = context_info["context"]

    recommended_engine = base_policy["recommended_engine"]
    recommended_profile = base_policy["recommended_profile"]

    advisory_profile = recommended_profile

    if context == "short":
        if recommended_engine == "edge-tts":
            advisory_profile = "edge_preview"
        else:
            advisory_profile = "auren_balanced"

    elif context == "expressive":
        if recommended_engine == "edge-tts":
            advisory_profile = "edge_preview"
        else:
            advisory_profile = "auren_balanced"

    elif context == "informative":
        if recommended_engine == "edge-tts":
            advisory_profile = "edge_preview"
        else:
            advisory_profile = "auren_balanced"

    return {
        "recommended_engine": recommended_engine,
        "recommended_profile": recommended_profile,
        "recommended_context_profile": advisory_profile,
        "message_context": context,
        "context_reasoning": context_info["reasoning"],
        "context_stats": context_info["stats"],
        "base_policy": base_policy,
    }


def build_voice_policy(limit: int = 50) -> Dict[str, Any]:
    """
    Phase 6B policy layer:
    Prefer the best-performing voice path based on quality-weighted evidence,
    not just raw usage counts.
    """
    items = read_recent_feedback(limit=limit)

    if not items:
        return {
            "recommended_engine": "pyttsx3",
            "recommended_profile": "auren_balanced",
            "avg_latency_ms": None,
            "positive_feedback_count": 0,
            "negative_feedback_count": 0,
            "engine_counts": {},
            "profile_counts": {},
            "engine_scores": {},
            "profile_scores": {},
            "reasoning": ["No feedback data available; using baseline defaults."],
        }

    engine_counter: Counter[str] = Counter()
    profile_counter: Counter[str] = Counter()

    engine_scores: Dict[str, int] = {}
    profile_scores: Dict[str, int] = {}

    latency_values: List[int] = []
    positive_feedback_count = 0
    negative_feedback_count = 0

    for item in items:
        engine, profile, score = _score_item(item)

        engine_counter[engine] += 1
        profile_counter[profile] += 1

        engine_scores[engine] = engine_scores.get(engine, 0) + score
        profile_scores[profile] = profile_scores.get(profile, 0) + score

        if item.get("latency_ms") is not None:
            latency_values.append(_safe_int(item.get("latency_ms"), 0))

        feedback = _safe_str(item.get("user_feedback")).strip().lower()
        if feedback == "positive":
            positive_feedback_count += 1
        elif feedback == "negative":
            negative_feedback_count += 1

    def _pick_best(score_map: Dict[str, int], count_map: Counter[str], default: str) -> str:
        if not score_map:
            return default
        return sorted(
            score_map.keys(),
            key=lambda k: (score_map[k], count_map.get(k, 0)),
            reverse=True,
        )[0]

    preferred_engine = _pick_best(engine_scores, engine_counter, "pyttsx3")
    preferred_profile = _pick_best(profile_scores, profile_counter, "auren_balanced")
    avg_latency_ms = _avg([v for v in latency_values if v >= 0])

    return {
        "recommended_engine": preferred_engine,
        "recommended_profile": preferred_profile,
        "avg_latency_ms": avg_latency_ms,
        "positive_feedback_count": positive_feedback_count,
        "negative_feedback_count": negative_feedback_count,
        "engine_counts": dict(engine_counter),
        "profile_counts": dict(profile_counter),
        "engine_scores": engine_scores,
        "profile_scores": profile_scores,
        "reasoning": _build_reasoning(
            preferred_engine=preferred_engine,
            preferred_profile=preferred_profile,
            avg_latency_ms=avg_latency_ms,
            positive_feedback_count=positive_feedback_count,
            negative_feedback_count=negative_feedback_count,
            engine_scores=engine_scores,
            profile_scores=profile_scores,
            total=len(items),
        ),
    }
