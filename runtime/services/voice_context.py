from __future__ import annotations

from typing import Dict


def classify_message_context(text: str) -> str:
    t = (text or "").strip()

    words = len(t.split())
    length = len(t)

    if length < 40 or words <= 6:
        return "short"

    expressive_markers = ["!", "...", "?", "—"]
    expressive_keywords = [
        "feel",
        "alive",
        "becoming",
        "listen",
        "understand",
        "moment",
        "truth",
        "real",
        "together",
        "evolve",
    ]

    if any(m in t for m in expressive_markers):
        return "expressive"

    if any(k in t.lower() for k in expressive_keywords):
        return "expressive"

    if length > 120 or words > 20:
        return "informative"

    return "informative"


def build_delivery_style(text: str) -> Dict[str, float]:
    context = classify_message_context(text)

    if context == "short":
        return {
            "pause_scale": 0.85,
            "rate_scale": 1.05,
        }

    if context == "informative":
        return {
            "pause_scale": 1.0,
            "rate_scale": 1.0,
        }

    if context == "expressive":
        return {
            "pause_scale": 1.15,
            "rate_scale": 0.92,
        }

    return {
        "pause_scale": 1.0,
        "rate_scale": 1.0,
    }
