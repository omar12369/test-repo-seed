# runtime/services/voice/preprocess.py
from __future__ import annotations

import html
import os
import re
from typing import List


def _b(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _i(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _s(name: str, default: str) -> str:
    return os.getenv(name, default)


# ---- ENV knobs (already in your .env) ----
PRE_REMOVE_PUNCT = _b("VOICE_PRE_REMOVE_PUNCT", True)  # defensive filter of garbage tokens
PRE_STRICT_QUOTES = _b("VOICE_PRE_STRICT_QUOTES", True)
PRE_COMMA_MS = _i("VOICE_PRE_COMMA_BREAK_MS", 120)
PRE_SENT_MS = _i("VOICE_PRE_SENTENCE_BREAK_MS", 220)
PRE_ELLIPSIS_MS = _i("VOICE_PRE_MAX_ELLIPSIS_BREAK_MS", 400)
PRE_TONE = _s("VOICE_PRE_DEFAULT_TONE", "neutral")  # neutral|soft|warm|bright

# ---- Normalizers ----
SMART_QUOTES = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u201f": '"',
}
SPACE_RX = re.compile(r"\s+", re.MULTILINE)
URL_RX = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RX = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b")
TAG_RX = re.compile(r"<[^>]+>")  # HTML-ish tags
CODE_FENCE_RX = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RX = re.compile(r"`[^`]+`")
JSON_BLOB_RX = re.compile(r"\{[^{}]{0,500}:[^{}]{0,500}\}")  # remove obvious key:value blobs
BRACKETS_RX = re.compile(r"(\[[^\]]*]|\([^)]+\)|\{[^{}]+\})")

# keep readable punctuation only
if PRE_REMOVE_PUNCT:
    JUNK_RX = re.compile(r"[^\w\s\.\,\!\?\;\:\'\-\u2026]")  # allow word chars + .,!?;:'- + ellipsis
else:
    JUNK_RX = re.compile(r"[^\S\s]")  # noop

ELLIPSIS_RX = re.compile(r"\.\.\.+")
MULTI_PUNCT_RX = re.compile(r"([,;:\.\!\?]){2,}")

# Sentence split on ., !, ?, or linebreak
SENTENCE_RX = re.compile(r"[^\r\n\.!\?]+[\.!\?]*", re.MULTILINE)


def _normalize_quotes(t: str) -> str:
    return "".join(SMART_QUOTES.get(ch, ch) for ch in t)


def _strip_wrapping_quotes(t: str) -> str:
    if not PRE_STRICT_QUOTES:
        return t
    s = t.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1].strip()
    return s


def preprocess_for_tts(raw: str) -> str:
    """Return a clean, human-readable string for pyttsx3 or for SSML building."""
    if not raw:
        return ""
    t = str(raw)

    # decode entities, normalize unicode quotes/spaces
    t = html.unescape(t)
    t = _normalize_quotes(t)

    # strip obvious non-speech content
    t = CODE_FENCE_RX.sub(" ", t)
    t = INLINE_CODE_RX.sub(" ", t)
    t = JSON_BLOB_RX.sub(" ", t)
    t = URL_RX.sub(" ", t)
    t = EMAIL_RX.sub(" ", t)
    t = TAG_RX.sub(" ", t)
    t = BRACKETS_RX.sub(" ", t)

    # ellipsis -> single glyph + remember for SSML break later
    t = ELLIPSIS_RX.sub("…", t)

    # remove junk symbols, collapse punctuation runs
    t = JUNK_RX.sub(" ", t)
    t = MULTI_PUNCT_RX.sub(lambda m: m.group(1), t)

    # firm up whitespace
    t = SPACE_RX.sub(" ", t).strip()

    # optional strict outer quotes
    t = _strip_wrapping_quotes(t)

    # Safety: if everything got stripped, provide a short placeholder
    if not t:
        t = "…"

    return t


def build_ssml_from_text(
    text: str,
    voice_name: str,
    rate_str: str = "+0%",
    pitch_str: str = "+0st",
    tone: str = PRE_TONE,  # neutral|soft|warm|bright
) -> str:
    """
    Build robust SSML:
      - split sentences and insert <break> after them
      - shorter breaks after commas/semicolons
      - extra break for ellipsis “…” up to PRE_ELLIPSIS_MS
    """
    # Pick a mild prosody tweak by tone (kept subtle)
    tone_rate = {"neutral": "+0%", "soft": "-2%", "warm": "+1%", "bright": "+3%"}.get(tone, "+0%")
    # combine external rate with tone bias (very small; Edge sums them)
    if rate_str.startswith(("+", "-")):
        # leave as provided; tone is mild anyway
        pass

    # per-sentence assembly
    sentences: List[str] = []
    for chunk in SENTENCE_RX.findall(text):
        s = chunk.strip()
        if not s:
            continue
        # inject comma breaks and ellipsis breaks
        s = (
            s.replace("…", f"</s><break time='{min(PRE_ELLIPSIS_MS, 800)}ms'/><s>")
            .replace(",", f",<break time='{PRE_COMMA_MS}ms'/>")
            .replace(";", f";<break time='{PRE_COMMA_MS}ms'/>")
        )
        sentences.append(f"<s>{s}</s>")

    if not sentences:
        sentences = [f"<s>{text}</s>"]

    body = f"<break time='{PRE_SENT_MS}ms'/>".join(sentences)

    ssml = f"""<speak version="1.0" xml:lang="en-US">
  <voice name="{voice_name}">
    <prosody rate="{rate_str}" pitch="{pitch_str}">
      {body}
    </prosody>
  </voice>
</speak>"""
    return ssml
