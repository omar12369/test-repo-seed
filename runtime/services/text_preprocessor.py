# runtime/services/text_preprocessor.py
from __future__ import annotations

import re
from typing import List

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]*`")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_JSON_LIKE_RE = re.compile(r"^\s*[\{\[].*[\}\]]\s*$", re.DOTALL)

# Remove obvious “log junk” lines (conservative)
_JUNK_LINE_RE = re.compile(
    r"^\s*(INFO:|DEBUG:|WARNING:|ERROR:|Traceback|File \"|line \d+|Exception:)\b",
    re.IGNORECASE,
)

# Collapse long repeated punctuation like ".........." -> "..."
_REPEAT_PUNCT_RE = re.compile(r"([\.!\?,;:\-—_])\1{3,}")


def clean_text(text: str) -> str:
    """
    Conservative cleaner:
    - Keeps leading words like names ("Omar, ...")
    - Removes URLs, code blocks, HTML tags, and obvious stacktrace/log lines
    - Normalizes whitespace + repeated punctuation
    """
    if not isinstance(text, str):
        return ""

    s = text.strip()
    if not s:
        return ""

    # If the entire payload is JSON-like, don't try to read it
    if _JSON_LIKE_RE.match(s):
        return ""

    # Remove fenced code blocks
    s = _CODE_FENCE_RE.sub(" ", s)

    # Remove inline code ticks
    s = _INLINE_CODE_RE.sub(" ", s)

    # Remove URLs
    s = _URL_RE.sub(" ", s)

    # Remove HTML tags
    s = _HTML_TAG_RE.sub(" ", s)

    # Remove obvious junk/log lines
    lines = []
    for line in s.splitlines():
        if _JUNK_LINE_RE.match(line):
            continue
        lines.append(line)
    s = " ".join(lines)

    # Normalize repeated punctuation like ".........." -> "..."
    s = _REPEAT_PUNCT_RE.sub(r"\1\1\1", s)

    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()

    return s


def segment_for_tts(text: str, max_chars: int = 260) -> List[str]:
    """
    Split text into speakable chunks for TTS streaming or multi-part generation.
    Keeps punctuation so the voice has natural pauses.
    """
    s = clean_text(text)
    if not s:
        return []

    # Split by sentence-ish boundaries while keeping punctuation
    parts = re.split(r"(?<=[\.\!\?\:;])\s+", s)

    chunks: List[str] = []
    buf = ""

    for p in parts:
        if not p:
            continue

        # If a single part is huge, hard-wrap it
        if len(p) > max_chars:
            # flush buffer first
            if buf:
                chunks.append(buf.strip())
                buf = ""
            # hard wrap large sentence
            for i in range(0, len(p), max_chars):
                chunks.append(p[i : i + max_chars].strip())
            continue

        if not buf:
            buf = p
        elif len(buf) + 1 + len(p) <= max_chars:
            buf = f"{buf} {p}"
        else:
            chunks.append(buf.strip())
            buf = p

    if buf:
        chunks.append(buf.strip())

    return chunks
