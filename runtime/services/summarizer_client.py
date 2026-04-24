from __future__ import annotations

import re
from typing import List

_STOP = set(
    """
a an and are as at be but by for if in into is it no nor not of on or s such t that the their then there these they this to was will with you your yours me my our we us
""".split()
)


def _sentences(text: str) -> List[str]:
    text = (text or "").strip()
    sents = re.split(r"(?<=[\.\!\?])\s+", text)
    return [re.sub(r"\s+", " ", s).strip() for s in sents if s.strip()]


def _tokens(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r"[A-Za-z0-9']+", text)]


def summarize_text(text: str, max_sentences: int = 3, max_len: int = 2000) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) > max_len:
        text = text[:max_len]

    sents = _sentences(text)
    if len(sents) <= max_sentences:
        return " ".join(sents)

    words = _tokens(text)
    freq = {}
    for w in words:
        if w in _STOP:
            continue
        freq[w] = freq.get(w, 0) + 1

    scored = []
    for idx, s in enumerate(sents):
        score = sum(freq.get(w, 0) for w in _tokens(s))
        scored.append((idx, score, s))

    top = sorted(scored, key=lambda x: x[1], reverse=True)[:max_sentences]
    top_sorted = [t for t in sorted(top, key=lambda x: x[0])]
    return " ".join(s for _, _, s in top_sorted)
