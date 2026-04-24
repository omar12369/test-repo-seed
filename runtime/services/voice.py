from __future__ import annotations

import contextlib
import io
import os
import re
import uuid
import wave
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pyttsx3

# -----------------------------------------------------------------------------
# ENV / PATHS
# -----------------------------------------------------------------------------
AUDIO_DIR = Path(os.getenv("AUDIO_DIR", "data/audio")).resolve()
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

VOICE_ENGINE = os.getenv("VOICE_ENGINE", "pyttsx3").strip().lower()
VOICE_PYTTSX3_VOICE = os.getenv("VOICE_PYTTSX3_VOICE", "").strip()

VOICE_RATE_MIN = float(os.getenv("VOICE_RATE_MIN", "0.55"))
VOICE_RATE_MAX = float(os.getenv("VOICE_RATE_MAX", "1.05"))

# Slightly softened range for steadier and less robotic cadence
WPM_MIN = int(os.getenv("VOICE_WPM_MIN", "102"))
WPM_MAX = int(os.getenv("VOICE_WPM_MAX", "162"))

# Default pauses (ms)
PAUSE_MS_COMMA_DEFAULT = int(os.getenv("PAUSE_MS_COMMA", "165"))
PAUSE_MS_SENTENCE_DEFAULT = int(os.getenv("PAUSE_MS_SENTENCE", "300"))
PAUSE_MS_ELLIPSIS_DEFAULT = int(os.getenv("PAUSE_MS_ELLIPSIS", "460"))

VOICE_DEFAULT_RATE_MUL = float(os.getenv("VOICE_DEFAULT_RATE_MUL", "0.89"))
VOICE_VOLUME_DEFAULT = float(os.getenv("VOICE_VOLUME_DEFAULT", "1.0"))

# Optional lead-in silence for safer browser playback
VOICE_WAV_LEADIN_MS = int(os.getenv("VOICE_WAV_LEADIN_MS", "0"))

# Short-phrase shaping
SHORT_PHRASE_COMMA_PAUSE_MS = int(os.getenv("VOICE_SHORT_PHRASE_COMMA_PAUSE_MS", "120"))
SHORT_PHRASE_SENTENCE_PAUSE_MS = int(os.getenv("VOICE_SHORT_PHRASE_SENTENCE_PAUSE_MS", "240"))

# Very short phrase handling for less clipped/robotic delivery
VERY_SHORT_PHRASE_WORDS = int(os.getenv("VOICE_VERY_SHORT_PHRASE_WORDS", "3"))
VERY_SHORT_PHRASE_COMMA_PAUSE_MS = int(os.getenv("VOICE_VERY_SHORT_PHRASE_COMMA_PAUSE_MS", "140"))
VERY_SHORT_PHRASE_SENTENCE_PAUSE_MS = int(
    os.getenv("VOICE_VERY_SHORT_PHRASE_SENTENCE_PAUSE_MS", "285")
)

# Extra soft landing for tiny one-thought lines
TINY_PHRASE_WORDS = int(os.getenv("VOICE_TINY_PHRASE_WORDS", "2"))
TINY_PHRASE_SENTENCE_PAUSE_MS = int(os.getenv("VOICE_TINY_PHRASE_SENTENCE_PAUSE_MS", "320"))

# Chunk sizing
MAX_CHUNK_LEN = int(os.getenv("VOICE_MAX_CHUNK_LEN", "320"))
MIN_CHUNK_LEN = int(os.getenv("VOICE_MIN_CHUNK_LEN", "52"))


# -----------------------------------------------------------------------------
# SMALL HELPERS
# -----------------------------------------------------------------------------
def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _clean_text(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""

    s = re.sub(r"\s+", " ", s)
    s = s.replace("—", ", ")
    s = s.replace("–", ", ")
    s = re.sub(r"\s+,", ",", s)
    s = re.sub(r"\s+([.!?])", r"\1", s)
    s = s.replace("!!!", "!").replace("??", "?")
    return s.strip()


def _soft_shape_short_phrase(text: str) -> str:
    """
    Very light shaping for very short lines only.

    Goal:
    - reduce abrupt/clipped endings
    - avoid aggressive punctuation patterns
    - keep behavior predictable and low-risk
    """
    s = _clean_text(text)
    if not s:
        return ""

    wc = _phrase_word_count(s)

    # Only shape truly short phrases.
    if wc > 6:
        return s

    # Normalize repeated punctuation gently.
    s = s.replace("..", ".")
    s = s.replace("!.", "!")
    s = s.replace("?.", "?")

    # If the phrase has no terminal punctuation and is very short,
    # add a period so it lands more naturally in TTS.
    if wc <= 4 and s[-1] not in ".!?":
        s = f"{s}."

    return s


def _rate_to_wpm(rate: float, phrase_words: int | None = None) -> int:
    """
    Convert rate multiplier to a calmer, steadier WPM.

    This version keeps the existing stable mapping, then applies a very small
    reduction for tiny phrases so they feel less sharp and metallic.
    """
    rmin = VOICE_RATE_MIN
    rmax = VOICE_RATE_MAX

    if rmax <= rmin:
        base_wpm = 126
    else:
        r = float(_clamp(rate, rmin, rmax))
        t = (r - rmin) / (rmax - rmin)

        # Slightly softer mapping for more natural cadence.
        gamma = 1.85
        t_curved = t**gamma

        base_wpm = int(WPM_MIN + (WPM_MAX - WPM_MIN) * t_curved)
        base_wpm = int(_clamp(float(base_wpm), 100.0, 182.0))

    if phrase_words is None:
        return base_wpm

    # Tiny phrases can sound extra metallic when pushed too directly.
    # Apply a very small slowdown only there.
    if phrase_words <= TINY_PHRASE_WORDS:
        return max(96, base_wpm - 6)
    if phrase_words <= VERY_SHORT_PHRASE_WORDS:
        return max(98, base_wpm - 3)

    return base_wpm


def _pick_pyttsx3_voice(engine: pyttsx3.Engine, desired: str) -> str:
    if not desired:
        return ""

    want = desired.lower()
    voices = engine.getProperty("voices") or []
    chosen = None

    for v in voices:
        name = (getattr(v, "name", "") or "").lower()
        vid = (getattr(v, "id", "") or "").lower()
        if want in name or want in vid:
            chosen = v
            break

    if chosen is not None:
        try:
            engine.setProperty("voice", getattr(chosen, "id", ""))
        except Exception:
            pass
        return getattr(chosen, "name", "") or getattr(chosen, "id", "") or ""

    return ""


def _init_engine() -> pyttsx3.Engine:
    """
    Fresh engine per synthesis unit.
    No singleton reuse.
    """
    eng = pyttsx3.init()
    _pick_pyttsx3_voice(eng, VOICE_PYTTSX3_VOICE)
    return eng


# -----------------------------------------------------------------------------
# TEXT CHUNKING
# -----------------------------------------------------------------------------
_SENTENCE_END_RE = re.compile(r"(\.\.\.|[.!?])")
_CLAUSE_SPLIT_RE = re.compile(r"(,|;|:)")


def _phrase_word_count(text: str) -> int:
    return len([w for w in text.strip().split(" ") if w])


def _join_small_tail(chunks: List[Tuple[str, int]], min_len: int) -> List[Tuple[str, int]]:
    if not chunks:
        return []

    merged: List[Tuple[str, int]] = []

    for text, pause in chunks:
        text = text.strip()
        if not text:
            continue

        if merged and len(text) < min_len:
            prev_text, _prev_pause = merged[-1]
            merged[-1] = (f"{prev_text} {text}".strip(), pause)
        else:
            merged.append((text, pause))

    return merged


def _split_long_text(text: str, max_len: int, pause_ms: int) -> List[Tuple[str, int]]:
    text = text.strip()
    if not text:
        return []

    out: List[Tuple[str, int]] = []
    remaining = text

    while len(remaining) > max_len:
        cut = remaining.rfind(" ", 0, max_len)
        if cut == -1:
            cut = max_len

        left = remaining[:cut].strip()
        remaining = remaining[cut:].lstrip()

        if left:
            out.append((left, pause_ms))

    if remaining:
        out.append((remaining, 0))

    return out


def _comma_pause_for(text: str, base_pause_ms: int, short_pause_ms: int) -> int:
    wc = _phrase_word_count(text)

    if wc <= VERY_SHORT_PHRASE_WORDS:
        return VERY_SHORT_PHRASE_COMMA_PAUSE_MS
    if wc <= 6:
        return int((short_pause_ms + base_pause_ms) / 2)
    return base_pause_ms


def _sentence_pause_for(text: str, base_pause_ms: int, short_pause_ms: int) -> int:
    wc = _phrase_word_count(text)

    if wc <= TINY_PHRASE_WORDS:
        return TINY_PHRASE_SENTENCE_PAUSE_MS
    if wc <= VERY_SHORT_PHRASE_WORDS:
        return VERY_SHORT_PHRASE_SENTENCE_PAUSE_MS
    if wc <= 7:
        return int((short_pause_ms + base_pause_ms) / 2)
    return base_pause_ms


def chunk_text_with_pauses(
    text: str,
    *,
    max_len: int = MAX_CHUNK_LEN,
    min_len: int = MIN_CHUNK_LEN,
    pause_comma_ms: int,
    pause_sentence_ms: int,
    pause_ellipsis_ms: int,
) -> List[Tuple[str, int]]:
    """
    Speech-friendly chunking with fewer micro-fragments.

    This version keeps medium thought-units together while giving short phrases
    slightly more breathing room so they land less abruptly.
    """
    s = _clean_text(text)
    if not s:
        return []

    pieces: List[Tuple[str, int]] = []
    sentence_parts = _SENTENCE_END_RE.split(s)

    for i in range(0, len(sentence_parts), 2):
        sentence = (sentence_parts[i] or "").strip()
        terminal = sentence_parts[i + 1] if i + 1 < len(sentence_parts) else ""

        if not sentence and not terminal:
            continue

        if len(sentence) <= max_len:
            sentence_chunks: List[Tuple[str, int]] = [(sentence, 0)] if sentence else []
        else:
            clause_parts = _CLAUSE_SPLIT_RE.split(sentence)
            sentence_chunks = []

            for j in range(0, len(clause_parts), 2):
                clause = (clause_parts[j] or "").strip()
                sep = clause_parts[j + 1] if j + 1 < len(clause_parts) else ""

                if not clause:
                    continue

                if len(clause) > max_len:
                    clause_chunks = _split_long_text(
                        clause,
                        max_len=max_len,
                        pause_ms=_comma_pause_for(
                            clause,
                            pause_comma_ms,
                            SHORT_PHRASE_COMMA_PAUSE_MS,
                        ),
                    )
                    sentence_chunks.extend(clause_chunks)
                else:
                    pause = 0
                    if sep in {",", ";", ":"}:
                        pause = _comma_pause_for(
                            clause,
                            pause_comma_ms,
                            SHORT_PHRASE_COMMA_PAUSE_MS,
                        )
                    sentence_chunks.append((clause, pause))

        sentence_chunks = _join_small_tail(sentence_chunks, min_len=min_len)

        if terminal and sentence_chunks:
            final_text, _ = sentence_chunks[-1]

            if terminal == "...":
                end_pause = pause_ellipsis_ms
            else:
                end_pause = _sentence_pause_for(
                    final_text,
                    pause_sentence_ms,
                    SHORT_PHRASE_SENTENCE_PAUSE_MS,
                )

            sentence_chunks[-1] = (final_text, end_pause)

        pieces.extend(sentence_chunks)

    pieces = _join_small_tail(pieces, min_len=min_len)
    return [(t.strip(), p) for t, p in pieces if t.strip()]


# -----------------------------------------------------------------------------
# WAV HELPERS
# -----------------------------------------------------------------------------
def _silence_bytes(duration_ms: int, nchannels: int, sampwidth: int, framerate: int) -> bytes:
    frames = int(framerate * (duration_ms / 1000.0))
    return b"\x00" * frames * nchannels * sampwidth


def _generate_chunk_wav_bytes(
    text: str,
    *,
    rate: float | None,
    volume: float | None,
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Synthesize one chunk into WAV bytes.
    Important:
    - fresh engine per chunk
    - identical rate policy for every chunk
    """
    eng = _init_engine()

    shaped_text = _soft_shape_short_phrase(text)
    phrase_words = _phrase_word_count(shaped_text)

    meta: Dict[str, Any] = {
        "voice_selected": "",
        "rate_wpm": None,
        "volume": None,
        "phrase_words": phrase_words,
    }

    try:
        meta["voice_selected"] = _pick_pyttsx3_voice(eng, VOICE_PYTTSX3_VOICE)
    except Exception:
        meta["voice_selected"] = ""

    try:
        effective_rate = float(rate) if rate is not None else VOICE_DEFAULT_RATE_MUL
    except Exception:
        effective_rate = VOICE_DEFAULT_RATE_MUL

    try:
        wpm = _rate_to_wpm(effective_rate, phrase_words=phrase_words)
        eng.setProperty("rate", wpm)
        meta["rate_wpm"] = wpm
    except Exception:
        meta["rate_wpm"] = None

    try:
        effective_volume = float(volume) if volume is not None else VOICE_VOLUME_DEFAULT
    except Exception:
        effective_volume = VOICE_VOLUME_DEFAULT

    try:
        effective_volume = float(_clamp(effective_volume, 0.0, 1.0))
        eng.setProperty("volume", effective_volume)
        meta["volume"] = effective_volume
    except Exception:
        meta["volume"] = None

    tmp = AUDIO_DIR / f"_chunk_{uuid.uuid4().hex}.wav"

    try:
        eng.save_to_file(shaped_text, str(tmp))
        eng.runAndWait()
        data = tmp.read_bytes()
    finally:
        try:
            eng.stop()
        except Exception:
            pass

        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass

    return data, meta


def _concat_wavs(wavs: List[Tuple[bytes, int]]) -> bytes:
    params = None
    chunks: List[bytes] = []
    pauses: List[int] = []

    for wb, pause in wavs:
        with contextlib.closing(wave.open(io.BytesIO(wb), "rb")) as r:
            nch = r.getnchannels()
            sw = r.getsampwidth()
            fr = r.getframerate()
            frames = r.readframes(r.getnframes())

            if params is None:
                params = (nch, sw, fr)

            chunks.append(frames)
            pauses.append(max(0, int(pause)))

    if params is None:
        return b""

    nch, sw, fr = params
    out_buf = io.BytesIO()

    with contextlib.closing(wave.open(out_buf, "wb")) as w:
        w.setnchannels(nch)
        w.setsampwidth(sw)
        w.setframerate(fr)

        if VOICE_WAV_LEADIN_MS > 0:
            w.writeframes(_silence_bytes(VOICE_WAV_LEADIN_MS, nch, sw, fr))

        for i, frames in enumerate(chunks):
            w.writeframes(frames)
            pause_ms = pauses[i]
            if pause_ms > 0:
                w.writeframes(_silence_bytes(pause_ms, nch, sw, fr))

    return out_buf.getvalue()


# -----------------------------------------------------------------------------
# PUBLIC API
# -----------------------------------------------------------------------------
def synthesize_to_wav(
    text: str,
    *,
    rate: float | None = None,
    volume: float | None = None,
    pause_comma_ms: int | None = None,
    pause_sentence_ms: int | None = None,
    pause_ellipsis_ms: int | None = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Synthesize text with steadier phrasing and return:
    (filename, meta)
    """
    if VOICE_ENGINE != "pyttsx3":
        raise RuntimeError(f"Unsupported engine in services/voice.py: {VOICE_ENGINE}")

    cleaned = _clean_text(text)
    if not cleaned:
        raise RuntimeError("Nothing to synthesize.")

    pc = int(pause_comma_ms) if pause_comma_ms is not None else PAUSE_MS_COMMA_DEFAULT
    ps = int(pause_sentence_ms) if pause_sentence_ms is not None else PAUSE_MS_SENTENCE_DEFAULT
    pe = int(pause_ellipsis_ms) if pause_ellipsis_ms is not None else PAUSE_MS_ELLIPSIS_DEFAULT

    chunks = chunk_text_with_pauses(
        cleaned,
        max_len=MAX_CHUNK_LEN,
        min_len=MIN_CHUNK_LEN,
        pause_comma_ms=pc,
        pause_sentence_ms=ps,
        pause_ellipsis_ms=pe,
    )
    if not chunks:
        raise RuntimeError("Nothing to synthesize.")

    wavs: List[Tuple[bytes, int]] = []
    first_chunk_meta: Dict[str, Any] = {}

    for index, (seg, pause_ms) in enumerate(chunks):
        wb, chunk_meta = _generate_chunk_wav_bytes(seg, rate=rate, volume=volume)
        wavs.append((wb, pause_ms))
        if index == 0:
            first_chunk_meta = dict(chunk_meta)

    merged = _concat_wavs(wavs)
    if not merged:
        raise RuntimeError("Synthesis failed to produce audio.")

    name = f"{uuid.uuid4().hex}.wav"
    out_path = AUDIO_DIR / name
    out_path.write_bytes(merged)

    size_bytes = out_path.stat().st_size if out_path.exists() else 0

    meta: Dict[str, Any] = {
        "voice_request": VOICE_PYTTSX3_VOICE,
        "voice_selected": first_chunk_meta.get("voice_selected", ""),
        "rate_mul": float(rate if rate is not None else VOICE_DEFAULT_RATE_MUL),
        "rate_wpm": (
            int(first_chunk_meta["rate_wpm"]) if first_chunk_meta.get("rate_wpm") is not None else 0
        ),
        "volume": (
            float(first_chunk_meta["volume"])
            if first_chunk_meta.get("volume") is not None
            else float(VOICE_VOLUME_DEFAULT)
        ),
        "leadin_ms": int(VOICE_WAV_LEADIN_MS),
        "bytes": int(size_bytes),
        "chunks": int(len(chunks)),
        "phrase_words": int(first_chunk_meta.get("phrase_words", 0)),
        "wpm_bounds": {"min": int(WPM_MIN), "max": int(WPM_MAX)},
    }

    return name, meta


def speak_to_wav(
    text: str,
    *,
    rate: float | None = None,
    volume: float | None = None,
    pause_comma_ms: int | None = None,
    pause_sentence_ms: int | None = None,
    pause_ellipsis_ms: int | None = None,
) -> str:
    """
    Backward-compatible wrapper returning filename only.
    """
    name, _meta = synthesize_to_wav(
        text,
        rate=rate,
        volume=volume,
        pause_comma_ms=pause_comma_ms,
        pause_sentence_ms=pause_sentence_ms,
        pause_ellipsis_ms=pause_ellipsis_ms,
    )
    return name


def get_audio_dir() -> Path:
    return AUDIO_DIR


def get_audio_path(name: str) -> Path:
    return AUDIO_DIR / name
