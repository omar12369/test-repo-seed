from __future__ import annotations

import wave
from pathlib import Path


def prepend_silence_wav(path: str | Path, ms: int = 200) -> None:
    """
    Prepends a short silence segment to a PCM WAV file.
    This prevents browsers/audio stacks from clipping the first syllables.
    Only safely supports 16-bit PCM WAV (sampwidth=2).
    """
    p = Path(path)

    with wave.open(str(p), "rb") as r:
        params = r.getparams()
        frames = r.readframes(r.getnframes())

    # Only handle 16-bit PCM cleanly
    if params.sampwidth != 2:
        return

    framerate = params.framerate
    nchannels = params.nchannels

    silence_frames = int(framerate * (ms / 1000.0))
    silence = (b"\x00\x00" * nchannels) * silence_frames

    with wave.open(str(p), "wb") as w:
        w.setparams(params)
        w.writeframes(silence + frames)
