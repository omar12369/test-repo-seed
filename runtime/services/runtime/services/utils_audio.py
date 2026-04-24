# runtime/services/utils_audio.py
from __future__ import annotations

from pathlib import Path


def polish_audio(in_path: Path) -> None:
    """
    Optional polish: normalize loudness + tiny fades.
    - No-op if pydub/ffmpeg is missing.
    - Overwrites the input file in-place.
    """
    try:
        from pydub import AudioSegment
        from pydub.effects import normalize
    except Exception:
        # pydub or its backends not available -> skip silently
        return

    in_path = Path(in_path)
    fmt = in_path.suffix.lstrip(".").lower()  # 'wav', 'mp3', 'm4a', ...
    try:
        audio = AudioSegment.from_file(in_path)
        audio = normalize(audio)
        audio = audio.fade_in(30).fade_out(50)
        audio.export(in_path, format=fmt)
    except Exception:
        # If anything fails (e.g., mp3 without ffmpeg), just leave original
        pass
