# runtime/services/audio_maintenance.py
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, List, Tuple


def _iter_audio(dirpath: Path) -> Iterable[Tuple[Path, float]]:
    for p in dirpath.glob("*"):
        if p.is_file() and p.suffix.lower() in {".wav", ".mp3", ".m4a", ".ogg"}:
            try:
                yield p, p.stat().st_mtime
            except Exception:
                continue


def cleanup_audio(
    dirpath: Path,
    max_files: int | None = 300,
    max_age_hours: int | None = 72,
) -> List[str]:
    """
    Deletes oldest audio files by age and/or count.
    Returns list of deleted filenames.
    """
    deleted: List[str] = []
    now = time.time()
    files = list(_iter_audio(dirpath))
    files.sort(key=lambda t: t[1], reverse=True)  # newest first

    # Age-based prune
    if max_age_hours and max_age_hours > 0:
        cutoff = now - (max_age_hours * 3600)
        for p, m in list(files):
            if m < cutoff:
                try:
                    p.unlink(missing_ok=True)
                    deleted.append(p.name)
                except Exception:
                    pass
        # refresh list after deletions
        files = [(p, m) for (p, m) in files if p.exists()]

    # Count-based prune
    if max_files and max_files > 0 and len(files) > max_files:
        for p, _ in files[max_files:]:
            try:
                p.unlink(missing_ok=True)
                deleted.append(p.name)
            except Exception:
                pass

    return deleted
