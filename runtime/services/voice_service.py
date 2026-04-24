from __future__ import annotations

import json
import re
import struct
import uuid
import wave
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Optional TTS engine. We run even if it's missing.
try:
    import pyttsx3  # offline, cross-platform
except Exception:  # pragma: no cover
    pyttsx3 = None  # type: ignore


SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _iso_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _sanitize_filename(name: str) -> str:
    # keep .wav/.json and simple chars; strip spaces/odd chars
    name = name.strip()
    name = SAFE_NAME_RE.sub("_", name)
    return name


@dataclass
class VoiceDefaults:
    voice: Optional[str] = None  # substring of installed voice name
    rate: Optional[int] = None  # engine default if None
    volume: Optional[float] = None  # 0.0..1.0


@dataclass
class AudioMeta:
    id: str
    ts: str
    text_len: int
    engine: str
    voice: Optional[str]
    rate: Optional[int]
    volume: Optional[float]
    path: str


class VoiceService:
    """
    Phase-3 voice service with:
      • Real TTS via pyttsx3 (if available), else silent WAV fallback
      • Voice listing and substring selection
      • Per-request overrides + service defaults
      • Safe path handling + metadata sidecar (.json)
      • Housekeeping: delete/prune
    """

    def __init__(
        self,
        logger=None,
        audio_root: str | Path = "data/audio",
        defaults: VoiceDefaults | None = None,
    ) -> None:
        self.logger = logger
        self.audio_root = Path(audio_root)
        self.audio_root.mkdir(parents=True, exist_ok=True)

        self.defaults = defaults or VoiceDefaults()
        self.engine = None
        self.engine_name = "none"

        if pyttsx3 is not None:
            try:
                self.engine = pyttsx3.init()
                self.engine_name = "pyttsx3"
                # Seed engine defaults if provided
                if self.defaults.rate is not None:
                    try:
                        self.engine.setProperty("rate", int(self.defaults.rate))
                    except Exception:
                        pass
                if self.defaults.volume is not None:
                    try:
                        self.engine.setProperty("volume", float(self.defaults.volume))
                    except Exception:
                        pass
                if self.defaults.voice:
                    self._apply_voice_substring(self.defaults.voice)
            except Exception as e:  # pragma: no cover
                self.engine = None
                self.engine_name = "none"
                if self.logger:
                    self.logger.warning("pyttsx3 init failed: %s (silent WAV fallback)", e)

    # ---------------------------------------------------------------------
    # Introspection
    # ---------------------------------------------------------------------
    def status(self) -> dict:
        return {
            "ok": True,
            "engine": self.engine_name,
            "defaults": asdict(self.defaults),
            "audio_dir": str(self.audio_root.resolve()),
        }

    def list_voices(self, *, contains: Optional[str] = None) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if self.engine is None:
            return out
        try:
            q = (contains or "").lower()
            for v in self.engine.getProperty("voices"):
                name = getattr(v, "name", "") or ""
                if q and q not in name.lower():
                    continue
                out.append(
                    {
                        "id": getattr(v, "id", ""),
                        "name": name,
                        "languages": getattr(v, "languages", None),
                        "gender": getattr(v, "gender", None),
                        "age": getattr(v, "age", None),
                    }
                )
        except Exception:
            pass
        return out

    # ---------------------------------------------------------------------
    # Core synthesis
    # ---------------------------------------------------------------------
    def speak_to_file(
        self,
        *,
        text: str,
        voice: Optional[str] = None,
        rate: Optional[int] = None,
        volume: Optional[float] = None,
        filename: Optional[str] = None,
    ) -> str:
        """
        Generate audio and save as a WAV file; returns absolute file path (string).
        Also writes a JSON sidecar with basic metadata.
        """
        wav_path = self._next_wav(filename)
        meta = AudioMeta(
            id=uuid.uuid4().hex,
            ts=_iso_now(),
            text_len=len(text or ""),
            engine=self.engine_name,
            voice=voice or self.defaults.voice,
            rate=rate if rate is not None else self.defaults.rate,
            volume=volume if volume is not None else self.defaults.volume,
            path=str(wav_path),
        )

        if self.engine is None:
            self._write_silent_wav(
                wav_path, sample_rate=16000, seconds=max(1.0, min(5.0, len(text) / 12.0))
            )
            self._write_meta(wav_path, meta)
            if self.logger:
                self.logger.info("silent WAV written: %s", wav_path.name)
            return str(wav_path)

        # Apply per-request engine settings safely
        try:
            if voice:
                self._apply_voice_substring(voice)
            if rate is not None:
                try:
                    self.engine.setProperty("rate", int(rate))
                except Exception:
                    pass
            if volume is not None:
                try:
                    self.engine.setProperty("volume", max(0.0, min(1.0, float(volume))))
                except Exception:
                    pass

            self.engine.save_to_file(text, str(wav_path))
            self.engine.runAndWait()
        finally:
            # Re-apply defaults where provided (keeps engine stable across requests)
            try:
                if self.defaults.rate is not None:
                    self.engine.setProperty("rate", int(self.defaults.rate))
                if self.defaults.volume is not None:
                    self.engine.setProperty("volume", float(self.defaults.volume))
                if self.defaults.voice:
                    self._apply_voice_substring(self.defaults.voice)
            except Exception:
                pass

        self._write_meta(wav_path, meta)
        if self.logger:
            self.logger.info("tts generated: %s", wav_path.name)
        return str(wav_path)

    # ---------------------------------------------------------------------
    # Housekeeping
    # ---------------------------------------------------------------------
    def delete(self, path: str | Path) -> bool:
        p = self._resolve_under_root(path)
        ok = False
        if p.exists():
            try:
                p.unlink()
                ok = True
            except Exception:
                ok = False
        # remove sidecar too
        m = p.with_suffix(".json")
        if m.exists():
            try:
                m.unlink()
            except Exception:
                pass
        return ok

    def prune(
        self, *, max_files: Optional[int] = None, max_age_minutes: Optional[int] = None
    ) -> dict:
        """
        Delete oldest audio files to keep directory small and/or recent.
        """
        files = sorted(
            (f for f in self.audio_root.glob("*.wav") if f.is_file()),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        removed: List[str] = []
        keep: List[Path] = list(files)

        if max_files is not None and max_files >= 0 and len(keep) > max_files:
            for f in keep[max_files:]:
                if self.delete(f):
                    removed.append(f.name)
            keep = keep[:max_files]

        if max_age_minutes is not None and max_age_minutes >= 0:
            cutoff = datetime.utcnow() - timedelta(minutes=max_age_minutes)
            survivors: List[Path] = []
            for f in keep:
                ts = datetime.utcfromtimestamp(int(f.stat().st_mtime))
                if ts < cutoff:
                    if self.delete(f):
                        removed.append(f.name)
                else:
                    survivors.append(f)
            keep = survivors

        return {"removed": removed, "kept": [f.name for f in keep]}

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------
    def _apply_voice_substring(self, needle: str) -> None:
        if self.engine is None:
            return
        n = (needle or "").lower()
        try:
            for v in self.engine.getProperty("voices"):
                name = (getattr(v, "name", "") or "").lower()
                if n in name:
                    self.engine.setProperty("voice", v.id)
                    return
        except Exception:
            pass

    def _next_wav(self, filename: Optional[str]) -> Path:
        if filename:
            filename = _sanitize_filename(filename)
            if not filename.lower().endswith(".wav"):
                filename += ".wav"
            p = self.audio_root / filename
        else:
            p = self.audio_root / f"{uuid.uuid4().hex}.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p.resolve()

    def _resolve_under_root(self, path: str | Path) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = (self.audio_root / p).resolve()
        else:
            p = p.resolve()
        root = self.audio_root.resolve()
        if root not in p.parents and p != root:
            # deny escape
            raise ValueError("path must be under audio_root")
        return p

    def _write_meta(self, wav_path: Path, meta: AudioMeta) -> None:
        sidecar = wav_path.with_suffix(".json")
        data = asdict(meta)
        try:
            with sidecar.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _write_silent_wav(
        self, path: Path, *, sample_rate: int = 16000, seconds: float = 1.0
    ) -> None:
        frames = int(sample_rate * max(0.1, float(seconds)))
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)  # mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            zero = struct.pack("<h", 0)  # 16-bit little-endian zero
            for _ in range(frames):
                wf.writeframes(zero)
