from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# -----------------------------------------------------------------------------
# PATHS / CONFIG
# -----------------------------------------------------------------------------
FEEDBACK_DIR = Path(os.getenv("FEEDBACK_DIR", "data/feedback")).resolve()
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

FEEDBACK_LOG_PATH = Path(
    os.getenv("FEEDBACK_LOG_PATH", str(FEEDBACK_DIR / "interaction_feedback.jsonl"))
).resolve()

# Create parent directory if a custom FEEDBACK_LOG_PATH is supplied
FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

DEFAULT_RECENT_LIMIT = int(os.getenv("FEEDBACK_RECENT_LIMIT", "50"))


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        return str(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "on"}:
            return True
        if v in {"false", "0", "no", "off"}:
            return False
    try:
        return bool(value)
    except Exception:
        return default


def _safe_list_of_str(value: Any) -> List[str]:
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


def _safe_optional_str(value: Any) -> Optional[str]:
    s = _safe_str(value).strip()
    return s if s else None


# -----------------------------------------------------------------------------
# NORMALIZATION
# -----------------------------------------------------------------------------
def build_feedback_record(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a raw feedback payload into a stable append-only record.
    """
    record: Dict[str, Any] = {
        "id": _safe_str(payload.get("id")).strip() or uuid.uuid4().hex,
        "ts": _safe_int(payload.get("ts"), 0),
        "session_id": _safe_optional_str(payload.get("session_id")),
        "user_input": _safe_str(payload.get("user_input")).strip(),
        "system_output": _safe_str(payload.get("system_output")).strip(),
        "channel": _safe_str(payload.get("channel")).strip() or "unknown",
        "input_mode": _safe_optional_str(payload.get("input_mode")),
        "output_mode": _safe_optional_str(payload.get("output_mode")),
        "engine": _safe_str(payload.get("engine")).strip() or "unknown",
        "profile": _safe_optional_str(payload.get("profile")),
        "success": _safe_bool(payload.get("success"), False),
        "latency_ms": (
            _safe_int(payload.get("latency_ms"), 0)
            if payload.get("latency_ms") is not None
            else None
        ),
        "playback_ok": (
            _safe_bool(payload.get("playback_ok"), False)
            if payload.get("playback_ok") is not None
            else None
        ),
        "user_feedback": _safe_optional_str(payload.get("user_feedback")),
        "feedback_tags": _safe_list_of_str(payload.get("feedback_tags")),
        "notes": _safe_optional_str(payload.get("notes")),
    }

    return record


def validate_feedback_record(record: Dict[str, Any]) -> List[str]:
    """
    Return a list of validation errors. Empty list = valid enough to store.
    """
    errors: List[str] = []

    if not _safe_str(record.get("id")).strip():
        errors.append("id is required")

    if _safe_int(record.get("ts"), 0) <= 0:
        errors.append("ts must be a positive integer timestamp")

    if not _safe_str(record.get("user_input")).strip():
        errors.append("user_input is required")

    if not _safe_str(record.get("system_output")).strip():
        errors.append("system_output is required")

    if not _safe_str(record.get("channel")).strip():
        errors.append("channel is required")

    if not _safe_str(record.get("engine")).strip():
        errors.append("engine is required")

    if "success" not in record:
        errors.append("success is required")

    return errors


# -----------------------------------------------------------------------------
# STORAGE
# -----------------------------------------------------------------------------
def append_feedback_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Append one normalized record to the JSONL log.
    """
    line = json.dumps(record, ensure_ascii=False)

    with FEEDBACK_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

    return record


def log_feedback(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize, validate, and append one feedback record.
    """
    record = build_feedback_record(payload)
    errors = validate_feedback_record(record)
    if errors:
        raise ValueError("; ".join(errors))

    return append_feedback_record(record)


def read_recent_feedback(limit: int = DEFAULT_RECENT_LIMIT) -> List[Dict[str, Any]]:
    """
    Read the most recent feedback entries from the JSONL log.
    Returns newest first.
    """
    if limit <= 0:
        return []

    if not FEEDBACK_LOG_PATH.exists():
        return []

    try:
        lines = FEEDBACK_LOG_PATH.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

    recent_lines = lines[-limit:]
    items: List[Dict[str, Any]] = []

    for line in reversed(recent_lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                items.append(obj)
        except Exception:
            continue

    return items


def feedback_storage_status() -> Dict[str, Any]:
    exists = FEEDBACK_LOG_PATH.exists()
    size_bytes = FEEDBACK_LOG_PATH.stat().st_size if exists else 0

    return {
        "feedback_dir": str(FEEDBACK_DIR),
        "feedback_log_path": str(FEEDBACK_LOG_PATH),
        "exists": exists,
        "bytes": size_bytes,
        "default_recent_limit": DEFAULT_RECENT_LIMIT,
    }
