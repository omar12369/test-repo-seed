# runtime/services/voice_history.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

HISTORY_SIZE = int(os.getenv("VOICE_HISTORY_SIZE", "200"))
HISTORY_FILE = Path(os.getenv("VOICE_HISTORY_LOG", "voice_history.json")).resolve()


def _load() -> List[Dict[str, Any]]:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text("utf-8"))
        except Exception:
            return []
    return []


def _save(items: List[Dict[str, Any]]) -> None:
    try:
        HISTORY_FILE.write_text(
            json.dumps(items[-HISTORY_SIZE:], ensure_ascii=False, indent=2), "utf-8"
        )
    except Exception:
        pass


def append_history(item: Dict[str, Any]) -> None:
    items = _load()
    items.append(item)
    _save(items)


def get_history() -> Dict[str, Any]:
    items = _load()
    return {"size": len(items), "items": items[-HISTORY_SIZE:]}
