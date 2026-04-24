from __future__ import annotations

import json
import shlex
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------- helpers ------------------------------------------------------------


def _now_ts() -> str:
    """UTC timestamp with trailing Z (ISO-ish)."""
    return datetime.utcnow().isoformat() + "Z"


def _norm_id(mem_id: str) -> str:
    """Normalize ids for comparison."""
    return (mem_id or "").strip()


def _to_lc_str(v: Any) -> str:
    """Safe lowercased string for comparisons."""
    return str(v if v is not None else "").lower()


# ---------- store --------------------------------------------------------------


class MemoryStore:
    """
    JSON-backed memory store with UUID ids.
    Creates `<project_root>/data/memory.json` if missing.
    Thread-safe for local usage.
    """

    def __init__(self, file_path: Path, logger=None):
        self.file_path = Path(file_path)
        self.logger = logger
        self.lock = threading.Lock()
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self._write_all([])

    # -- internal I/O -----------------------------------------------------------

    def _read_all(self) -> List[Dict[str, Any]]:
        try:
            with self.file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            if self.logger:
                self.logger.warning("memory.json is corrupt; resetting to []")
            self._write_all([])
            return []
        except FileNotFoundError:
            self._write_all([])
            return []

    def _write_all(self, items: List[Dict[str, Any]]) -> None:
        tmp = self.file_path.with_suffix(self.file_path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        tmp.replace(self.file_path)

    # -- public API -------------------------------------------------------------

    def append(
        self,
        text: str,
        role: str = "user",
        extras: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Add a new memory entry and return it.
        Always includes: id, ts, role, text.
        You may pass `extras` to attach fields like summary/labels/meta/etc.
        """
        entry: Dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "ts": _now_ts(),
            "role": role,
            "text": text,
        }
        if extras:
            entry.update(extras)

        with self.lock:
            items = self._read_all()
            items.append(entry)
            self._write_all(items)

        if self.logger:
            self.logger.info("Memory appended: id=%s role=%s len=%d", entry["id"], role, len(text))
        return entry

    def last(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return last n entries (all if n<=0)."""
        with self.lock:
            items = self._read_all()
            return items[-n:] if n > 0 else items

    # ---------- searching ------------------------------------------------------

    def _parse_query(self, q: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """
        Parse a simple query language:

          - Plain terms:  feature  ui
          - Field terms:  labels.phase:3   phase:3   topic:ui
                          meta.source:swagger  source:swagger

        Field terms are ANDed together with plain terms.
        Values may be quoted: topic:"quick actions"
        Returns (plain_terms, field_filters)
          field_filters is a list of (namespace, key, value)
            namespace in {"labels","meta","*"} where "*" tries labels then meta
        """
        terms: List[str] = []
        filters: List[Tuple[str, str, str]] = []

        # Use shlex so quoted strings work
        for tok in shlex.split(q or ""):
            if ":" in tok:
                left, value = tok.split(":", 1)
                ns = "*"
                key = left
                if "." in left:
                    ns, key = left.split(".", 1)
                    ns = ns.lower().strip()
                    if ns not in {"labels", "meta"}:
                        ns = "*"
                filters.append((ns, key.strip().lower(), value.strip().lower()))
            else:
                terms.append(tok.strip().lower())
        return terms, filters

    def _match_filters(
        self, item: Dict[str, Any], terms: List[str], filters: List[Tuple[str, str, str]]
    ) -> bool:
        txt = _to_lc_str(item.get("text", ""))
        labels = item.get("labels") or {}
        meta = item.get("meta") or {}

        # All plain terms must be in the text
        for t in terms:
            if t and t not in txt:
                return False

        # All field filters must match
        for ns, key, val in filters:
            matched = False
            if ns in {"labels", "*"}:
                if key in labels and val == _to_lc_str(labels.get(key)):
                    matched = True
            if not matched and ns in {"meta", "*"}:
                if key in meta and val == _to_lc_str(meta.get(key)):
                    matched = True
            if not matched:
                return False

        return True

    def search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search over text AND/OR labels/meta using the simple language above.
        Results are newest-first, up to `limit`.
        """
        q = (query or "").strip()
        if not q:
            return []

        terms, filters = self._parse_query(q)

        with self.lock:
            items = self._read_all()

        out: List[Dict[str, Any]] = []
        for it in reversed(items):  # newest-first
            if not isinstance(it, dict):
                continue
            if self._match_filters(it, terms, filters):
                out.append(it)
                if len(out) >= max(1, limit):
                    break
        return out

    # ---------- id operations --------------------------------------------------

    def get_by_id(self, mem_id: str) -> Optional[Dict[str, Any]]:
        mid = _norm_id(str(mem_id))
        if not mid:
            return None
        with self.lock:
            for it in self._read_all():
                if _norm_id(str(it.get("id", ""))) == mid:
                    return it
        return None

    def update_by_id(
        self,
        mem_id: str,
        *,
        text: Optional[str] = None,
        role: Optional[str] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        mid = _norm_id(str(mem_id))
        if not mid:
            return None

        with self.lock:
            items = self._read_all()
            for i, it in enumerate(items):
                if _norm_id(str(it.get("id", ""))) == mid:
                    if text is not None:
                        it["text"] = text
                    if role is not None:
                        it["role"] = role
                    if extras:
                        it.update(extras)
                    items[i] = it
                    self._write_all(items)
                    if self.logger:
                        self.logger.info("Memory updated: id=%s", mid)
                    return it
        return None

    def delete_by_id(self, mem_id: str) -> bool:
        mid = _norm_id(str(mem_id))
        if not mid:
            return False
        with self.lock:
            items = self._read_all()
            new_items = [it for it in items if _norm_id(str(it.get("id", ""))) != mid]
            if len(new_items) == len(items):
                return False
            self._write_all(new_items)
            if self.logger:
                self.logger.info("Memory deleted: id=%s", mid)
            return True
