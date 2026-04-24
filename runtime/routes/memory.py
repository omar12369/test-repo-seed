# runtime/routes/memory.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from runtime.services.memory_service import MemoryStore

router = APIRouter(prefix="/v1/memory", tags=["memory"])

log = logging.getLogger("seed.memory")
DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "memory.json"
store = MemoryStore(file_path=DATA_FILE, logger=log)

# ---------- helpers ----------


def _norm(v: Any) -> str:
    return str(v if v is not None else "").strip()


def _all_items() -> List[Dict[str, Any]]:
    # Your store returns all when n <= 0
    return store.last(n=0)


def _resolve_id(maybe_id: str) -> Optional[str]:
    """
    Return the canonical id from the file that matches maybe_id (after normalization),
    or None if not found. This scan avoids any strict-compare edge cases.
    """
    target = _norm(maybe_id)
    for it in _all_items():
        if _norm(it.get("id")) == target:
            return it.get("id")
    return None


def _get_item_by_any_id(maybe_id: str) -> Optional[Dict[str, Any]]:
    real = _resolve_id(maybe_id)
    if real is None:
        return None
    # Try direct first (fast path), then scan fallback
    item = store.get_by_id(real)
    if item:
        return item
    for it in _all_items():
        if _norm(it.get("id")) == _norm(real):
            return it
    return None


# ---------- schemas ----------


class MemoryAppendIn(BaseModel):
    text: str = Field(..., description="content")
    role: str = Field(default="user")
    extras: Optional[Dict[str, Any]] = None


class MemoryUpdateIn(BaseModel):
    text: Optional[str] = None
    role: Optional[str] = None
    extras: Optional[Dict[str, Any]] = None


# ---------- endpoints ----------


@router.post("/append", response_model=Dict[str, Any], status_code=201)
def memory_append(payload: MemoryAppendIn):
    return store.append(text=payload.text, role=payload.role, extras=payload.extras)


@router.get("/search", response_model=List[Dict[str, Any]])
def memory_search(query: str = Query(..., description="plain/field query"), limit: int = 20):
    return store.search(query=query, limit=limit)


@router.get("/last", response_model=List[Dict[str, Any]])
def memory_last(n: int = 10):
    return store.last(n=n)


@router.get("/{id}", response_model=Dict[str, Any])
def memory_get(id: str):
    item = _get_item_by_any_id(id)
    if not item:
        raise HTTPException(status_code=404, detail="memory not found")
    return item


@router.put("/{id}", response_model=Dict[str, Any])
def memory_update(id: str, payload: MemoryUpdateIn):
    real = _resolve_id(id)
    if real is None:
        raise HTTPException(status_code=404, detail="memory not found")
    updated = store.update_by_id(
        mem_id=real, text=payload.text, role=payload.role, extras=payload.extras
    )
    if not updated:
        # Extremely defensive fallback
        item = _get_item_by_any_id(real)
        if not item:
            raise HTTPException(status_code=404, detail="memory not found")
        updated = {**item}
        if payload.text is not None:
            updated["text"] = payload.text
        if payload.role is not None:
            updated["role"] = payload.role
        if payload.extras:
            updated.update(payload.extras)
        # Push the patched item list back
        all_items = _all_items()
        for i, it in enumerate(all_items):
            if _norm(it.get("id")) == _norm(real):
                all_items[i] = updated
                break
        # Write via store's private I/O (safe since the service already uses it)
        store._write_all(all_items)  # noqa: SLF001
    return updated


@router.delete("/{id}", status_code=204)
def memory_delete(id: str):
    real = _resolve_id(id)
    if real is None:
        raise HTTPException(status_code=404, detail="memory not found")
    ok = store.delete_by_id(mem_id=real)
    if not ok:
        # Fallback: manually filter and write
        all_items = _all_items()
        new_items = [it for it in all_items if _norm(it.get("id")) != _norm(real)]
        if len(new_items) == len(all_items):
            raise HTTPException(status_code=404, detail="memory not found")
        store._write_all(new_items)  # noqa: SLF001
    return


# ---------- debug ----------


@router.get("/_debug/ids")
def memory_debug_ids():
    items = _all_items()
    return {"file": str(DATA_FILE), "count": len(items), "ids": [it.get("id") for it in items]}
