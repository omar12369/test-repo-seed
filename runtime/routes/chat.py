# runtime/routes/chat.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/v1", tags=["chat"])


# --------- models ---------


class ChatIn(BaseModel):
    user_id: str = Field(..., description="Who is speaking")
    message: str = Field(..., description="User's message")
    labels: Optional[Dict[str, Any]] = None  # optional tags


class ChatOut(BaseModel):
    message_id: str
    summary: Optional[str] = None
    saved_memory_id: str
    meta: Dict[str, Any]


class ChatRecord(BaseModel):
    id: str
    user_id: str
    role: str
    text: str
    summary: Optional[str] = None
    labels: Optional[Dict[str, Any]] = None
    ts: str


# --------- routes ---------


@router.post("/chat/create_chat", response_model=ChatOut)
async def create_chat(request: Request, payload: ChatIn) -> ChatOut:
    """
    Save the user's message to memory (meta.type='chat'), generate a simple
    echo reply for now, and save that reply to memory as well.
    """
    ms = getattr(request.app.state, "memory_store", None)
    if ms is None:
        raise HTTPException(status_code=500, detail="MemoryStore unavailable")

    user_meta = {"type": "chat", "user_id": payload.user_id}
    labels = payload.labels or {}

    # 1) Save user message
    user_entry = ms.append(
        text=payload.message,
        role="user",
        extras={
            "summary": None,
            "labels": labels,
            "meta": user_meta,
        },
    )

    # 2) Naive assistant reply
    reply_text = f"Echo: {payload.message}"
    asst_entry = ms.append(
        text=reply_text,
        role="assistant",
        extras={
            "summary": None,
            "labels": {"source": "system"},
            "meta": {"type": "chat", "user_id": payload.user_id},
        },
    )

    return ChatOut(
        message_id=user_entry["id"],
        summary=None,
        saved_memory_id=asst_entry["id"],
        meta={
            "user_saved_id": user_entry["id"],
            "assistant_saved_id": asst_entry["id"],
        },
    )


@router.get("/chat/last", response_model=List[ChatRecord])
async def get_last_chats(
    request: Request,
    n: int = Query(20, ge=1, le=200),
) -> List[ChatRecord]:
    """
    Return last N chat messages (both user & assistant) from memory.json,
    filtered where meta.type == 'chat'. Newest-first.
    """
    ms = getattr(request.app.state, "memory_store", None)
    if ms is None:
        raise HTTPException(status_code=500, detail="MemoryStore unavailable")

    raw = ms.last(1000)  # read a window, then filter
    out: List[ChatRecord] = []

    for item in reversed(raw):  # newest-first
        if not isinstance(item, dict):
            continue
        meta = item.get("meta") or {}
        if meta.get("type") != "chat":
            continue

        out.append(
            ChatRecord(
                id=item.get("id", ""),
                user_id=meta.get("user_id", ""),
                role=item.get("role", ""),
                text=item.get("text", ""),
                summary=item.get("summary"),
                labels=item.get("labels"),
                ts=item.get("ts", ""),
            )
        )
        if len(out) >= n:
            break

    return out
