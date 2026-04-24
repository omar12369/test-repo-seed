# runtime/routes/voice_realtime_chat.py
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

# We call the existing Phase-3 voice endpoint code directly (NOT speak_to_wav)
from runtime.routes.voice import SpeakIn, SpeakOut
from runtime.routes.voice import speak as voice_speak
from runtime.services.chat_service import reply_text as chat_reply_text
from runtime.services.text_preprocessor import clean_text

chat_router = APIRouter(prefix="/v1/chat", tags=["chat"])


# ----------------------------
# Models
# ----------------------------
class ChatTextIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class ChatTextOut(BaseModel):
    ok: bool
    reply: str


class ChatVoiceIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    profile: Optional[str] = Field(None, description="Voice profile (e.g., auren_balanced)")


class ChatVoiceOut(BaseModel):
    ok: bool
    reply: str
    file: str
    url: str
    abs_url: str
    engine: str
    mime: str = "audio/wav"


# ----------------------------
# Routes
# ----------------------------
@chat_router.post("/text", response_model=ChatTextOut)
def chat_text(body: ChatTextIn) -> ChatTextOut:
    user_text = clean_text(body.text or "").strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Empty prompt after cleaning.")

    reply = clean_text(chat_reply_text(user_text)).strip() or "(silence)"
    return ChatTextOut(ok=True, reply=reply)


@chat_router.post("/voice", response_model=ChatVoiceOut)
def chat_voice(body: ChatVoiceIn, request: Request) -> ChatVoiceOut:
    user_text = clean_text(body.text or "").strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Empty prompt after cleaning.")

    reply = clean_text(chat_reply_text(user_text)).strip() or "(silence)"

    # Reuse Phase-3 voice pipeline exactly (generates WAV + returns URLs/meta)
    out: SpeakOut = voice_speak(SpeakIn(text=reply, profile=body.profile), request=request)

    return ChatVoiceOut(
        ok=True,
        reply=reply,
        file=out.name,
        url=out.url,
        abs_url=out.abs_url,
        engine=out.engine,
        mime="audio/wav",
    )
