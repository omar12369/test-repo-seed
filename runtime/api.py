from __future__ import annotations

from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="SEED API", version="0.1.0")


# --------- Basic health / ping ----------


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.get("/v1/ping")
def ping(msg: str = "seed") -> dict:
    return {"pong": msg}


# --------- Example typed POST endpoint ----------


class MessageIn(BaseModel):
    text: str = Field(..., min_length=1)


class MessageOut(BaseModel):
    received_text: str
    length: int
    timestamp: str


@app.post("/v1/message", response_model=MessageOut)
def message(body: MessageIn) -> MessageOut:
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    return MessageOut(
        received_text=body.text,
        length=len(body.text),
        timestamp=now,
    )
