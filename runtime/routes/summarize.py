# runtime/routes/summarize.py
# Simple HTTP endpoint (extractive) that uses summarize_text in services.

from fastapi import APIRouter
from pydantic import BaseModel, Field

from runtime.services.summarizer_client import summarize_text  # <— ABSOLUTE import

router = APIRouter(prefix="/v1", tags=["summarize"])


class SummarizeRequest(BaseModel):
    text: str = Field(..., description="Plain text to summarize")
    max_sentences: int = Field(3, ge=1, le=21, description="How many sentences to keep?")
    max_len: int = Field(
        2000, ge=200, le=100000, description="Safety cap on input length before summarizing"
    )


@router.post("/summarize")
def summarize(req: SummarizeRequest) -> dict:
    summary = summarize_text(
        req.text,
        max_sentences=req.max_sentences,
        max_len=req.max_len,
    )
    return {"summary": summary}


__all__ = ["router"]
