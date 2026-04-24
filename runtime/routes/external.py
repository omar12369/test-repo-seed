# runtime/routes/external.py
from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from runtime.services.web_client import UnsafeURLError, http_get_json

router = APIRouter(prefix="/v1", tags=["external"])


@router.get("/fetch")
async def fetch(
    request: Request,
    url: str = Query(..., description="Public http/https URL to fetch"),
    remember: bool = Query(False, description="If true, save a short summary to memory"),
    label: Optional[str] = Query(None, description="Optional note to include in memory entry"),
):
    """
    Fetch JSON (or text) from a public URL with simple SSRF protections.
    If 'remember' is true, store a compact summary in memory.

    Example:
      /v1/fetch?url=https://api.github.com
      /v1/fetch?url=https://example.com&remember=true&label=first+test
    """
    try:
        result = await http_get_json(url)
    except UnsafeURLError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # If remote server errors or timeouts occur, surface a clean message
        raise HTTPException(status_code=502, detail=f"Upstream error: {e}")

    saved = False
    memory_id = None

    if remember:
        # Try to summarize result into a short, readable preview
        kind = "json" if "data" in result else "text" if "text" in result else "unknown"
        if kind == "json":
            try:
                preview = json.dumps(result["data"], ensure_ascii=False)[:600]
            except Exception:
                preview = "<unserializable JSON>"
        elif kind == "text":
            preview = (result.get("text") or "")[:600]
        else:
            preview = "<no content>"

        # Build the memory note
        note_lines = [
            "[FETCH]",
            f"url: {url}",
            f"status: {result.get('status')}",
            f"kind: {kind}",
        ]
        if label:
            note_lines.append(f"label: {label}")
        note_lines.append("preview:")
        note_lines.append(preview)

        entry_text = "\n".join(note_lines)

        # Save to memory if the store is present
        store = getattr(request.app.state, "memory_store", None)
        if store is not None:
            entry = store.append(text=entry_text, role="system")
            saved = True
            memory_id = entry.get("id")

    # Return normal fetch result + metadata about memory save (if any)
    return {
        **result,
        "remembered": saved,
        "memory_id": memory_id,
    }
