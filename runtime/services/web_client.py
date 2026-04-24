from __future__ import annotations

import os

from fastapi import APIRouter
from fastapi.staticfiles import StaticFiles

web_router = APIRouter(tags=["web"])


def mount_web(app) -> None:
    """
    Mount the web client directory at /web.

    Your runtime/main.py already mounts /web directly. This helper exists
    so older code that imports web_client.py doesn't break.
    """
    web_dir = os.getenv("WEB_DIR", "web_client")
    if os.path.isdir(web_dir):
        app.mount("/web", StaticFiles(directory=web_dir, html=True), name="web")
