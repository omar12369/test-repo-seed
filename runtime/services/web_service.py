# runtime/services/web_service.py
from __future__ import annotations

import ipaddress
import os
import re
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx

# -----------------------------
# Config (safe defaults)
# -----------------------------
DEFAULT_TIMEOUT = 10.0  # seconds
MAX_PREVIEW_BYTES = 1000  # limit preview size returned to caller
ALLOWED_SCHEMES = {"http", "https"}

# Toggle: allow localhost/127.0.0.1 only when explicitly enabled (dev convenience)
ALLOW_LOCALHOST = os.getenv("ALLOW_LOCALHOST_FETCH", "false").lower() == "true"

# Simple <title> extractor (best-effort)
TITLE_RE = re.compile(
    r"<\s*title[^>]*>(.*?)</\s*title\s*>",
    re.IGNORECASE | re.DOTALL,
)


# -----------------------------
# Helpers
# -----------------------------
def _is_ip_private(host: str) -> bool:
    """
    True if `host` is a literal private/loopback IP.
    We do not resolve DNS here (intentionally).
    """
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_private or ip.is_loopback or ip.is_link_local
    except ValueError:
        # Not a literal IP; likely a hostname.
        return False


def _extract_title(html: str) -> Optional[str]:
    m = TITLE_RE.search(html)
    if not m:
        return None
    # collapse whitespace
    return re.sub(r"\s+", " ", m.group(1)).strip()


def _safe_url(url: str) -> str:
    """
    Basic SSRF guard:
      - only http/https
      - require hostname
      - block localhost/127.0.0.1 unless ALLOW_LOCALHOST_FETCH=true
      - block literal private/loopback IPs
    """
    p = urlparse(url.strip())
    scheme = (p.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError("Only http/https URLs are allowed.")

    if not p.netloc:
        raise ValueError("URL must include a host.")

    host = (p.hostname or "").lower()

    # Localhost/loopback guard
    if host in {"localhost", "127.0.0.1"}:
        if not ALLOW_LOCALHOST:
            raise ValueError("Localhost is blocked.")
        # allowed in dev when toggled
        return url

    # Block literal private IPs (e.g., 10.x, 192.168.x, 172.16-31.x)
    if _is_ip_private(host):
        raise ValueError("Private/loopback IPs are blocked.")

    return url


# -----------------------------
# Service
# -----------------------------
class WebService:
    """
    Minimal, sandboxed web fetcher:
      - follows redirects
      - short timeout
      - returns: final URL, status, content-type, <title>, and a short preview
    """

    def __init__(self, logger=None):
        self.logger = logger

    async def fetch_preview(self, url: str) -> Dict[str, Any]:
        url = _safe_url(url)

        headers = {
            "User-Agent": "SEED-Fetch/1.0 (+local)",
            "Accept": "text/html, text/plain;q=0.9, */*;q=0.1",
        }

        async with httpx.AsyncClient(
            headers=headers,
            timeout=DEFAULT_TIMEOUT,
            follow_redirects=True,
            verify=True,
        ) as client:
            resp = await client.get(url)

        # derive response info
        ctype = resp.headers.get("content-type", "").lower()
        text_like = "text/html" in ctype or "text/plain" in ctype or ctype.startswith("text/")

        # best-effort preview (bounded)
        raw = resp.text if text_like else ""
        preview = raw[:MAX_PREVIEW_BYTES]
        title = _extract_title(raw) if raw else None

        info = {
            "url": str(resp.url),
            "status_code": resp.status_code,
            "content_type": ctype or None,
            "title": title,
            "preview": preview,
        }

        if self.logger:
            self.logger.info("Web fetch %s -> %s", info["url"], info["status_code"])

        return info
