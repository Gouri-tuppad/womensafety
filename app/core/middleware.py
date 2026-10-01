"""
middleware.py – Basic rate-limiting middleware (in-memory, per-IP).

For production scale, swap the in-memory dict for Redis counters.
"""

from __future__ import annotations

import time
import logging
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.config import settings

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Fixed-window rate limiter.

    Tracks request counts per client IP within a 60-second window.
    Returns ``429 Too Many Requests`` when the limit is exceeded.
    """

    def __init__(self, app, max_requests: int | None = None, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests or settings.RATE_LIMIT_PER_MINUTE
        self.window = window_seconds
        # ip -> list of timestamps
        self._clients: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks and docs
        if request.url.path in ("/health", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Prune old timestamps
        self._clients[client_ip] = [
            t for t in self._clients[client_ip] if now - t < self.window
        ]

        if len(self._clients[client_ip]) >= self.max_requests:
            logger.warning("Rate limit hit for %s", client_ip)
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
                headers={"Retry-After": str(self.window)},
            )

        self._clients[client_ip].append(now)
        return await call_next(request)
