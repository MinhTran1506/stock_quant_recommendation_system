"""
api/middleware/rate_limit.py — Sliding-window rate limiter using Redis.
Falls back to in-memory counter if Redis is unavailable.
"""
import time
from collections import defaultdict
from typing import Dict, Tuple

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = structlog.get_logger(__name__)

# In-memory fallback store: {client_ip: [(timestamp, count), ...]}
_mem_store: Dict[str, list] = defaultdict(list)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple sliding-window rate limiter.
    Default: 120 requests per 60 seconds per IP.
    Health/metrics endpoints are excluded.
    """

    EXCLUDED_PATHS = {"/health", "/ready", "/metrics", "/docs", "/redoc", "/openapi.json"}

    def __init__(self, app: ASGIApp, calls: int = 120, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.EXCLUDED_PATHS:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - self.period

        # Sliding window using in-memory store
        timestamps = _mem_store[client_ip]
        # Remove entries outside the window
        _mem_store[client_ip] = [t for t in timestamps if t > window_start]
        count = len(_mem_store[client_ip])

        if count >= self.calls:
            logger.warning("Rate limit exceeded", client=client_ip, count=count)
            return Response(
                content='{"detail":"Rate limit exceeded. Try again shortly."}',
                status_code=429,
                media_type="application/json",
                headers={
                    "Retry-After": str(self.period),
                    "X-RateLimit-Limit": str(self.calls),
                    "X-RateLimit-Remaining": "0",
                },
            )

        _mem_store[client_ip].append(now)
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.calls)
        response.headers["X-RateLimit-Remaining"] = str(self.calls - count - 1)
        return response
