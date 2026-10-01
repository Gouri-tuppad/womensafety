"""
health.py – System health check that probes DB and Redis.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from sqlalchemy import text

from app.deps import engine, get_redis

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", summary="System health check (DB + Redis)")
async def health_check():
    """
    Return overall health plus individual component status.

    Always returns ``200`` so load-balancers can distinguish between
    "app is running" and "dependency is down" via the body.
    """
    components: dict[str, str] = {}

    # ── Database ─────────────────────────────────────────────────────────
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        components["database"] = "healthy"
    except Exception as exc:
        logger.warning("Health: DB unreachable – %s", exc)
        components["database"] = f"unhealthy: {exc}"

    # ── Redis ────────────────────────────────────────────────────────────
    try:
        r = get_redis()
        r.ping()
        components["redis"] = "healthy"
    except Exception as exc:
        logger.warning("Health: Redis unreachable – %s", exc)
        components["redis"] = f"unhealthy: {exc}"

    # ── Overall ──────────────────────────────────────────────────────────
    all_healthy = all(v == "healthy" for v in components.values())

    return {
        "status": "healthy" if all_healthy else "degraded",
        "components": components,
    }
