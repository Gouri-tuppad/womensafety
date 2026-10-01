"""
deps.py – FastAPI dependency injection (DB sessions, Redis, current user).
"""

from __future__ import annotations

import uuid
import logging
from typing import Annotated, AsyncGenerator

from fastapi import Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import settings
from app.core.security import decode_access_token, oauth2_scheme

logger = logging.getLogger(__name__)

# ── Async SQLAlchemy engine ──────────────────────────────────────────────────

engine: AsyncEngine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_pre_ping=True,
)

async_session_factory = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session (FastAPI Depends)."""
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


# ── Redis (lazy) ─────────────────────────────────────────────────────────────

_redis_client = None


def get_redis():
    """Return a Redis client (lazy-initialised)."""
    global _redis_client
    if _redis_client is None:
        import redis as redis_lib
        _redis_client = redis_lib.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis_client


# ── Current user dependency (combines token + DB) ────────────────────────────

async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: AsyncSession = Depends(get_db),
):
    """Decode the JWT, load the user from DB, verify active status."""
    from app.models.user import User  # deferred to avoid circular import

    payload = decode_access_token(token)
    user_id = uuid.UUID(payload["sub"])

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive.",
        )
    return user
