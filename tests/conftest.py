"""
conftest.py – Test fixtures for SafeWay Step 1.

Uses an in-memory SQLite database (via aiosqlite) for auth tests.
Only auth-related tables (users, trusted_contacts) are created because
PostGIS geometry types are not supported in SQLite.
"""
from __future__ import annotations

# Ensure compatibility between scikit-learn >=1.6 and xgboost <=2.1.4
try:
    from sklearn.base import ClassifierMixin, RegressorMixin

    if not hasattr(RegressorMixin, "_estimator_type"):
        RegressorMixin._estimator_type = "regressor"
    if not hasattr(ClassifierMixin, "_estimator_type"):
        ClassifierMixin._estimator_type = "classifier"
except ImportError:
    pass

import asyncio
import uuid
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.models.base import Base
from app.models.user import User, TrustedContact

# ── Test engine (SQLite in-memory) ───────────────────────────────────────────

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

test_engine: AsyncEngine = create_async_engine(TEST_DB_URL, echo=False)
test_session_factory = async_sessionmaker(
    test_engine, class_=AsyncSession, expire_on_commit=False,
)



# ── Create / drop tables ────────────────────────────────────────────────────

@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_database():
    """Create auth-related tables once before any tests run."""
    async with test_engine.begin() as conn:
        # Only create tables that work in SQLite (no PostGIS geometry)
        await conn.run_sync(lambda sync_conn: User.__table__.create(sync_conn, checkfirst=True))
        await conn.run_sync(lambda sync_conn: TrustedContact.__table__.create(sync_conn, checkfirst=True))
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(lambda sync_conn: TrustedContact.__table__.drop(sync_conn, checkfirst=True))
        await conn.run_sync(lambda sync_conn: User.__table__.drop(sync_conn, checkfirst=True))


# ── DB session override ─────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    async with test_session_factory() as session:
        yield session
        # Roll back any uncommitted changes between tests
        await session.rollback()


# ── AsyncClient with dependency override ─────────────────────────────────────

@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """HTTP client that swaps the real DB dep for the test SQLite session."""
    from app.main import app
    from app.deps import get_db

    async def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


# ── Helper fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def sample_register_payload() -> dict:
    """Valid registration payload."""
    return {
        "email": f"test_{uuid.uuid4().hex[:8]}@example.com",
        "phone": f"+91{uuid.uuid4().int % 10**10:010d}",
        "password": "SecurePass123!",
        "name": "Test User",
        "role": "woman",
    }
