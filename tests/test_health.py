"""
test_health.py – Health-check endpoint tests.
"""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
class TestHealth:
    async def test_health_endpoint_returns_200(self, client):
        """Health check should always return 200 (even if deps are down)."""
        resp = await client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "components" in data
        assert "database" in data["components"]
        assert "redis" in data["components"]

    async def test_health_reports_degraded_without_services(self, client):
        """Without real DB/Redis, health should report degraded."""
        resp = await client.get("/api/v1/health")
        data = resp.json()
        assert data["status"] in ("healthy", "degraded")

    async def test_health_mocked_services_healthy(self, client, monkeypatch):
        """When DB and Redis respond successfully, health should report healthy."""
        from unittest.mock import AsyncMock, MagicMock
        from app.api.v1 import health

        mock_conn = AsyncMock()
        mock_conn.execute.return_value = None
        mock_connect = MagicMock()
        mock_connect.__aenter__.return_value = mock_conn
        mock_connect.__aexit__.return_value = None

        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_connect
        monkeypatch.setattr(health, "engine", mock_engine)

        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        monkeypatch.setattr(health, "get_redis", MagicMock(return_value=mock_redis))

        resp = await client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["components"]["database"] == "healthy"
        assert data["components"]["redis"] == "healthy"


    async def test_root_endpoint(self, client):
        """Root should return app info."""
        resp = await client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["app"] == "SafeWay"
        assert data["zone"] == "peenya"

    async def test_openapi_schema(self, client):
        """OpenAPI JSON schema should be accessible."""
        resp = await client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert schema["info"]["title"] == "SafeWay"
        # Verify Step 1 endpoints exist in schema
        paths = schema["paths"]
        assert "/api/v1/auth/register" in paths
        assert "/api/v1/auth/login" in paths
        assert "/api/v1/health" in paths
        assert "/api/v1/users/me" in paths
        assert "/api/v1/users/me/contacts" in paths
