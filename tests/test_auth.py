"""
test_auth.py – Registration, login, JWT, and role-based access tests.
"""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
class TestRegister:
    async def test_register_success(self, client, sample_register_payload):
        resp = await client.post("/api/v1/auth/register", json=sample_register_payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["email"] == sample_register_payload["email"]
        assert data["role"] == "woman"
        assert data["trust_score"] == 1.0
        assert data["is_active"] is True
        assert "id" in data
        # Password must not be returned
        assert "hashed_password" not in data
        assert "password" not in data

    async def test_register_duplicate_email(self, client, sample_register_payload):
        await client.post("/api/v1/auth/register", json=sample_register_payload)
        resp = await client.post("/api/v1/auth/register", json=sample_register_payload)
        assert resp.status_code == 409

    async def test_register_short_password(self, client):
        resp = await client.post("/api/v1/auth/register", json={
            "email": "short@example.com",
            "phone": "+910000000001",
            "password": "abc",
            "name": "Short",
            "role": "woman",
        })
        assert resp.status_code == 422  # Pydantic validation

    async def test_register_invalid_role(self, client):
        resp = await client.post("/api/v1/auth/register", json={
            "email": "bad_role@example.com",
            "phone": "+910000000002",
            "password": "SecurePass123!",
            "name": "BadRole",
            "role": "hacker",
        })
        assert resp.status_code == 422

    async def test_register_all_roles(self, client):
        """Every valid role should be accepted."""
        for role in ("woman", "child", "parent", "volunteer", "authority"):
            import uuid
            resp = await client.post("/api/v1/auth/register", json={
                "email": f"{role}_{uuid.uuid4().hex[:6]}@example.com",
                "phone": f"+91{uuid.uuid4().int % 10**10:010d}",
                "password": "SecurePass123!",
                "name": f"Test {role}",
                "role": role,
            })
            assert resp.status_code == 201, f"Failed for role={role}: {resp.text}"


@pytest.mark.asyncio
class TestLogin:
    async def test_login_success(self, client, sample_register_payload):
        # Register first
        await client.post("/api/v1/auth/register", json=sample_register_payload)

        # Login
        resp = await client.post("/api/v1/auth/login", json={
            "email": sample_register_payload["email"],
            "password": sample_register_payload["password"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["role"] == "woman"
        assert "user_id" in data

    async def test_login_wrong_password(self, client, sample_register_payload):
        await client.post("/api/v1/auth/register", json=sample_register_payload)
        resp = await client.post("/api/v1/auth/login", json={
            "email": sample_register_payload["email"],
            "password": "WrongPassword999!",
        })
        assert resp.status_code == 401

    async def test_login_nonexistent_user(self, client):
        resp = await client.post("/api/v1/auth/login", json={
            "email": "nobody@example.com",
            "password": "whatever",
        })
        assert resp.status_code == 401

    async def test_login_oauth2_form_success(self, client, sample_register_payload):
        """OAuth2 form login via /token endpoint."""
        await client.post("/api/v1/auth/register", json=sample_register_payload)
        resp = await client.post("/api/v1/auth/token", data={
            "username": sample_register_payload["email"],
            "password": sample_register_payload["password"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["role"] == "woman"



@pytest.mark.asyncio
class TestProtectedEndpoints:
    async def test_profile_without_token(self, client):
        """Accessing /users/me without a token should return 401."""
        resp = await client.get("/api/v1/users/me")
        assert resp.status_code == 401

    async def test_profile_with_token(self, client, sample_register_payload):
        # Register + login
        await client.post("/api/v1/auth/register", json=sample_register_payload)
        login_resp = await client.post("/api/v1/auth/login", json={
            "email": sample_register_payload["email"],
            "password": sample_register_payload["password"],
        })
        token = login_resp.json()["access_token"]

        # Access profile
        resp = await client.get(
            "/api/v1/users/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["email"] == sample_register_payload["email"]

    async def test_profile_update(self, client, sample_register_payload):
        await client.post("/api/v1/auth/register", json=sample_register_payload)
        login_resp = await client.post("/api/v1/auth/login", json={
            "email": sample_register_payload["email"],
            "password": sample_register_payload["password"],
        })
        token = login_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        resp = await client.patch(
            "/api/v1/users/me",
            json={"name": "Updated Name", "profile_type": "commuter"},
            headers=headers,
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "Updated Name"
        assert resp.json()["profile_type"] == "commuter"


@pytest.mark.asyncio
class TestTrustedContacts:
    async def _auth_headers(self, client, sample_register_payload) -> dict:
        await client.post("/api/v1/auth/register", json=sample_register_payload)
        login_resp = await client.post("/api/v1/auth/login", json={
            "email": sample_register_payload["email"],
            "password": sample_register_payload["password"],
        })
        token = login_resp.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    async def test_crud_flow(self, client, sample_register_payload):
        headers = await self._auth_headers(client, sample_register_payload)

        # Create
        resp = await client.post("/api/v1/users/me/contacts", json={
            "contact_name": "Mom",
            "contact_phone": "+919876543210",
            "relationship_label": "mother",
            "priority": 1,
        }, headers=headers)
        assert resp.status_code == 201
        contact = resp.json()
        contact_id = contact["id"]
        assert contact["contact_name"] == "Mom"

        # List
        resp = await client.get("/api/v1/users/me/contacts", headers=headers)
        assert resp.status_code == 200
        assert len(resp.json()) >= 1

        # Update
        resp = await client.patch(
            f"/api/v1/users/me/contacts/{contact_id}",
            json={"contact_name": "Mother", "priority": 2},
            headers=headers,
        )
        assert resp.status_code == 200
        assert resp.json()["contact_name"] == "Mother"
        assert resp.json()["priority"] == 2

        # Delete
        resp = await client.delete(
            f"/api/v1/users/me/contacts/{contact_id}",
            headers=headers,
        )
        assert resp.status_code == 204

    async def test_contacts_without_auth(self, client):
        resp = await client.get("/api/v1/users/me/contacts")
        assert resp.status_code == 401


@pytest.mark.asyncio
class TestRoleBasedAccess:
    async def test_require_roles_allows_matching_role(self):
        import uuid
        from app.core.security import create_access_token, require_roles

        token = create_access_token(uuid.uuid4(), "authority")
        checker = require_roles("authority", "volunteer")
        payload = await checker(token)
        assert payload["role"] == "authority"

    async def test_require_roles_denies_unauthorized_role(self):
        import uuid
        from fastapi import HTTPException
        from app.core.security import create_access_token, require_roles

        token = create_access_token(uuid.uuid4(), "woman")
        checker = require_roles("authority")
        with pytest.raises(HTTPException) as exc_info:
            await checker(token)
        assert exc_info.value.status_code == 403


@pytest.mark.asyncio
class TestRateLimiter:
    async def test_rate_limiter_blocks_excessive_requests(self):
        from starlette.requests import Request
        from starlette.responses import Response
        from app.core.middleware import RateLimitMiddleware

        app_mock = lambda scope, receive, send: None
        mw = RateLimitMiddleware(app_mock, max_requests=2, window_seconds=60)

        async def call_next(req):
            return Response("ok", status_code=200)

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/test",
            "headers": [],
            "client": ("127.0.0.1", 12345),
        }
        req = Request(scope)
        res1 = await mw.dispatch(req, call_next)
        assert res1.status_code == 200
        res2 = await mw.dispatch(req, call_next)
        assert res2.status_code == 200
        res3 = await mw.dispatch(req, call_next)
        assert res3.status_code == 429

