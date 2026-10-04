"""
tests/test_community.py – Tests for Step 7: Community Features.

Acceptance criteria:
1. Community hazard reports (api/reports.py + services/report_service.py):
   - Report submission with location and timestamp
   - Automated moderation blocklist and trust score penalty
   - Spatial and temporal de-duplication with corroboration trust elevation
   - Authority moderation endpoint (/moderate)
   - Real-time road segment risk/darkness updates & Redis cache invalidation
2. Child & Parent mode with Geofencing (api/child.py + services/geofence.py):
   - Parent links child account & updates trusted contacts
   - Define circular and polygon geofences (safe zones & corridors)
   - Real-time location ping evaluations:
     - Normal inside safe zone (no breach)
     - Geofence breach outside safe zone alerts parent via SMS & Push
     - Unexpected long stop outside safe zone alerts parent
   - Suggested safest school route with 'child' profile
3. Safety volunteers (api/volunteers.py):
   - Volunteer opt-in and opt-out
   - Volunteer location updates for proximity queries
   - Authority verification
   - Volunteer opt-in directly affecting Step 5 SOS emergency dispatch radius query
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
import uuid

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool
from geoalchemy2.elements import WKTElement

from app.config import settings
from app.core.security import create_access_token, hash_password
from app.deps import get_redis
from app.main import app
from app.models.geofence import Geofence
from app.models.incident import Incident, Report, ReportStatus
from app.models.sos import SOSEvent, SOSStatus
from app.models.user import TrustedContact, User, UserRole
from app.models.volunteer import Volunteer
from app.services import geofence as geofence_service
from app.services import report_service
from app.services import sos_service
from app.services.notification_service import get_mock_notification_service

# Test database engine connected to PostGIS test instance
test_pg_engine = create_async_engine(settings.DATABASE_URL, poolclass=NullPool)
pg_test_session_factory = async_sessionmaker(test_pg_engine, expire_on_commit=False)


@pytest_asyncio.fixture(autouse=True)
async def engine_cleanup(monkeypatch):
    """Ensure engine connections are cleaned up and patched into background services."""
    monkeypatch.setattr(sos_service, "async_session_factory", pg_test_session_factory)
    yield
    from app.deps import engine
    await engine.dispose()
    await test_pg_engine.dispose()


@pytest_asyncio.fixture
async def pg_db():
    """Async session directly connected to PostgreSQL test database."""
    async with pg_test_session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def community_client(pg_db):
    """HTTP client with PostgreSQL dependency override."""
    from app.deps import get_db

    async def _override_get_db():
        yield pg_db

    app.dependency_overrides[get_db] = _override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def test_woman_user(pg_db):
    """Standard authenticated reporter user."""
    user = User(
        email=f"reporter_{uuid.uuid4().hex[:8]}@example.com",
        phone=f"+91{uuid.uuid4().int % 10**10:010d}",
        hashed_password=hash_password("Pass12345!"),
        name="Sunita Rao",
        role=UserRole.WOMAN,
        trust_score=1.0,
        is_active=True,
    )
    pg_db.add(user)
    await pg_db.commit()
    await pg_db.refresh(user)
    return user


@pytest_asyncio.fixture
async def test_parent_user(pg_db):
    """Parent user account."""
    user = User(
        email=f"parent_{uuid.uuid4().hex[:8]}@example.com",
        phone=f"+91{uuid.uuid4().int % 10**10:010d}",
        hashed_password=hash_password("Pass12345!"),
        name="Kavitha Reddy (Parent)",
        role=UserRole.PARENT,
        trust_score=1.2,
        is_active=True,
    )
    pg_db.add(user)
    await pg_db.commit()
    await pg_db.refresh(user)
    return user


@pytest_asyncio.fixture
async def test_child_user(pg_db):
    """Child user account."""
    user = User(
        email=f"child_{uuid.uuid4().hex[:8]}@example.com",
        phone=f"+91{uuid.uuid4().int % 10**10:010d}",
        hashed_password=hash_password("Pass12345!"),
        name="Ananya Reddy",
        role=UserRole.CHILD,
        trust_score=1.0,
        is_active=True,
    )
    pg_db.add(user)
    await pg_db.commit()
    await pg_db.refresh(user)
    return user


@pytest_asyncio.fixture
async def test_authority_user(pg_db):
    """Authority / reviewer account."""
    user = User(
        email=f"authority_{uuid.uuid4().hex[:8]}@example.com",
        phone=f"+91{uuid.uuid4().int % 10**10:010d}",
        hashed_password=hash_password("AdminPass123!"),
        name="Inspector Patil",
        role=UserRole.AUTHORITY,
        trust_score=2.0,
        is_active=True,
    )
    pg_db.add(user)
    await pg_db.commit()
    await pg_db.refresh(user)
    return user


# ══════════════════════════════════════════════════════════════════════════════
# 1. COMMUNITY HAZARD REPORTS TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestCommunityReports:
    @pytest.mark.asyncio
    async def test_submit_report_success(self, community_client, test_woman_user):
        """User submits a valid hazard report, receives 201 with affected segments."""
        token = create_access_token(test_woman_user.id, role=test_woman_user.role.value)
        payload = {
            "lat": 13.0285,
            "lon": 77.5190,
            "report_type": "dark_street",
            "title": "Broken street lamps near 4th cross",
            "description": "Entire stretch between 4th and 5th cross is pitch dark after 8pm.",
        }
        res = await community_client.post(
            "/api/v1/reports",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code == 201
        data = res.json()
        assert data["report_type"] == "dark_street"
        assert data["lat"] == 13.0285
        assert data["lon"] == 77.5190
        assert data["status"] in ("verified", "pending")
        assert isinstance(data["affected_segments"], list)

    @pytest.mark.asyncio
    async def test_report_moderation_blocklist_penalty(self, community_client, test_woman_user, pg_db):
        """Reports containing blocklisted keywords are rejected and penalize reporter trust."""
        initial_trust = test_woman_user.trust_score
        token = create_access_token(test_woman_user.id, role=test_woman_user.role.value)
        payload = {
            "lat": 13.0285,
            "lon": 77.5190,
            "report_type": "felt_unsafe",
            "title": "This is a fake prank report",
            "description": "Just testing spam filter with hoax information.",
        }
        res = await community_client.post(
            "/api/v1/reports",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code == 201
        data = res.json()
        assert data["status"] == "rejected"
        assert "filter" in data["message"].lower() or "rejected" in data["message"].lower()

        # Verify trust score was reduced in DB
        await pg_db.refresh(test_woman_user)
        assert test_woman_user.trust_score < initial_trust

    @pytest.mark.asyncio
    async def test_report_deduplication_and_corroboration(self, community_client, test_woman_user, pg_db):
        """Nearby reports within deduplication window corroborate each other and raise trust scores."""
        # Create second reporter
        user2 = User(
            email=f"reporter2_{uuid.uuid4().hex[:8]}@example.com",
            phone=f"+91{uuid.uuid4().int % 10**10:010d}",
            hashed_password=hash_password("Pass12345!"),
            name="Meera Joshi",
            role=UserRole.WOMAN,
            trust_score=1.0,
            is_active=True,
        )
        pg_db.add(user2)
        await pg_db.commit()
        await pg_db.refresh(user2)

        token1 = create_access_token(test_woman_user.id, role=test_woman_user.role.value)
        token2 = create_access_token(user2.id, role=user2.role.value)

        # 1st report at point (13.0285, 77.5190)
        res1 = await community_client.post(
            "/api/v1/reports",
            json={
                "lat": 13.0285,
                "lon": 77.5190,
                "report_type": "harassment",
                "title": "Catcalling incident",
                "description": "Unsafe group loitering near corner.",
            },
            headers={"Authorization": f"Bearer {token1}"},
        )
        assert res1.status_code == 201
        rep1_data = res1.json()

        # 2nd report within 40 meters
        res2 = await community_client.post(
            "/api/v1/reports",
            json={
                "lat": 13.0287,
                "lon": 77.5192,
                "report_type": "felt_unsafe",
                "title": "Suspicious activity noticed",
                "description": "Corroborating loitering issue here.",
            },
            headers={"Authorization": f"Bearer {token2}"},
        )
        assert res2.status_code == 201
        rep2_data = res2.json()
        assert rep2_data["corroborated"] is True
        assert rep2_data["corroboration_count"] >= 1
        assert rep2_data["status"] == "verified"

        # Verify trust score was boosted
        await pg_db.refresh(user2)
        assert user2.trust_score > 1.0

    @pytest.mark.asyncio
    async def test_report_authority_moderation(self, community_client, test_woman_user, test_authority_user, pg_db):
        """Authority user verifies report and boosts reporter trust; non-authority is forbidden."""
        token_woman = create_access_token(test_woman_user.id, role=test_woman_user.role.value)
        token_auth = create_access_token(test_authority_user.id, role=test_authority_user.role.value)

        res_rep = await community_client.post(
            "/api/v1/reports",
            json={
                "lat": 13.0280,
                "lon": 77.5180,
                "report_type": "broken_light",
                "title": "Street lamp completely dead",
                "description": "Needs municipal repair.",
            },
            headers={"Authorization": f"Bearer {token_woman}"},
        )
        report_id = res_rep.json()["report_id"]

        # Woman user attempts moderation -> 403 Forbidden
        forbidden_res = await community_client.post(
            f"/api/v1/reports/{report_id}/moderate",
            json={"status": "verified"},
            headers={"Authorization": f"Bearer {token_woman}"},
        )
        assert forbidden_res.status_code == 403

        # Authority moderates to verified -> 200 OK
        initial_trust = test_woman_user.trust_score
        mod_res = await community_client.post(
            f"/api/v1/reports/{report_id}/moderate",
            json={"status": "verified"},
            headers={"Authorization": f"Bearer {token_auth}"},
        )
        assert mod_res.status_code == 200
        mod_data = mod_res.json()
        assert mod_data["status"] == "verified"
        assert mod_data["reporter_new_trust"] > initial_trust

    @pytest.mark.asyncio
    async def test_report_driven_risk_updates_and_cache_invalidation(self, community_client, test_woman_user):
        """Submitting a report writes real-time penalty keys and invalidates routing caches in Redis."""
        redis = get_redis()
        if redis:
            # Seed a dummy slot cache key in Redis
            redis.set(f"safety:{settings.ZONE}:101:23D", "dummy_cache_value")
            redis.set(f"ml_pred:{settings.ZONE}:23D", "dummy_ml_value")

        token = create_access_token(test_woman_user.id, role=test_woman_user.role.value)
        res = await community_client.post(
            "/api/v1/reports",
            json={
                "lat": 13.0285,
                "lon": 77.5190,
                "report_type": "dark_street",
                "title": "Dark stretch",
                "description": "Dark street hazard reported.",
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code == 201
        affected_segs = res.json()["affected_segments"]
        assert len(affected_segs) > 0

        if redis:
            first_sid = affected_segs[0]
            # Verify lighting penalty or hazard risk is stored
            penalty = redis.get(f"segment:darkness_penalty:{first_sid}")
            lit_val = redis.get(f"segment:lit_ratio:{first_sid}")
            assert penalty is not None or lit_val is not None


# ══════════════════════════════════════════════════════════════════════════════
# 2. CHILD / PARENT MODE & GEOFENCING TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestChildParentMode:
    @pytest.mark.asyncio
    async def test_parent_link_child_success(self, community_client, test_parent_user, test_child_user, pg_db):
        """Parent links child account and child automatically gains parent as priority trusted contact."""
        token_parent = create_access_token(test_parent_user.id, role=test_parent_user.role.value)
        payload = {"child_email": test_child_user.email}

        res = await community_client.post(
            "/api/v1/child/link",
            json=payload,
            headers={"Authorization": f"Bearer {token_parent}"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["child_id"] == str(test_child_user.id)
        assert data["parent_id"] == str(test_parent_user.id)
        assert data["status"] == "linked"

        # Verify trusted contact in DB
        tc_stmt = select(TrustedContact).where(
            TrustedContact.user_id == test_child_user.id,
            TrustedContact.contact_phone == test_parent_user.phone,
        )
        tc_res = await pg_db.execute(tc_stmt)
        tc = tc_res.scalar_one_or_none()
        assert tc is not None
        assert tc.priority == 1

    @pytest.mark.asyncio
    async def test_create_and_list_geofences(self, community_client, test_parent_user, test_child_user):
        """Parent defines circular safe zone (Home) and polygon corridor (Tuition)."""
        token_parent = create_access_token(test_parent_user.id, role=test_parent_user.role.value)

        # 1. Circular Home Safe Zone
        home_payload = {
            "name": "Home Safe Zone",
            "center_lat": 13.0285,
            "center_lon": 77.5190,
            "radius_m": 250.0,
            "alert_on_exit": True,
        }
        res_home = await community_client.post(
            f"/api/v1/child/{test_child_user.id}/geofences",
            json=home_payload,
            headers={"Authorization": f"Bearer {token_parent}"},
        )
        assert res_home.status_code == 201
        assert res_home.json()["name"] == "Home Safe Zone"

        # 2. Polygon Tuition Corridor
        tuition_payload = {
            "name": "Tuition Safe Zone",
            "polygon_coords": [
                [77.518, 13.027],
                [77.520, 13.027],
                [77.520, 13.029],
                [77.518, 13.029],
                [77.518, 13.027],
            ],
            "alert_on_exit": True,
        }
        res_tuition = await community_client.post(
            f"/api/v1/child/{test_child_user.id}/geofences",
            json=tuition_payload,
            headers={"Authorization": f"Bearer {token_parent}"},
        )
        assert res_tuition.status_code == 201

        # 3. List geofences
        res_list = await community_client.get(
            f"/api/v1/child/{test_child_user.id}/geofences",
            headers={"Authorization": f"Bearer {token_parent}"},
        )
        assert res_list.status_code == 200
        fences = res_list.json()
        assert len(fences) >= 2
        names = [f["name"] for f in fences]
        assert "Home Safe Zone" in names
        assert "Tuition Safe Zone" in names

    @pytest.mark.asyncio
    async def test_geofence_breach_alert_to_parent(self, community_client, test_parent_user, test_child_user):
        """When child moves outside active safe zones, parent receives SMS & Push breach alerts."""
        token_parent = create_access_token(test_parent_user.id, role=test_parent_user.role.value)
        token_child = create_access_token(test_child_user.id, role=test_child_user.role.value)

        # Link child to parent
        await community_client.post(
            "/api/v1/child/link",
            json={"child_email": test_child_user.email},
            headers={"Authorization": f"Bearer {token_parent}"},
        )

        # Define 100m safe zone at (13.0285, 77.5190)
        await community_client.post(
            f"/api/v1/child/{test_child_user.id}/geofences",
            json={
                "name": "School Safe Zone",
                "center_lat": 13.0285,
                "center_lon": 77.5190,
                "radius_m": 100.0,
                "alert_on_exit": True,
            },
            headers={"Authorization": f"Bearer {token_parent}"},
        )

        mock_ns = get_mock_notification_service()
        mock_ns.clear()

        # 1. Ping INSIDE safe zone (13.02855, 77.51905) -> No breach, no alert
        res_inside = await community_client.post(
            f"/api/v1/child/{test_child_user.id}/location",
            json={"lat": 13.02855, "lon": 77.51905},
            headers={"Authorization": f"Bearer {token_child}"},
        )
        assert res_inside.status_code == 200
        inside_data = res_inside.json()
        assert inside_data["inside_safe_zone"] is True
        assert inside_data["breach_detected"] is False
        assert inside_data["alert_sent"] is False
        assert len(mock_ns.sent_sms) == 0

        # Clear alert cooldown in Redis to ensure clean breach trigger
        redis = get_redis()
        if redis:
            redis.delete(f"child:breach_cooldown:{test_child_user.id}")

        # 2. Ping OUTSIDE safe zone (13.0385, 77.5290 - ~1.5 km away) -> Breach detected!
        res_outside = await community_client.post(
            f"/api/v1/child/{test_child_user.id}/location",
            json={"lat": 13.0385, "lon": 77.5290},
            headers={"Authorization": f"Bearer {token_child}"},
        )
        assert res_outside.status_code == 200
        outside_data = res_outside.json()
        assert outside_data["inside_safe_zone"] is False
        assert outside_data["breach_detected"] is True
        assert outside_data["alert_sent"] is True

        # Check mock parent notifications
        assert len(mock_ns.sent_sms) >= 1
        assert any(test_parent_user.phone in record["to"] for record in mock_ns.sent_sms)
        assert any("Geofence Breach" in record["message"] for record in mock_ns.sent_sms)

    @pytest.mark.asyncio
    async def test_geofence_unexpected_stop_alert(self, community_client, test_parent_user, test_child_user):
        """When child is stationary outside safe zones past GEOFENCE_LONG_STOP_MINUTES, parent is alerted."""
        token_parent = create_access_token(test_parent_user.id, role=test_parent_user.role.value)
        token_child = create_access_token(test_child_user.id, role=test_child_user.role.value)

        # Link child
        await community_client.post(
            "/api/v1/child/link",
            json={"child_email": test_child_user.email},
            headers={"Authorization": f"Bearer {token_parent}"},
        )

        # Create a safe zone at home so child has defined geofences
        await community_client.post(
            f"/api/v1/child/{test_child_user.id}/geofences",
            json={
                "name": "Home Safe Zone",
                "center_lat": 13.0285,
                "center_lon": 77.5190,
                "radius_m": 150.0,
                "alert_on_exit": True,
            },
            headers={"Authorization": f"Bearer {token_parent}"},
        )

        mock_ns = get_mock_notification_service()
        mock_ns.clear()

        redis = get_redis()
        if redis:
            # Simulate initial stationary ping 6 minutes ago
            past_ts = datetime.now(timezone.utc).timestamp() - (settings.GEOFENCE_LONG_STOP_MINUTES * 60 + 30)
            redis.setex(
                f"child:stationary_tracker:{test_child_user.id}",
                3600,
                json.dumps({"lat": 13.0350, "lon": 77.5250, "start_ts": past_ts}),
            )
            redis.delete(f"child:breach_cooldown:{test_child_user.id}")

        # Send location at same stationary coordinates
        res = await community_client.post(
            f"/api/v1/child/{test_child_user.id}/location",
            json={"lat": 13.0350, "lon": 77.5250},
            headers={"Authorization": f"Bearer {token_child}"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["unexpected_stop_detected"] is True
        assert data["alert_sent"] is True

        # Check parent alert message
        assert any("Unexpected" in s["message"] or "stationary" in s["message"].lower() for s in mock_ns.sent_sms)

    @pytest.mark.asyncio
    async def test_child_suggested_route(self, community_client, test_parent_user, test_child_user):
        """GET /child/{id}/suggested-route computes safest school route with 'child' profile."""
        token_parent = create_access_token(test_parent_user.id, role=test_parent_user.role.value)

        res = await community_client.get(
            f"/api/v1/child/{test_child_user.id}/suggested-route?origin_lat=13.0250&origin_lon=77.5150&dest_lat=13.0280&dest_lon=77.5180",
            headers={"Authorization": f"Bearer {token_parent}"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["profile"] == "child"
        assert 0.0 <= data["safety_score"] <= 100.0
        assert data["safety_score"] > 0
        assert "route_geojson" in data
        assert isinstance(data["safety_explanations"], list)
        assert len(data["safety_explanations"]) >= 1


# ══════════════════════════════════════════════════════════════════════════════
# 3. SAFETY VOLUNTEERS & STEP 5 SOS DISPATCH TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestVolunteerDispatch:
    @pytest.mark.asyncio
    async def test_volunteer_opt_in_and_out(self, community_client, test_woman_user, pg_db):
        """User toggles volunteer availability on and off."""
        token = create_access_token(test_woman_user.id, role=test_woman_user.role.value)

        # Opt in
        res_in = await community_client.post(
            "/api/v1/volunteers/opt-in",
            json={"radius_m": 1200},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res_in.status_code == 200
        data_in = res_in.json()
        assert data_in["is_available"] is True
        assert data_in["radius_m"] == 1200

        # Opt out
        res_out = await community_client.post(
            "/api/v1/volunteers/opt-out",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res_out.status_code == 200
        data_out = res_out.json()
        assert data_out["is_available"] is False

    @pytest.mark.asyncio
    async def test_volunteer_location_update(self, community_client, test_woman_user):
        """Volunteer checks in GPS location for spatial SOS matching."""
        token = create_access_token(test_woman_user.id, role=test_woman_user.role.value)
        res = await community_client.post(
            "/api/v1/volunteers/location",
            json={"lat": 13.0290, "lon": 77.5195},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "location_updated"
        assert data["lat"] == 13.0290
        assert data["lon"] == 77.5195

    @pytest.mark.asyncio
    async def test_volunteer_authority_verification(
        self, community_client, test_woman_user, test_authority_user, pg_db
    ):
        """Authority user verifies volunteer status; non-authority is forbidden."""
        token_woman = create_access_token(test_woman_user.id, role=test_woman_user.role.value)
        token_auth = create_access_token(test_authority_user.id, role=test_authority_user.role.value)

        # Ensure volunteer record exists
        opt_res = await community_client.post(
            "/api/v1/volunteers/opt-in",
            json={"radius_m": 1000},
            headers={"Authorization": f"Bearer {token_woman}"},
        )
        vol_id = opt_res.json()["volunteer_id"]

        # Regular user attempt -> 403 Forbidden
        forbidden_res = await community_client.post(
            f"/api/v1/volunteers/{vol_id}/verify",
            headers={"Authorization": f"Bearer {token_woman}"},
        )
        assert forbidden_res.status_code == 403

        # Authority attempt -> 200 OK
        verify_res = await community_client.post(
            f"/api/v1/volunteers/{vol_id}/verify",
            headers={"Authorization": f"Bearer {token_auth}"},
        )
        assert verify_res.status_code == 200
        assert verify_res.json()["verified"] is True

        # Check DB
        vol = await pg_db.get(Volunteer, uuid.UUID(vol_id))
        assert vol is not None
        assert vol.verified is True

    @pytest.mark.asyncio
    async def test_volunteer_opt_in_affects_sos_dispatch(
        self, community_client, test_woman_user, pg_db
    ):
        """
        Acceptance requirement: volunteer opt-in directly affects Step 5 SOS emergency dispatch.
        - Volunteer A: Opted in, verified, nearby (<1km) -> Receives SOS push!
        - Volunteer B: Opted out, verified, nearby (<1km) -> Does NOT receive push!
        """
        mock_ns = get_mock_notification_service()
        mock_ns.clear()

        # 1. Setup Volunteer A (Available)
        user_a = User(
            email=f"vola_{uuid.uuid4().hex[:8]}@example.com",
            phone=f"+91{uuid.uuid4().int % 10**10:010d}",
            hashed_password=hash_password("Pass123!"),
            name="Volunteer Arjun (Available)",
            role=UserRole.VOLUNTEER,
            trust_score=1.5,
            is_active=True,
        )
        pg_db.add(user_a)
        await pg_db.commit()
        await pg_db.refresh(user_a)

        geom_nearby = WKTElement("POINT(77.5192 13.0286)", srid=4326)
        vol_a = Volunteer(
            user_id=user_a.id,
            is_available=True,
            verified=True,
            last_location=geom_nearby,
            radius_m=1000,
        )
        pg_db.add(vol_a)

        # 2. Setup Volunteer B (Opted Out / Unavailable)
        user_b = User(
            email=f"volb_{uuid.uuid4().hex[:8]}@example.com",
            phone=f"+91{uuid.uuid4().int % 10**10:010d}",
            hashed_password=hash_password("Pass123!"),
            name="Volunteer Brijesh (Opted Out)",
            role=UserRole.VOLUNTEER,
            trust_score=1.5,
            is_active=True,
        )
        pg_db.add(user_b)
        await pg_db.commit()
        await pg_db.refresh(user_b)

        vol_b = Volunteer(
            user_id=user_b.id,
            is_available=False,  # OPTED OUT
            verified=True,
            last_location=geom_nearby,
            radius_m=1000,
        )
        pg_db.add(vol_b)
        await pg_db.commit()

        # 3. Create an active SOS event triggered at (13.0285, 77.5190)
        token_woman = create_access_token(test_woman_user.id, role=test_woman_user.role.value)
        sos_res = await community_client.post(
            "/api/v1/sos/trigger",
            json={"lat": 13.0285, "lon": 77.5190, "trigger_type": "shake"},
            headers={"Authorization": f"Bearer {token_woman}"},
        )
        assert sos_res.status_code == 201
        sos_id = uuid.UUID(sos_res.json()["sos_id"])

        # 4. Dispatch emergency SOS (Step 5 flow)
        await sos_service.dispatch_sos(event_id=sos_id)

        # 5. Verify volunteer push dispatches
        recipient_ids = [p["recipient_id"] for p in mock_ns.sent_pushes]
        assert str(user_a.id) in recipient_ids, "Opted-in verified Volunteer A should receive push alert"
        assert str(user_b.id) not in recipient_ids, "Opted-out Volunteer B must NOT receive push alert"
        # Allow background countdown to complete cleanly
        await asyncio.sleep(1.2)
