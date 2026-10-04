"""
tests/test_trips.py – Tests for Step 6: Live Trip Monitoring & Anomaly Detection.

Covers:
- POST /api/v1/trips/start (stores planned route)
- POST /api/v1/trips/{id}/location (location + motion features)
- POST /api/v1/trips/{id}/end
- POST /api/v1/trips/{id}/feedback ('Did you feel safe?' 1-5 + EMA risk updates)
- POST /api/v1/trips/{id}/respond ('I'm okay')
- Anomaly simulations (deviation, long stop in high-risk segment, motion anomaly)
- 'Are you okay?' push prompt and auto-escalation into SOS when unanswered
- Prompt dismissal when user responds
- Privacy access control (trusted contacts only) and location retention purge
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
import uuid

import numpy as np
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

from app.config import settings
from app.core.security import create_access_token, hash_password
from app.deps import get_redis
from app.main import app
from app.models.feedback import Feedback
from app.models.sos import SOSEvent, SOSStatus
from app.models.trip import Trip, TripLocation, TripStatus
from app.models.user import TrustedContact, User, UserRole
from app.services.notification_service import get_mock_notification_service

# Test engine with NullPool
test_pg_engine = create_async_engine(settings.DATABASE_URL, poolclass=NullPool)
pg_test_session_factory = async_sessionmaker(test_pg_engine, expire_on_commit=False)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture(autouse=True)
async def engine_cleanup(monkeypatch):
    """Ensure engine connections are closed per test and patched into services."""
    import app.services.sos_service as sos_mod
    import app.services.trip_monitor as trip_mod

    monkeypatch.setattr(sos_mod, "async_session_factory", pg_test_session_factory)
    monkeypatch.setattr(trip_mod, "async_session_factory", pg_test_session_factory)
    yield
    from app.deps import engine
    await engine.dispose()
    await test_pg_engine.dispose()


@pytest_asyncio.fixture
async def pg_db():
    async with pg_test_session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def trip_client(pg_db):
    from app.deps import get_db

    async def _override_get_db():
        yield pg_db

    app.dependency_overrides[get_db] = _override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def primary_user(pg_db):
    """Woman user with registered trusted emergency contacts."""
    user = User(
        email=f"woman_trip_{uuid.uuid4().hex[:8]}@example.com",
        phone=f"+91{uuid.uuid4().int % 10**10:010d}",
        hashed_password=hash_password("TripSafe123!"),
        name="Kavya Gowda",
        role=UserRole.WOMAN,
        is_active=True,
    )
    pg_db.add(user)
    await pg_db.commit()
    await pg_db.refresh(user)

    trusted_phone = f"+91{uuid.uuid4().int % 10**10:010d}"
    contact = TrustedContact(
        user_id=user.id,
        contact_name="Rani Gowda (Mother)",
        contact_phone=trusted_phone,
        relationship_label="Mother",
        priority=1,
    )
    pg_db.add(contact)
    await pg_db.commit()
    user.test_trusted_phone = trusted_phone
    return user


@pytest_asyncio.fixture
def auth_headers(primary_user):
    token = create_access_token(primary_user.id, primary_user.role.value)
    return {"Authorization": f"Bearer {token}"}


@pytest_asyncio.fixture
async def trusted_contact_user(pg_db, primary_user):
    """Another user whose phone matches the primary user's trusted contact."""
    user = User(
        email=f"rani_{uuid.uuid4().hex[:8]}@example.com",
        phone=primary_user.test_trusted_phone,
        hashed_password=hash_password("MotherPass123!"),
        name="Rani Gowda",
        role=UserRole.WOMAN,
        is_active=True,
    )
    pg_db.add(user)
    await pg_db.commit()
    await pg_db.refresh(user)
    return user


@pytest_asyncio.fixture
def trusted_headers(trusted_contact_user):
    token = create_access_token(trusted_contact_user.id, trusted_contact_user.role.value)
    return {"Authorization": f"Bearer {token}"}


@pytest_asyncio.fixture
async def stranger_user(pg_db):
    """Unrelated user with no relation to primary user."""
    user = User(
        email=f"stranger_{uuid.uuid4().hex[:8]}@example.com",
        phone=f"+91{uuid.uuid4().int % 10**10:010d}",
        hashed_password=hash_password("StrangerPass123!"),
        name="Random Stranger",
        role=UserRole.WOMAN,
        is_active=True,
    )
    pg_db.add(user)
    await pg_db.commit()
    await pg_db.refresh(user)
    return user


@pytest_asyncio.fixture
def stranger_headers(stranger_user):
    token = create_access_token(stranger_user.id, stranger_user.role.value)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def sample_route_geojson():
    """Planned polyline from Peenya Metro to 1st Stage."""
    return {
        "type": "LineString",
        "coordinates": [
            [77.5150, 13.0250],
            [77.5160, 13.0260],
            [77.5170, 13.0270],
            [77.5180, 13.0280],
        ],
    }


def generate_walking_window(n_points: int = 25) -> list[list[float]]:
    """Generate normal walking accelerometer data."""
    t = np.linspace(0, 1.5, n_points)
    ay = 9.8 + 1.2 * np.sin(2 * np.pi * 1.8 * t)
    ax = np.random.normal(0, 0.3, n_points)
    az = np.random.normal(0, 0.3, n_points)
    return np.column_stack([ax, ay, az]).tolist()


def generate_phone_drop_window(n_points: int = 25) -> list[list[float]]:
    """Generate violent phone drop / impact data."""
    data = np.zeros((n_points, 3), dtype=float)
    data[:, 1] = 9.8
    # Freefall
    data[5:9, :] = 0.1
    # Impact spike (> 40 m/s^2)
    data[10, :] = [45.0, -50.0, 38.0]
    data[11, :] = [-20.0, 25.0, -18.0]
    return data.tolist()


# ── Step 6 Test Cases ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestTripStartAndLifecycle:
    async def test_trip_start_stores_planned_route(self, trip_client, auth_headers, sample_route_geojson):
        """POST /api/v1/trips/start initializes trip and stores planned geometry."""
        res = await trip_client.post(
            "/api/v1/trips/start",
            json={
                "origin_lat": 13.0250,
                "origin_lon": 77.5150,
                "destination_lat": 13.0280,
                "destination_lon": 77.5180,
                "route_geojson": sample_route_geojson,
                "safety_score": 85.0,
                "mode": "safest",
            },
            headers=auth_headers,
        )
        assert res.status_code == 201
        data = res.json()
        assert data["status"] == "active"
        assert "trip_id" in data
        assert data["safety_score"] == 85.0
        assert data["route_geojson"] == sample_route_geojson

    async def test_trip_location_recorded_normal_walking(self, trip_client, auth_headers, sample_route_geojson):
        """POST /api/v1/trips/{id}/location on-route with normal walking yields no anomaly."""
        start_res = await trip_client.post(
            "/api/v1/trips/start",
            json={
                "origin_lat": 13.0250,
                "origin_lon": 77.5150,
                "destination_lat": 13.0280,
                "destination_lon": 77.5180,
                "route_geojson": sample_route_geojson,
            },
            headers=auth_headers,
        )
        trip_id = start_res.json()["trip_id"]

        # Ping right along the planned polyline with normal walking motion
        loc_res = await trip_client.post(
            f"/api/v1/trips/{trip_id}/location",
            json={
                "lat": 13.0260,
                "lon": 77.5160,
                "accuracy_m": 8.0,
                "motion_window": generate_walking_window(),
            },
            headers=auth_headers,
        )
        assert loc_res.status_code == 200
        data = loc_res.json()
        assert data["status"] == "recorded"
        assert data["anomaly_detected"] is False
        assert data["prompt_sent"] is False

    async def test_trip_end_completes_and_computes_duration(self, trip_client, auth_headers, sample_route_geojson):
        """POST /api/v1/trips/{id}/end transitions status to completed."""
        start_res = await trip_client.post(
            "/api/v1/trips/start",
            json={
                "origin_lat": 13.0250,
                "origin_lon": 77.5150,
                "destination_lat": 13.0280,
                "destination_lon": 77.5180,
                "route_geojson": sample_route_geojson,
            },
            headers=auth_headers,
        )
        trip_id = start_res.json()["trip_id"]

        end_res = await trip_client.post(f"/api/v1/trips/{trip_id}/end", headers=auth_headers)
        assert end_res.status_code == 200
        data = end_res.json()
        assert data["status"] == "completed"
        assert data["duration_seconds"] >= 0.0


@pytest.mark.asyncio
class TestAnomalySimulationAndEscalation:
    async def test_route_deviation_prompts_and_auto_escalates(
        self, trip_client, auth_headers, sample_route_geojson, monkeypatch, pg_db,
    ):
        """
        Simulation 1: Route deviation > threshold
        Triggers 'Are you okay?' push prompt and auto-escalates to SOS when unanswered.
        """
        # Fast escalation timer for tests: 1 second
        monkeypatch.setattr(settings, "TRIP_ARE_YOU_OKAY_SECONDS", 1)
        monkeypatch.setattr(settings, "SOS_COUNTDOWN_SECONDS", 0)

        notifier = get_mock_notification_service()
        notifier.clear()
        r = get_redis()
        if r:
            r.flushdb()

        start_res = await trip_client.post(
            "/api/v1/trips/start",
            json={
                "origin_lat": 13.0250,
                "origin_lon": 77.5150,
                "destination_lat": 13.0280,
                "destination_lon": 77.5180,
                "route_geojson": sample_route_geojson,
            },
            headers=auth_headers,
        )
        trip_id = start_res.json()["trip_id"]

        # Ping located > 2 km away from planned route (deviation)
        dev_res = await trip_client.post(
            f"/api/v1/trips/{trip_id}/location",
            json={
                "lat": 13.0450,
                "lon": 77.5350,
                "accuracy_m": 5.0,
            },
            headers=auth_headers,
        )
        assert dev_res.status_code == 200
        dev_data = dev_res.json()
        assert dev_data["anomaly_detected"] is True
        assert dev_data["anomaly_type"] == "route_deviation"
        assert dev_data["prompt_sent"] is True

        # Check 'Are you okay?' push was recorded
        assert any("Are you okay?" in p["title"] for p in notifier.sent_pushes)

        # Wait for unanswered prompt timer to expire (1.2s > 1s)
        await asyncio.sleep(1.2)

        # Verify auto-escalation: trip status becomes sos_triggered
        trip = await pg_db.get(Trip, uuid.UUID(trip_id))
        assert trip.status == TripStatus.SOS_TRIGGERED

        # Verify SOS event was created in DB
        sos_stmt = select(SOSEvent).where(SOSEvent.trip_id == uuid.UUID(trip_id))
        sos_res = await pg_db.execute(sos_stmt)
        sos_event = sos_res.scalar_one_or_none()
        assert sos_event is not None
        assert "route_deviation" in sos_event.trigger_type

    async def test_unusual_stop_in_high_risk_segment_auto_escalates(
        self, trip_client, auth_headers, sample_route_geojson, monkeypatch, pg_db,
    ):
        """
        Simulation 2: Stationary longer than N minutes in a high-risk area.
        Triggers 'Are you okay?' and auto-escalates.
        """
        monkeypatch.setattr(settings, "TRIP_ARE_YOU_OKAY_SECONDS", 1)
        monkeypatch.setattr(settings, "SOS_COUNTDOWN_SECONDS", 0)
        monkeypatch.setattr(settings, "TRIP_UNUSUAL_STOP_MINUTES", 2.0)

        notifier = get_mock_notification_service()
        notifier.clear()
        r = get_redis()
        if r:
            r.flushdb()

        now = datetime.now(timezone.utc)
        # Start trip 10 minutes ago with low safety score (high risk = 0.75)
        start_res = await trip_client.post(
            "/api/v1/trips/start",
            json={
                "origin_lat": 13.0250,
                "origin_lon": 77.5150,
                "destination_lat": 13.0280,
                "destination_lon": 77.5180,
                "route_geojson": sample_route_geojson,
                "safety_score": 25.0,
                "started_at": (now - timedelta(minutes=10)).isoformat(),
            },
            headers=auth_headers,
        )
        trip_id = start_res.json()["trip_id"]

        # Location 1: 3 minutes ago at the same spot
        await trip_client.post(
            f"/api/v1/trips/{trip_id}/location",
            json={
                "lat": 13.0255,
                "lon": 77.5155,
                "accuracy_m": 5.0,
                "timestamp": (now - timedelta(minutes=3)).isoformat(),
            },
            headers=auth_headers,
        )

        # Location 2: Now, still at the same spot (stationary for 3 minutes > 2 min threshold)
        stop_res = await trip_client.post(
            f"/api/v1/trips/{trip_id}/location",
            json={
                "lat": 13.0255,
                "lon": 77.5155,
                "accuracy_m": 5.0,
                "timestamp": now.isoformat(),
            },
            headers=auth_headers,
        )
        assert stop_res.status_code == 200
        stop_data = stop_res.json()
        assert stop_data["anomaly_detected"] is True
        assert stop_data["anomaly_type"] == "unusual_stop"
        assert stop_data["prompt_sent"] is True

        # Wait for auto-escalation
        await asyncio.sleep(1.2)

        trip = await pg_db.get(Trip, uuid.UUID(trip_id))
        assert trip.status == TripStatus.SOS_TRIGGERED

    async def test_motion_anomaly_phone_drop_auto_escalates(
        self, trip_client, auth_headers, sample_route_geojson, monkeypatch, pg_db,
    ):
        """
        Simulation 3: Abnormal motion (phone drop / violent struggle impact).
        Triggers 'Are you okay?' and auto-escalates.
        """
        monkeypatch.setattr(settings, "TRIP_ARE_YOU_OKAY_SECONDS", 1)
        monkeypatch.setattr(settings, "SOS_COUNTDOWN_SECONDS", 0)

        notifier = get_mock_notification_service()
        notifier.clear()
        r = get_redis()
        if r:
            r.flushdb()

        start_res = await trip_client.post(
            "/api/v1/trips/start",
            json={
                "origin_lat": 13.0250,
                "origin_lon": 77.5150,
                "destination_lat": 13.0280,
                "destination_lon": 77.5180,
                "route_geojson": sample_route_geojson,
            },
            headers=auth_headers,
        )
        trip_id = start_res.json()["trip_id"]

        # Ping on route but with high impact phone drop motion window
        motion_res = await trip_client.post(
            f"/api/v1/trips/{trip_id}/location",
            json={
                "lat": 13.0255,
                "lon": 77.5155,
                "accuracy_m": 5.0,
                "motion_window": generate_phone_drop_window(),
            },
            headers=auth_headers,
        )
        assert motion_res.status_code == 200
        m_data = motion_res.json()
        assert m_data["anomaly_detected"] is True
        assert "motion_" in m_data["anomaly_type"]
        assert m_data["prompt_sent"] is True

        # Wait for auto-escalation
        await asyncio.sleep(1.2)

        trip = await pg_db.get(Trip, uuid.UUID(trip_id))
        assert trip.status == TripStatus.SOS_TRIGGERED


@pytest.mark.asyncio
class TestAnomalyResponseAndDismissal:
    async def test_respond_im_okay_cancels_escalation(
        self, trip_client, auth_headers, sample_route_geojson, monkeypatch, pg_db,
    ):
        """
        POST /api/v1/trips/{id}/respond dismisses prompt and cancels SOS auto-escalation.
        """
        monkeypatch.setattr(settings, "TRIP_ARE_YOU_OKAY_SECONDS", 2)
        monkeypatch.setattr(settings, "SOS_COUNTDOWN_SECONDS", 0)

        r = get_redis()
        if r:
            r.flushdb()

        start_res = await trip_client.post(
            "/api/v1/trips/start",
            json={
                "origin_lat": 13.0250,
                "origin_lon": 77.5150,
                "destination_lat": 13.0280,
                "destination_lon": 77.5180,
                "route_geojson": sample_route_geojson,
            },
            headers=auth_headers,
        )
        trip_id = start_res.json()["trip_id"]

        # Trigger deviation anomaly
        await trip_client.post(
            f"/api/v1/trips/{trip_id}/location",
            json={"lat": 13.0450, "lon": 77.5350, "accuracy_m": 5.0},
            headers=auth_headers,
        )

        # Confirm Redis key was created
        if r:
            assert r.get(f"trip:are_you_okay:{trip_id}") is not None

        # User responds "I'm okay" inside the window
        resp = await trip_client.post(
            f"/api/v1/trips/{trip_id}/respond",
            json={"message": "I took a detour to a convenience store, I'm okay."},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["escalation_cancelled"] is True

        # Confirm Redis key is deleted
        if r:
            assert r.get(f"trip:are_you_okay:{trip_id}") is None

        # Wait past the timer duration
        await asyncio.sleep(2.2)

        # Confirm trip is NOT escalated to SOS (remains active)
        trip = await pg_db.get(Trip, uuid.UUID(trip_id))
        assert trip.status == TripStatus.ACTIVE

        sos_stmt = select(SOSEvent).where(SOSEvent.trip_id == uuid.UUID(trip_id))
        sos_res = await pg_db.execute(sos_stmt)
        assert sos_res.scalar_one_or_none() is None


@pytest.mark.asyncio
class TestFeedbackAndLearning:
    async def test_feedback_submission_and_ema_weight_adjustment(
        self, trip_client, auth_headers, sample_route_geojson, pg_db,
    ):
        """
        POST /api/v1/trips/{id}/feedback stores feedback and updates segment risk EMA.
        """
        r = get_redis()
        if r:
            r.flushdb()

        # Route geojson with explicit segment IDs in properties
        route_with_segs = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": [[77.515, 13.025], [77.516, 13.026]]},
                    "properties": {"segment_id": 999111, "osm_id": 999111},
                }
            ],
        }

        start_res = await trip_client.post(
            "/api/v1/trips/start",
            json={
                "origin_lat": 13.0250,
                "origin_lon": 77.5150,
                "destination_lat": 13.0260,
                "destination_lon": 77.5160,
                "route_geojson": route_with_segs,
            },
            headers=auth_headers,
        )
        trip_id = start_res.json()["trip_id"]

        # Submit low safety feedback (1 star -> risk 1.0)
        fb_res = await trip_client.post(
            f"/api/v1/trips/{trip_id}/feedback",
            json={
                "safety_rating": 1,
                "route_rating": 2,
                "comment": "Very dark alley behind factory wall with no lights.",
            },
            headers=auth_headers,
        )
        assert fb_res.status_code == 200
        fb_data = fb_res.json()
        assert fb_data["safety_rating"] == 1
        assert "feedback_id" in fb_data

        # Verify Feedback row in PostgreSQL
        fb_row = await pg_db.get(Feedback, uuid.UUID(fb_data["feedback_id"]))
        assert fb_row is not None
        assert fb_row.safety_rating == 1

        # Verify Redis EMA weight update for segment 999111
        if r:
            ema_val = r.get("segment:ema_risk:999111")
            assert ema_val is not None
            # Default 0.5 smoothed towards 1.0: 0.8*0.5 + 0.2*1.0 = 0.60
            assert float(ema_val) > 0.55


@pytest.mark.asyncio
class TestTripPrivacyAndRetention:
    async def test_privacy_location_sharing_access_control(
        self, trip_client, auth_headers, stranger_headers, trusted_headers, sample_route_geojson,
    ):
        """
        Live trip location is accessible ONLY to the owner and registered trusted contacts.
        Strangers receive 403 Forbidden.
        """
        start_res = await trip_client.post(
            "/api/v1/trips/start",
            json={
                "origin_lat": 13.0250,
                "origin_lon": 77.5150,
                "destination_lat": 13.0280,
                "destination_lon": 77.5180,
                "route_geojson": sample_route_geojson,
            },
            headers=auth_headers,
        )
        trip_id = start_res.json()["trip_id"]

        # 1. Owner can view live location
        owner_res = await trip_client.get(f"/api/v1/trips/{trip_id}/live", headers=auth_headers)
        assert owner_res.status_code == 200
        assert len(owner_res.json()["trail"]) >= 1

        # 2. Registered trusted contact can view live location
        trusted_res = await trip_client.get(f"/api/v1/trips/{trip_id}/live", headers=trusted_headers)
        assert trusted_res.status_code == 200

        # 3. Unrelated stranger receives 403 Forbidden
        stranger_res = await trip_client.get(f"/api/v1/trips/{trip_id}/live", headers=stranger_headers)
        assert stranger_res.status_code == 403

    async def test_privacy_purge_old_locations(
        self, trip_client, auth_headers, sample_route_geojson, pg_db,
    ):
        """
        Old location pings older than retention period are deleted.
        """
        start_res = await trip_client.post(
            "/api/v1/trips/start",
            json={
                "origin_lat": 13.0250,
                "origin_lon": 77.5150,
                "destination_lat": 13.0280,
                "destination_lon": 77.5180,
                "route_geojson": sample_route_geojson,
            },
            headers=auth_headers,
        )
        trip_id = uuid.UUID(start_res.json()["trip_id"])

        # Insert an old location 10 days ago
        from geoalchemy2.elements import WKTElement
        old_loc = TripLocation(
            trip_id=trip_id,
            geometry=WKTElement("POINT(77.515 13.025)", srid=4326),
            accuracy_m=5.0,
            recorded_at=datetime.now(timezone.utc) - timedelta(days=10),
        )
        pg_db.add(old_loc)
        await pg_db.commit()

        # Run purge endpoint with 7-day retention
        purge_res = await trip_client.post(
            "/api/v1/trips/purge-locations?days=7",
            headers=auth_headers,
        )
        assert purge_res.status_code == 200
        assert purge_res.json()["deleted_location_records"] >= 1
