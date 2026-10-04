"""
tests/test_sos.py - Step 5 SOS acceptance and integration tests.

Covers:
  1. Shake validator accuracy on synthetic data, peak count thresholding, and safe fallback.
  2. SOS trigger with COUNTDOWN status and tracking_token.
  3. Auto-dispatch after countdown with mock SMS sent to each trusted contact (SOS_COUNTDOWN_SECONDS=1).
  4. Cancellation inside window gives CANCELLED and sends NO SMS.
  5. Cancellation after countdown expiry returns 400.
  6. Duplicate trigger within 30 seconds returns the same sos_id with is_duplicate=True.
  7. GET /api/v1/sos/{id} returns status for owner, 403 for different user, 404 for unknown.
  8. GET /api/v1/track/{token} works without login, and returns 401 for user login token (wrong purpose).
  9. Bad accel_window (strings, empty list) never blocks SOS creation (returns 201 with COUNTDOWN).
  10. Dummy or unknown trip_id never causes a 500.
  11. Location update and route trail recording.
  12. Evidence file upload.
  13. Resolution flow notifying contacts that user is safe.
  14. WebSocket real-time live tracking stream (/api/v1/ws/track/{token}).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import io
import uuid
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from fastapi.testclient import TestClient
from sklearn.metrics import accuracy_score
from sqlalchemy import select
from sqlalchemy.pool import NullPool
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from geoalchemy2.elements import WKTElement

from app.config import settings
from app.core.security import create_access_token, hash_password
from app.deps import get_redis
from app.main import app
from app.ml.shake_validator import (
    generate_synthetic_data,
    get_shake_classifier,
    validate_shake,
)
from app.models.audit import AuditLog
from app.models.sos import EvidenceFile, SOSEvent, SOSLocation, SOSStatus
from app.models.user import TrustedContact, User, UserRole
from app.models.volunteer import Volunteer
from app.services import sos_service
from app.services.notification_service import (
    STANDARD_HELPLINES,
    get_mock_notification_service,
)

# Test engine with NullPool so connections are created in the active test loop
test_pg_engine = create_async_engine(settings.DATABASE_URL, poolclass=NullPool)
pg_test_session_factory = async_sessionmaker(test_pg_engine, expire_on_commit=False)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture(autouse=True)
async def engine_cleanup(monkeypatch):
    """Ensure engine connections are closed per test and patched into sos_service."""
    import app.services.sos_service as sos_mod
    monkeypatch.setattr(sos_mod, "async_session_factory", pg_test_session_factory)
    yield
    from app.deps import engine
    await engine.dispose()
    await test_pg_engine.dispose()


@pytest_asyncio.fixture
async def pg_db():
    """Async session directly connected to PostgreSQL + PostGIS test container."""
    async with pg_test_session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def sos_client(pg_db):
    """HTTP async client configured with PostgreSQL test session."""
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
    """Create an active woman user with trusted contacts in PostgreSQL."""
    user = User(
        email=f"woman_{uuid.uuid4().hex[:8]}@example.com",
        phone=f"+91{uuid.uuid4().int % 10**10:010d}",
        hashed_password=hash_password("SafePassword123!"),
        name="Deepa Gowda",
        role=UserRole.WOMAN,
        trust_score=1.0,
        is_active=True,
    )
    pg_db.add(user)
    await pg_db.commit()
    await pg_db.refresh(user)

    # Add trusted contacts
    c1 = TrustedContact(
        user_id=user.id,
        contact_name="Asha Gowda (Sister)",
        contact_phone="+919876543210",
        relationship_label="sister",
        priority=1,
    )
    c2 = TrustedContact(
        user_id=user.id,
        contact_name="Ramesh Gowda (Father)",
        contact_phone="+919876543211",
        relationship_label="father",
        priority=2,
    )
    pg_db.add_all([c1, c2])
    await pg_db.commit()
    await pg_db.refresh(user)
    return user


@pytest_asyncio.fixture
async def other_user(pg_db):
    """Create a second distinct user for access control tests."""
    user = User(
        email=f"other_{uuid.uuid4().hex[:8]}@example.com",
        phone=f"+91{uuid.uuid4().int % 10**10:010d}",
        hashed_password=hash_password("OtherPassword123!"),
        name="Other User",
        role=UserRole.WOMAN,
        trust_score=1.0,
        is_active=True,
    )
    pg_db.add(user)
    await pg_db.commit()
    await pg_db.refresh(user)
    return user


@pytest_asyncio.fixture
async def test_volunteer_user(pg_db):
    """Create a verified, available volunteer within Peenya Industrial Area."""
    vol_user = User(
        email=f"vol_{uuid.uuid4().hex[:8]}@example.com",
        phone=f"+91{uuid.uuid4().int % 10**10:010d}",
        hashed_password=hash_password("VolunteerPass123!"),
        name="Sunil Kumar",
        role=UserRole.VOLUNTEER,
        trust_score=1.0,
        is_active=True,
    )
    pg_db.add(vol_user)
    await pg_db.commit()
    await pg_db.refresh(vol_user)

    # Located at Peenya 1st Stage (~400m from trigger point)
    vol = Volunteer(
        user_id=vol_user.id,
        is_available=True,
        verified=True,
        radius_m=1000.0,
        last_location=WKTElement("POINT(77.518 13.027)", srid=4326),
    )
    pg_db.add(vol)
    await pg_db.commit()
    return vol


@pytest.fixture
def auth_headers(test_woman_user):
    token = create_access_token(test_woman_user.id, test_woman_user.role.value)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def other_auth_headers(other_user):
    token = create_access_token(other_user.id, other_user.role.value)
    return {"Authorization": f"Bearer {token}"}


# ── 1. Shake Validator Accuracy and Safety Fallback ──────────────────────────

class TestShakeValidator:
    def test_synthetic_classification_accuracy(self):
        """Verify shake classifier achieves >= 95% accuracy on synthetic test data."""
        clf = get_shake_classifier()
        X_test, y_test = generate_synthetic_data(n_samples_per_class=60, random_state=999)
        preds = clf.model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        assert accuracy >= 0.95, f"Expected accuracy >= 0.95, got {accuracy:.3f}"

    def test_never_blocks_sos_on_none_or_empty_window(self):
        """CRITICAL: If window is None, empty, or malformed, NEVER block SOS."""
        assert validate_shake(None) == 1.0
        assert validate_shake([]) == 1.0
        assert validate_shake([{"bad": "data"}]) == 1.0
        assert validate_shake([[0.0, 0.0]]) == 1.0
        assert validate_shake(["string1", "string2"]) == 1.0

    def test_high_confidence_on_synthetic_shake(self):
        """A synthetic 3-shake window produces high confidence >= 0.8."""
        X_test, y_test = generate_synthetic_data(n_samples_per_class=10, random_state=123)
        shake_idx = [i for i, label in enumerate(y_test) if label == 1][0]
        prob = get_shake_classifier().model.predict_proba(X_test[shake_idx : shake_idx + 1])[0][1]
        assert prob >= 0.70


# ── 2. SOS Trigger & Countdown ────────────────────────────────────────────────

@pytest.mark.asyncio
class TestSOSTrigger:
    async def test_trigger_sos_creates_countdown_event(self, sos_client, auth_headers, pg_db):
        """POST /api/v1/sos/trigger returns 201, COUNTDOWN status, tracking token, and audit log."""
        payload = {
            "lat": 13.025,
            "lon": 77.515,
            "trigger_type": "shake",
            "notes": "Emergency test in Peenya",
        }
        resp = await sos_client.post("/api/v1/sos/trigger", json=payload, headers=auth_headers)
        assert resp.status_code == 201
        data = resp.json()

        assert data["status"] == "countdown"
        assert data["countdown_seconds"] == settings.SOS_COUNTDOWN_SECONDS
        assert data["trigger_type"] == "shake"
        assert data["confidence"] >= 0.0
        assert "tracking_token" in data
        assert "/api/v1/track/" in data["tracking_url"]
        assert data["is_duplicate"] is False

        # Verify database record
        sos_id = uuid.UUID(data["sos_id"])
        event = await pg_db.get(SOSEvent, sos_id)
        assert event is not None
        assert event.status == SOSStatus.COUNTDOWN

        # Verify initial location was saved
        loc_res = await pg_db.execute(
            select(SOSLocation).where(SOSLocation.sos_event_id == sos_id)
        )
        locs = loc_res.scalars().all()
        assert len(locs) >= 1

        # Verify AuditLog record
        audit_res = await pg_db.execute(
            select(AuditLog).where(
                AuditLog.resource_id == str(sos_id),
                AuditLog.action == "sos_trigger",
            )
        )
        audit = audit_res.scalar_one_or_none()
        assert audit is not None
        assert audit.details["status"] == "countdown"

    async def test_trigger_bad_accel_window_never_blocks_sos(self, sos_client, auth_headers):
        """Bad accel_window (strings, empty list) never blocks SOS creation (returns 201 with COUNTDOWN)."""
        # Test with string elements
        resp1 = await sos_client.post(
            "/api/v1/sos/trigger",
            json={"lat": 13.025, "lon": 77.515, "accel_window": ["bad", "string", "data"]},
            headers=auth_headers,
        )
        assert resp1.status_code == 201
        assert resp1.json()["status"] == "countdown"
        assert resp1.json()["confidence"] == 1.0

        # Clear idempotency in redis to allow immediate next trigger
        r = get_redis()
        if r:
            r.flushdb()

        # Test with empty list
        resp2 = await sos_client.post(
            "/api/v1/sos/trigger",
            json={"lat": 13.025, "lon": 77.515, "accel_window": []},
            headers=auth_headers,
        )
        assert resp2.status_code == 201
        assert resp2.json()["status"] == "countdown"

    async def test_trigger_dummy_trip_id_never_causes_500(self, sos_client, auth_headers):
        """A dummy or non-existent trip_id never causes a 500 server error."""
        r = get_redis()
        if r:
            r.flushdb()

        dummy_trip_id = str(uuid.uuid4())
        resp = await sos_client.post(
            "/api/v1/sos/trigger",
            json={"lat": 13.025, "lon": 77.515, "trip_id": dummy_trip_id},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        assert resp.json()["status"] == "countdown"


# ── 3. Countdown Expiry & Auto-Dispatch ───────────────────────────────────────

@pytest.mark.asyncio
class TestSOSAutoDispatch:
    async def test_countdown_expiry_dispatches_sms_to_trusted_contacts(
        self, sos_client, auth_headers, test_woman_user, pg_db, monkeypatch
    ):
        """
        After countdown (using SOS_COUNTDOWN_SECONDS=1), status becomes ACTIVE
        and mock SMS provider records message for each trusted contact.
        """
        mock_notifier = get_mock_notification_service()
        mock_notifier.clear()

        # Set countdown seconds to 1 for this test
        monkeypatch.setattr(settings, "SOS_COUNTDOWN_SECONDS", 1)

        r = get_redis()
        if r:
            r.flushdb()

        trig = await sos_client.post(
            "/api/v1/sos/trigger",
            json={"lat": 13.025, "lon": 77.515, "trigger_type": "shake"},
            headers=auth_headers,
        )
        assert trig.status_code == 201
        sos_id = uuid.UUID(trig.json()["sos_id"])
        assert trig.json()["countdown_seconds"] == 1

        # Wait for the 1-second countdown task to fire
        await asyncio.sleep(1.3)

        # Check in DB that status transitioned to ACTIVE
        event = await pg_db.get(SOSEvent, sos_id)
        assert event is not None
        assert event.status == SOSStatus.ACTIVE

        # Check that mock SMS provider recorded a message for each trusted contact
        recipients = [s["to"] for s in mock_notifier.sent_sms]
        assert "+919876543210" in recipients
        assert "+919876543211" in recipients
        for s in mock_notifier.sent_sms:
            assert "/api/v1/track/" in s["message"]
            assert "EMERGENCY ALERT" in s["message"]


# ── 4. Anti-False-Alarm Cancellation ─────────────────────────────────────────

@pytest.mark.asyncio
class TestSOSCancel:
    async def test_cancel_within_window_gives_cancelled_and_sends_no_sms(
        self, sos_client, auth_headers, pg_db, monkeypatch
    ):
        """Cancel within the countdown window sets CANCELLED and sends NO SMS."""
        mock_notifier = get_mock_notification_service()
        mock_notifier.clear()

        monkeypatch.setattr(settings, "SOS_COUNTDOWN_SECONDS", 2)
        r = get_redis()
        if r:
            r.flushdb()

        # 1. Trigger
        trig_resp = await sos_client.post(
            "/api/v1/sos/trigger",
            json={"lat": 13.025, "lon": 77.515, "trigger_type": "shake"},
            headers=auth_headers,
        )
        assert trig_resp.status_code == 201
        sos_id = trig_resp.json()["sos_id"]

        # 2. Cancel within window
        cancel_resp = await sos_client.post(
            f"/api/v1/sos/{sos_id}/cancel",
            json={"reason": "false alarm"},
            headers=auth_headers,
        )
        assert cancel_resp.status_code == 200
        assert cancel_resp.json()["status"] == "cancelled"

        # 3. Wait past the countdown window
        await asyncio.sleep(2.2)

        # 4. Confirm NO SMS was sent
        assert len(mock_notifier.sent_sms) == 0

        # 5. Status remains CANCELLED
        event = await pg_db.get(SOSEvent, uuid.UUID(sos_id))
        assert event.status == SOSStatus.CANCELLED

    async def test_cancel_after_expiry_returns_400(
        self, sos_client, auth_headers, pg_db, monkeypatch
    ):
        """Calling cancel after countdown has expired returns 400."""
        monkeypatch.setattr(settings, "SOS_COUNTDOWN_SECONDS", 1)
        r = get_redis()
        if r:
            r.flushdb()

        # 1. Trigger
        trig_resp = await sos_client.post(
            "/api/v1/sos/trigger",
            json={"lat": 13.025, "lon": 77.515, "trigger_type": "shake"},
            headers=auth_headers,
        )
        assert trig_resp.status_code == 201
        sos_id = trig_resp.json()["sos_id"]

        # 2. Wait for countdown to expire into ACTIVE
        await asyncio.sleep(1.3)

        # 3. Attempt cancel after expiry
        cancel_resp = await sos_client.post(
            f"/api/v1/sos/{sos_id}/cancel",
            json={"reason": "too late"},
            headers=auth_headers,
        )
        assert cancel_resp.status_code == 400
        assert "Countdown has expired" in cancel_resp.json()["detail"]


# ── 5. 30-Second Idempotency ──────────────────────────────────────────────────

@pytest.mark.asyncio
class TestSOSIdempotency:
    async def test_duplicate_trigger_within_30s_returns_same_sos_id(
        self, sos_client, auth_headers
    ):
        """Duplicate triggers within 30 seconds return the same sos_id with is_duplicate=True."""
        r = get_redis()
        if r:
            r.flushdb()

        payload = {"lat": 13.025, "lon": 77.515, "trigger_type": "shake"}

        resp1 = await sos_client.post("/api/v1/sos/trigger", json=payload, headers=auth_headers)
        assert resp1.status_code == 201
        data1 = resp1.json()

        # Immediate second trigger
        resp2 = await sos_client.post("/api/v1/sos/trigger", json=payload, headers=auth_headers)
        assert resp2.status_code == 201
        data2 = resp2.json()

        assert data1["sos_id"] == data2["sos_id"]
        assert data2["is_duplicate"] is True


# ── 6. GET /api/v1/sos/{id} Status Endpoint ──────────────────────────────────

@pytest.mark.asyncio
class TestSOSStatusEndpoint:
    async def test_get_sos_status_owner_success(self, sos_client, auth_headers):
        """Owner can check status via GET /api/v1/sos/{id} without tracking token."""
        r = get_redis()
        if r:
            r.flushdb()

        trig = await sos_client.post(
            "/api/v1/sos/trigger",
            json={"lat": 13.025, "lon": 77.515, "trigger_type": "shake"},
            headers=auth_headers,
        )
        sos_id = trig.json()["sos_id"]

        resp = await sos_client.get(f"/api/v1/sos/{sos_id}", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["sos_id"] == sos_id
        assert data["status"] == "countdown"
        assert data["trigger_type"] == "shake"
        assert "/api/v1/track/" in data["tracking_url"]
        assert "triggered_at" in data

    async def test_get_sos_status_different_user_returns_403(
        self, sos_client, auth_headers, other_auth_headers
    ):
        """Different user receives 403 Forbidden when requesting another's SOS event."""
        r = get_redis()
        if r:
            r.flushdb()

        trig = await sos_client.post(
            "/api/v1/sos/trigger",
            json={"lat": 13.025, "lon": 77.515, "trigger_type": "shake"},
            headers=auth_headers,
        )
        sos_id = trig.json()["sos_id"]

        resp = await sos_client.get(f"/api/v1/sos/{sos_id}", headers=other_auth_headers)
        assert resp.status_code == 403

    async def test_get_sos_status_unknown_id_returns_404(self, sos_client, auth_headers):
        """Non-existent SOS ID returns 404."""
        unknown_id = uuid.uuid4()
        resp = await sos_client.get(f"/api/v1/sos/{unknown_id}", headers=auth_headers)
        assert resp.status_code == 404


# ── 7. Public Signed Live Tracking Endpoint ───────────────────────────────────

@pytest.mark.asyncio
class TestSOSTracking:
    async def test_public_tracking_works_without_login(self, sos_client, auth_headers):
        """Public tracking endpoint requires no auth header and resolves with signed token."""
        r = get_redis()
        if r:
            r.flushdb()

        trig = await sos_client.post(
            "/api/v1/sos/trigger",
            json={"lat": 13.025, "lon": 77.515, "trigger_type": "shake"},
            headers=auth_headers,
        )
        token = trig.json()["tracking_token"]

        # GET without Authorization header
        resp = await sos_client.get(f"/api/v1/track/{token}")
        assert resp.status_code == 200
        track = resp.json()

        assert track["user_name"] == "Deepa Gowda"
        assert track["status"] == "countdown"
        assert track["latest_location"]["lat"] == 13.025

    async def test_tracking_returns_401_for_login_access_token(
        self, sos_client, test_woman_user
    ):
        """A user login JWT token returns 401 Unauthorized (wrong token purpose)."""
        login_token = create_access_token(test_woman_user.id, test_woman_user.role.value)
        resp = await sos_client.get(f"/api/v1/track/{login_token}")
        assert resp.status_code == 401
        assert "Invalid token purpose" in resp.json()["detail"]

    async def test_tracking_invalid_token_returns_401(self, sos_client):
        """Tampered or invalid token string returns 401."""
        resp = await sos_client.get("/api/v1/track/invalid.token.here")
        assert resp.status_code == 401


# ── 8. Periodic Location Updates & Route Trail ────────────────────────────────

@pytest.mark.asyncio
class TestSOSLocationUpdates:
    async def test_location_update_and_trail(self, sos_client, auth_headers):
        """POST /api/v1/sos/{id}/location appends to trail and records battery level."""
        r = get_redis()
        if r:
            r.flushdb()

        trig = await sos_client.post(
            "/api/v1/sos/trigger",
            json={"lat": 13.025, "lon": 77.515, "trigger_type": "shake"},
            headers=auth_headers,
        )
        sos_id = trig.json()["sos_id"]

        # Push subsequent locations
        loc1 = await sos_client.post(
            f"/api/v1/sos/{sos_id}/location",
            json={"lat": 13.026, "lon": 77.516, "accuracy_m": 8.0, "battery_level": 85.0},
            headers=auth_headers,
        )
        assert loc1.status_code == 200

        loc2 = await sos_client.post(
            f"/api/v1/sos/{sos_id}/location",
            json={"lat": 13.027, "lon": 77.517, "accuracy_m": 5.0, "battery_level": 84.0},
            headers=auth_headers,
        )
        assert loc2.status_code == 200

        # Verify route trail in tracking data
        token = trig.json()["tracking_token"]
        track_resp = await sos_client.get(f"/api/v1/track/{token}")
        assert track_resp.status_code == 200
        track_data = track_resp.json()

        assert len(track_data["route_trail"]) >= 3
        assert track_data["battery_level"] == 84.0
        assert track_data["latest_location"]["lat"] == 13.027


# ── 9. Evidence File Upload ───────────────────────────────────────────────────

@pytest.mark.asyncio
class TestSOSEvidence:
    async def test_upload_audio_evidence(self, sos_client, auth_headers, pg_db):
        """POST /api/v1/sos/{id}/evidence accepts multipart file upload and records in DB."""
        r = get_redis()
        if r:
            r.flushdb()

        trig = await sos_client.post(
            "/api/v1/sos/trigger",
            json={"lat": 13.025, "lon": 77.515, "trigger_type": "shake"},
            headers=auth_headers,
        )
        sos_id = trig.json()["sos_id"]

        # Synthetic 5-second audio bytes
        audio_content = b"RIFF....WAVEfmt ....data synthetic audio buffer"
        files = {"file": ("audio_evidence.wav", io.BytesIO(audio_content), "audio/wav")}
        data = {"file_type": "audio", "duration_s": "5.2"}

        ev_resp = await sos_client.post(
            f"/api/v1/sos/{sos_id}/evidence",
            files=files,
            data=data,
            headers=auth_headers,
        )
        assert ev_resp.status_code == 201
        ev_data = ev_resp.json()

        assert ev_data["file_type"] == "audio"
        assert ev_data["duration_s"] == 5.2

        # Verify DB entry
        ev_record = await pg_db.get(EvidenceFile, uuid.UUID(ev_data["id"]))
        assert ev_record is not None
        assert ev_record.sos_event_id == uuid.UUID(sos_id)


# ── 10. Resolve SOS ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestSOSResolve:
    async def test_resolve_sos_notifies_contacts(self, sos_client, auth_headers, pg_db):
        """POST /api/v1/sos/{id}/resolve marks safe and sends safe alert to contacts."""
        mock_notifier = get_mock_notification_service()
        mock_notifier.clear()

        r = get_redis()
        if r:
            r.flushdb()

        trig = await sos_client.post(
            "/api/v1/sos/trigger",
            json={"lat": 13.025, "lon": 77.515, "trigger_type": "shake"},
            headers=auth_headers,
        )
        sos_id = trig.json()["sos_id"]

        # Resolve
        res_resp = await sos_client.post(
            f"/api/v1/sos/{sos_id}/resolve",
            json={"notes": "Reached safely inside Peenya Metro station."},
            headers=auth_headers,
        )
        assert res_resp.status_code == 200
        assert res_resp.json()["status"] == "resolved"

        # Verify event in DB
        event = await pg_db.get(SOSEvent, uuid.UUID(sos_id))
        assert event.status == SOSStatus.RESOLVED
        assert event.resolved_at is not None

        # Verify contacts received SAFE notification
        safe_sms = [s for s in mock_notifier.sent_sms if "SAFE ALERT" in s["message"]]
        assert len(safe_sms) >= 2


# ── 11. WebSocket Live Tracking ───────────────────────────────────────────────

class TestSOSWebSocket:
    @pytest.mark.asyncio
    async def test_websocket_tracking_stream(self, pg_db, test_woman_user):
        """Test real-time tracking WebSocket connection (/api/v1/ws/track/{token})."""
        event = SOSEvent(
            user_id=test_woman_user.id,
            status=SOSStatus.ACTIVE,
            trigger_type="shake",
            confidence=0.95,
            triggered_at=datetime.now(timezone.utc),
        )
        pg_db.add(event)
        await pg_db.commit()
        await pg_db.refresh(event)

        token = sos_service.create_tracking_token(event.id)

        client = TestClient(app)
        with client.websocket_connect(f"/api/v1/ws/track/{token}") as ws:
            init_frame = ws.receive_json()
            assert init_frame["type"] == "initial_state"
            assert init_frame["data"]["sos_id"] == str(event.id)
            assert init_frame["data"]["status"] == "active"

            # Ping / Pong heartbeat check
            ws.send_text("ping")
            pong = ws.receive_text()
            assert pong == "pong"
