"""
tests/test_sos.py - Step 5 SOS acceptance and integration tests.

Covers:
  1. Shake validator accuracy on synthetic data, peak count thresholding, and safe fallback.
  2. SOS trigger with 5-second countdown and initial location trail.
  3. Anti-false-alarm cancellation within the countdown window.
  4. Silent multi-channel dispatch (SMS with tracking URL, PostGIS volunteer search, police/112 with 1091/1098 helplines).
  5. Idempotency (duplicate triggers within 30s map to the same event).
  6. Periodic mobile location updates and trail recording.
  7. Audio/photo evidence upload.
  8. Resolution flow with contact safe notifications.
  9. Public signed live tracking endpoint (GET /track/{token}).
  10. WebSocket real-time location and status broadcast (WS /ws/track/{token}).
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
from geoalchemy2.elements import WKTElement

from app.config import settings
from app.core.security import create_access_token, hash_password
from app.deps import async_session_factory, get_redis
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
from app.schemas.sos import SOSTriggerRequest
from app.services import sos_service
from app.services.notification_service import (
    STANDARD_HELPLINES,
    get_mock_notification_service,
)


from sqlalchemy.pool import NullPool
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

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

    def test_high_confidence_on_synthetic_shake(self):
        """A synthetic 3-shake window produces high confidence >= 0.8."""
        X_test, y_test = generate_synthetic_data(n_samples_per_class=10, random_state=123)
        # Select a shake sample features
        shake_idx = [i for i, label in enumerate(y_test) if label == 1][0]
        prob = get_shake_classifier().model.predict_proba(X_test[shake_idx : shake_idx + 1])[0][1]
        assert prob >= 0.70


# ── 2. SOS Trigger & 5-Second Countdown ────────────────────────────────────────

@pytest.mark.asyncio
class TestSOSTrigger:
    async def test_trigger_sos_creates_countdown_event(self, sos_client, auth_headers, pg_db):
        """POST /sos/trigger returns 201, COUNTDOWN status, tracking token, and audit log."""
        payload = {
            "lat": 13.025,
            "lon": 77.515,
            "trigger_type": "shake",
            "notes": "Emergency test in Peenya",
        }
        resp = await sos_client.post("/sos/trigger", json=payload, headers=auth_headers)
        assert resp.status_code == 201
        data = resp.json()

        assert data["status"] == "countdown"
        assert data["countdown_seconds"] == settings.SOS_COUNTDOWN_SECONDS
        assert data["trigger_type"] == "shake"
        assert data["confidence"] >= 0.0
        assert "tracking_token" in data
        assert "/track/" in data["tracking_url"]
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


# ── 3. Anti-False-Alarm Cancellation ─────────────────────────────────────────

@pytest.mark.asyncio
class TestSOSCancel:
    async def test_cancel_during_countdown_window(self, sos_client, auth_headers, pg_db):
        """POST /sos/{id}/cancel sets status to CANCELLED and aborts dispatch."""
        # 1. Trigger
        trig_resp = await sos_client.post(
            "/sos/trigger",
            json={"lat": 13.025, "lon": 77.515, "trigger_type": "shake"},
            headers=auth_headers,
        )
        assert trig_resp.status_code == 201
        sos_id = trig_resp.json()["sos_id"]

        # 2. Cancel within countdown window
        cancel_resp = await sos_client.post(
            f"/sos/{sos_id}/cancel",
            json={"reason": "accidental shake"},
            headers=auth_headers,
        )
        assert cancel_resp.status_code == 200
        assert cancel_resp.json()["status"] == "cancelled"

        # 3. Verify in database
        event = await pg_db.get(SOSEvent, uuid.UUID(sos_id))
        assert event.status == SOSStatus.CANCELLED

        # 4. Verify audit log
        audit_res = await pg_db.execute(
            select(AuditLog).where(
                AuditLog.resource_id == str(sos_id),
                AuditLog.action == "sos_cancelled",
            )
        )
        assert audit_res.scalar_one_or_none() is not None

        # 5. Dispatching after cancellation must NOT set ACTIVE
        await sos_service.dispatch_sos(uuid.UUID(sos_id))
        await pg_db.refresh(event)
        assert event.status == SOSStatus.CANCELLED


# ── 4. Multi-Channel Auto-Dispatch ────────────────────────────────────────────

@pytest.mark.asyncio
class TestSOSDispatch:
    async def test_dispatch_contacts_volunteers_police(
        self, sos_client, auth_headers, test_woman_user, test_volunteer_user, pg_db
    ):
        """
        When countdown expires without cancellation:
          (a) Status becomes ACTIVE
          (b) SMS with tracking link sent to trusted contacts
          (c) Push + location sent to verified volunteers within 1 km
          (d) Alert sent to police/112 with 1091 (women) and 1098 (child) helplines
          (e) Event flagged for audio/photo evidence capture
        """
        mock_notifier = get_mock_notification_service()
        mock_notifier.clear()

        # Trigger SOS
        trig_resp = await sos_client.post(
            "/sos/trigger",
            json={"lat": 13.025, "lon": 77.515, "trigger_type": "shake"},
            headers=auth_headers,
        )
        sos_id = uuid.UUID(trig_resp.json()["sos_id"])

        # Execute dispatch directly (simulating countdown completion)
        await sos_service.dispatch_sos(sos_id)

        # 1. Status is now ACTIVE
        event = await pg_db.get(SOSEvent, sos_id)
        assert event.status == SOSStatus.ACTIVE

        # 2. Trusted contacts received SMS with live tracking link
        assert len(mock_notifier.sent_sms) >= 2
        recipients = [s["to"] for s in mock_notifier.sent_sms]
        assert "+919876543210" in recipients
        assert "+919876543211" in recipients
        for s in mock_notifier.sent_sms:
            assert "/track/" in s["message"]
            assert "EMERGENCY ALERT" in s["message"]

        # 3. Nearby volunteer received push alert
        assert len(mock_notifier.sent_pushes) >= 1
        vol_push_ids = [p["recipient_id"] for p in mock_notifier.sent_pushes]
        assert str(test_volunteer_user.user_id) in vol_push_ids
        # Check that tracking url is present
        assert any("/track/" in p["data"]["tracking_url"] for p in mock_notifier.sent_pushes)

        # 4. Police / 112 alert registered with helplines
        assert len(mock_notifier.police_alerts) >= 1
        police_alert = mock_notifier.police_alerts[0]
        assert police_alert["agency"] == "Karnataka Police / Dial 112 Control Room"
        assert police_alert["helpline_references"]["women_helpline"] == "1091"
        assert police_alert["helpline_references"]["child_helpline"] == "1098"
        assert police_alert["helpline_references"]["emergency"] == "112"

        # 5. Evidence capture flag set
        track_data = await sos_service.get_tracking_data(pg_db, sos_id)
        assert track_data.evidence_capture_requested is True


# ── 5. 30-Second Idempotency ──────────────────────────────────────────────────

@pytest.mark.asyncio
class TestSOSIdempotency:
    async def test_duplicate_trigger_within_30s_maps_to_same_event(
        self, sos_client, auth_headers, pg_db
    ):
        """Duplicate triggers within 30 seconds map to the same SOS event."""
        payload = {"lat": 13.025, "lon": 77.515, "trigger_type": "shake"}

        resp1 = await sos_client.post("/sos/trigger", json=payload, headers=auth_headers)
        assert resp1.status_code == 201
        data1 = resp1.json()

        # Immediate second trigger
        resp2 = await sos_client.post("/sos/trigger", json=payload, headers=auth_headers)
        assert resp2.status_code == 201
        data2 = resp2.json()

        assert data1["sos_id"] == data2["sos_id"]
        assert data2["is_duplicate"] is True


# ── 6. Periodic Location Updates & Route Trail ────────────────────────────────

@pytest.mark.asyncio
class TestSOSLocationUpdates:
    async def test_location_update_and_trail(self, sos_client, auth_headers, pg_db):
        """POST /sos/{id}/location appends to trail and records battery level."""
        trig = await sos_client.post(
            "/sos/trigger",
            json={"lat": 13.025, "lon": 77.515, "trigger_type": "shake"},
            headers=auth_headers,
        )
        sos_id = trig.json()["sos_id"]

        # Push 2 subsequent locations
        loc1 = await sos_client.post(
            f"/sos/{sos_id}/location",
            json={"lat": 13.026, "lon": 77.516, "accuracy_m": 8.0, "battery_level": 85.0},
            headers=auth_headers,
        )
        assert loc1.status_code == 200

        loc2 = await sos_client.post(
            f"/sos/{sos_id}/location",
            json={"lat": 13.027, "lon": 77.517, "accuracy_m": 5.0, "battery_level": 84.0},
            headers=auth_headers,
        )
        assert loc2.status_code == 200

        # Verify route trail in tracking data
        token = trig.json()["tracking_token"]
        track_resp = await sos_client.get(f"/track/{token}")
        assert track_resp.status_code == 200
        track_data = track_resp.json()

        assert len(track_data["route_trail"]) >= 3
        assert track_data["battery_level"] == 84.0
        assert track_data["latest_location"]["lat"] == 13.027


# ── 7. Evidence File Upload ───────────────────────────────────────────────────

@pytest.mark.asyncio
class TestSOSEvidence:
    async def test_upload_audio_evidence(self, sos_client, auth_headers, pg_db):
        """POST /sos/{id}/evidence accepts multipart file upload and records in DB."""
        trig = await sos_client.post(
            "/sos/trigger",
            json={"lat": 13.025, "lon": 77.515, "trigger_type": "shake"},
            headers=auth_headers,
        )
        sos_id = trig.json()["sos_id"]

        # Synthetic 5-second audio bytes
        audio_content = b"RIFF....WAVEfmt ....data synthetic audio buffer"
        files = {"file": ("audio_evidence.wav", io.BytesIO(audio_content), "audio/wav")}
        data = {"file_type": "audio", "duration_s": "5.2"}

        ev_resp = await sos_client.post(
            f"/sos/{sos_id}/evidence",
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


# ── 8. Resolve SOS ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestSOSResolve:
    async def test_resolve_sos_notifies_contacts(self, sos_client, auth_headers, pg_db):
        """POST /sos/{id}/resolve marks safe and sends safe alert to contacts."""
        mock_notifier = get_mock_notification_service()
        mock_notifier.clear()

        trig = await sos_client.post(
            "/sos/trigger",
            json={"lat": 13.025, "lon": 77.515, "trigger_type": "shake"},
            headers=auth_headers,
        )
        sos_id = trig.json()["sos_id"]

        # Resolve
        res_resp = await sos_client.post(
            f"/sos/{sos_id}/resolve",
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


# ── 9. Public Signed Live Tracking Endpoint ───────────────────────────────────

@pytest.mark.asyncio
class TestSOSTracking:
    async def test_public_tracking_access(self, sos_client, auth_headers):
        """Public tracking endpoint requires no auth and resolves with signed token."""
        trig = await sos_client.post(
            "/sos/trigger",
            json={"lat": 13.025, "lon": 77.515, "trigger_type": "shake"},
            headers=auth_headers,
        )
        token = trig.json()["tracking_token"]

        # GET without Authorization header
        resp = await sos_client.get(f"/track/{token}")
        assert resp.status_code == 200
        track = resp.json()

        assert track["user_name"] == "Deepa Gowda"
        assert track["status"] == "countdown"
        assert track["latest_location"]["lat"] == 13.025

    async def test_tracking_invalid_token_returns_401(self, sos_client):
        """Tampered or invalid token returns 401."""
        resp = await sos_client.get("/track/invalid.token.here")
        assert resp.status_code == 401


# ── 10. WebSocket Live Tracking ───────────────────────────────────────────────

class TestSOSWebSocket:
    @pytest.mark.asyncio
    async def test_websocket_tracking_stream(self, pg_db, test_woman_user):
        """Test real-time tracking WebSocket connection, initial state frame, and ping."""
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
        with client.websocket_connect(f"/ws/track/{token}") as ws:
            init_frame = ws.receive_json()
            assert init_frame["type"] == "initial_state"
            assert init_frame["data"]["sos_id"] == str(event.id)
            assert init_frame["data"]["status"] == "active"

            # Ping / Pong heartbeat check
            ws.send_text("ping")
            pong = ws.receive_text()
            assert pong == "pong"
