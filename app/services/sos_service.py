"""
app/services/sos_service.py - Core SOS orchestration service.

Handles:
- Shake validation and SOS trigger with 5-second countdown
- Anti-false-alarm cancellation within the countdown window
- Background auto-dispatch on countdown expiry:
    (a) SMS with signed live-tracking link to trusted contacts
    (b) Push + location to verified volunteers within 1 km (PostGIS radius query)
    (c) Emergency alert to Police/112 endpoint (with 1091/1098 helpline references)
    (d) Evidence capture flag for mobile client
- Idempotency within 30 seconds
- Immutable audit logging for every state change
- Public signed tracking tokens and WebSocket live broadcast manager
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import logging
import os
from pathlib import Path
from typing import Any, Sequence
import uuid

from fastapi import HTTPException, UploadFile, WebSocket, WebSocketDisconnect, status
import jwt
from sqlalchemy import cast, func, select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from geoalchemy2 import Geography
from geoalchemy2.elements import WKTElement
from geoalchemy2.shape import to_shape
from shapely.geometry import Point

from app.config import settings
from app.deps import async_session_factory, get_redis
from app.ml.shake_validator import validate_shake
from app.models.audit import AuditLog
from app.models.sos import EvidenceFile, SOSEvent, SOSLocation, SOSStatus
from app.models.user import User
from app.models.volunteer import Volunteer
from app.schemas.sos import (
    EvidenceUploadResponse,
    LocationPoint,
    SOSCancelResponse,
    SOSLocationUpdate,
    SOSResolveResponse,
    SOSStatusResponse,
    SOSTrackResponse,
    SOSTriggerRequest,
    SOSTriggerResponse,
)
from app.services.notification_service import (
    STANDARD_HELPLINES,
    get_notification_service,
)

logger = logging.getLogger(__name__)

# Directory where evidence files (audio/video/images) are stored
EVIDENCE_DIR = Path("data/evidence")


# ── WebSocket Manager ────────────────────────────────────────────────────────

class SOSConnectionManager:
    """Manages active WebSockets for live SOS tracking broadcasts."""

    def __init__(self) -> None:
        self.active_connections: dict[uuid.UUID, set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def connect(self, sos_id: uuid.UUID, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.active_connections[sos_id].add(websocket)
        logger.info("WebSocket connected for SOS %s (total: %d)", sos_id, len(self.active_connections[sos_id]))

    async def disconnect(self, sos_id: uuid.UUID, websocket: WebSocket) -> None:
        async with self._lock:
            self.active_connections[sos_id].discard(websocket)
            if not self.active_connections[sos_id]:
                self.active_connections.pop(sos_id, None)
        logger.info("WebSocket disconnected for SOS %s", sos_id)

    async def broadcast(self, sos_id: uuid.UUID, message: dict[str, Any]) -> None:
        async with self._lock:
            sockets = list(self.active_connections.get(sos_id, []))

        stale: list[WebSocket] = []
        for ws in sockets:
            try:
                await ws.send_json(message)
            except Exception:
                stale.append(ws)

        if stale:
            async with self._lock:
                for ws in stale:
                    self.active_connections[sos_id].discard(ws)


ws_manager = SOSConnectionManager()


# ── Signed Tracking Token ───────────────────────────────────────────────────

def create_tracking_token(sos_id: uuid.UUID, expires_in_seconds: int = 86400) -> str:
    """Create a signed, time-limited JWT for public live tracking."""
    expire = datetime.now(timezone.utc) + timedelta(seconds=expires_in_seconds)
    payload = {
        "sub": str(sos_id),
        "purpose": "sos_tracking",
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_tracking_token(token: str) -> uuid.UUID:
    """Verify and decode a tracking token, returning the underlying sos_id."""
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
        )
        if payload.get("purpose") != "sos_tracking":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token purpose.",
            )
        return uuid.UUID(payload["sub"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tracking token has expired.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid tracking token: {exc}",
        )


# ── Geometry Coordinate Helpers ─────────────────────────────────────────────

def _coords_from_geom(geom: Any) -> tuple[float, float]:
    """Extract (lat, lon) from GeoAlchemy geometry element or WKT."""
    if geom is None:
        return 0.0, 0.0
    try:
        shape = to_shape(geom)
        if isinstance(shape, Point):
            return float(shape.y), float(shape.x)
    except Exception:
        pass
    # Fallback string parse
    raw = str(geom)
    if "POINT" in raw:
        parts = raw.split("(")[-1].rstrip(")").split()
        if len(parts) >= 2:
            return float(parts[1]), float(parts[0])
    return 0.0, 0.0


# ── Audit Logging Helper ────────────────────────────────────────────────────

async def _log_audit(
    db: AsyncSession,
    user_id: uuid.UUID | None,
    action: str,
    resource_id: str,
    details: dict[str, Any] | None = None,
    ip_address: str | None = None,
) -> None:
    """Record an immutable audit log entry."""
    entry = AuditLog(
        user_id=user_id,
        action=action,
        resource_type="sos_event",
        resource_id=resource_id,
        details=details or {},
        ip_address=ip_address,
    )
    db.add(entry)


# ── SOS Core Operations ─────────────────────────────────────────────────────

async def trigger_sos(
    db: AsyncSession,
    current_user: User,
    req: SOSTriggerRequest,
    ip_address: str | None = None,
    countdown_seconds: int | None = None,
) -> SOSTriggerResponse:
    """
    Trigger an SOS event with 5-second countdown and 30-second idempotency.
    """
    countdown_sec = (
        countdown_seconds
        if countdown_seconds is not None
        else settings.SOS_COUNTDOWN_SECONDS
    )
    redis = get_redis()
    idempotency_key = f"sos:idempotency:{current_user.id}"

    # 1. Idempotency Check (Redis 30-second TTL)
    if redis:
        try:
            cached_id = redis.get(idempotency_key)
            if cached_id:
                event = await db.get(SOSEvent, uuid.UUID(cached_id))
                if event and event.status in (SOSStatus.COUNTDOWN, SOSStatus.ACTIVE):
                    logger.info(
                        "Idempotent SOS trigger for user %s within 30s. Returning existing event %s",
                        current_user.id,
                        event.id,
                    )
                    token = create_tracking_token(event.id)
                    return SOSTriggerResponse(
                        sos_id=event.id,
                        status=event.status.value,
                        countdown_seconds=0,
                        trigger_type=event.trigger_type,
                        confidence=event.confidence,
                        triggered_at=event.triggered_at,
                        tracking_token=token,
                        tracking_url=f"{settings.API_V1_PREFIX}/track/{token}",
                        evidence_capture_requested=(event.status == SOSStatus.ACTIVE),
                        is_duplicate=True,
                    )
        except Exception as exc:
            logger.warning("Redis idempotency check warning: %s", exc)

    # 2. Shake validation (NEVER blocks SOS if failing or absent)
    confidence = 1.0
    if req.trigger_type == "shake":
        confidence = validate_shake(req.accel_window)
    else:
        confidence = 1.0

    # 3. Create SOSEvent (safely check trip_id if provided)
    valid_trip_id = None
    if req.trip_id:
        try:
            from app.models.trip import Trip
            trip = await db.get(Trip, req.trip_id)
            if trip:
                valid_trip_id = trip.id
            else:
                logger.info("Trip %s not found in DB; continuing with trip_id=None", req.trip_id)
        except Exception as trip_err:
            logger.warning("Trip check failed (%s); continuing with trip_id=None", trip_err)

    now = req.timestamp or datetime.now(timezone.utc)
    event = SOSEvent(
        user_id=current_user.id,
        trip_id=valid_trip_id,
        status=SOSStatus.COUNTDOWN,
        trigger_type=req.trigger_type,
        confidence=confidence,
        triggered_at=now,
        notes=req.notes,
    )
    db.add(event)
    await db.flush()

    # 4. Record initial location
    wkt_geom = WKTElement(f"POINT({req.lon} {req.lat})", srid=4326)
    initial_location = SOSLocation(
        sos_event_id=event.id,
        geometry=wkt_geom,
        accuracy_m=10.0,
        recorded_at=now,
    )
    db.add(initial_location)

    # 5. Audit log
    await _log_audit(
        db=db,
        user_id=current_user.id,
        action="sos_trigger",
        resource_id=str(event.id),
        details={
            "trigger_type": req.trigger_type,
            "confidence": confidence,
            "lat": req.lat,
            "lon": req.lon,
            "status": SOSStatus.COUNTDOWN.value,
        },
        ip_address=ip_address,
    )
    await db.commit()

    # 6. Set Redis idempotency (30s) and countdown (5s)
    if redis:
        try:
            redis.setex(idempotency_key, 30, str(event.id))
            redis.setex(f"sos:countdown:{event.id}", countdown_sec, "countdown")
            # Store initial coordinates for fast lookup
            redis.set(f"sos:{event.id}:lat", str(req.lat))
            redis.set(f"sos:{event.id}:lon", str(req.lon))
        except Exception as exc:
            logger.warning("Redis setup error during SOS trigger: %s", exc)

    # 7. Start non-blocking asyncio countdown task
    asyncio.create_task(run_countdown_and_dispatch(event.id, countdown_sec))

    token = create_tracking_token(event.id)
    return SOSTriggerResponse(
        sos_id=event.id,
        status=SOSStatus.COUNTDOWN.value,
        countdown_seconds=countdown_sec,
        trigger_type=event.trigger_type,
        confidence=event.confidence,
        triggered_at=event.triggered_at,
        tracking_token=token,
        tracking_url=f"{settings.API_V1_PREFIX}/track/{token}",
        evidence_capture_requested=False,
        is_duplicate=False,
    )


async def run_countdown_and_dispatch(
    event_id: uuid.UUID,
    countdown_seconds: float,
) -> None:
    """
    Non-blocking background countdown:
    Wait for countdown_seconds. If the event is still COUNTDOWN (not cancelled),
    transition to ACTIVE and run silent multi-channel dispatch in parallel.
    """
    try:
        await asyncio.sleep(countdown_seconds)
        await dispatch_sos(event_id)
    except Exception as exc:
        logger.error("Error during SOS countdown/dispatch for event %s: %s", event_id, exc, exc_info=True)


async def dispatch_sos(event_id: uuid.UUID) -> None:
    """
    Execute silent dispatch when countdown completes without cancellation:
      (a) SMS with live tracking link to trusted contacts
      (b) Push + location to opted-in verified volunteers within 1 km
      (c) Alert to police / 112 endpoint with 1091/1098 helpline references
      (d) Flag evidence capture active
    """
    async with async_session_factory() as db:
        event = await db.get(SOSEvent, event_id)
        if not event:
            logger.warning("Cannot dispatch SOS: event %s not found", event_id)
            return

        if event.status != SOSStatus.COUNTDOWN:
            logger.info(
                "SOS %s is in status '%s'; skipping dispatch.",
                event_id,
                event.status.value,
            )
            return

        # Transition status to ACTIVE
        event.status = SOSStatus.ACTIVE
        await _log_audit(
            db=db,
            user_id=event.user_id,
            action="sos_dispatched",
            resource_id=str(event.id),
            details={"status": SOSStatus.ACTIVE.value},
        )
        await db.commit()

        # Flag evidence capture in Redis
        redis = get_redis()
        if redis:
            try:
                redis.set(f"sos:{event.id}:evidence_capture", "true")
            except Exception:
                pass

        # Broadcast status transition to any connected tracking WebSockets
        await ws_manager.broadcast(
            event.id,
            {
                "type": "status_change",
                "status": SOSStatus.ACTIVE.value,
                "evidence_capture_requested": True,
            },
        )

        # Retrieve user details & trusted contacts
        user_query = (
            select(User)
            .options(selectinload(User.trusted_contacts))
            .where(User.id == event.user_id)
        )
        user_res = await db.execute(user_query)
        user = user_res.scalar_one_or_none()
        user_name = user.name if user else "SafeWay User"
        user_phone = user.phone if user else "Unknown"

        # Latest location extraction
        loc_res = await db.execute(
            select(SOSLocation)
            .where(SOSLocation.sos_event_id == event.id)
            .order_by(SOSLocation.recorded_at.desc())
            .limit(1)
        )
        latest_loc = loc_res.scalar_one_or_none()
        event_lat, event_lon = _coords_from_geom(latest_loc.geometry if latest_loc else None)

        tracking_token = create_tracking_token(event.id)
        tracking_url = f"{settings.API_V1_PREFIX}/track/{tracking_token}"


        notification_service = get_notification_service()

        # ── (a) SMS to trusted contacts with live-tracking link ───────────────
        sms_tasks = []
        if user and user.trusted_contacts:
            sms_body = (
                f"EMERGENCY ALERT: {user_name} triggered an SOS! "
                f"Track live: {tracking_url}"
            )
            for contact in user.trusted_contacts:
                sms_tasks.append(
                    notification_service.send_sms(contact.contact_phone, sms_body)
                )

        # ── (b) Push to verified volunteers within 1 km (PostGIS) ─────────────
        volunteers: Sequence[Volunteer] = []
        try:
            vol_query = (
                select(Volunteer)
                .where(
                    Volunteer.is_available.is_(True),
                    Volunteer.verified.is_(True),
                    Volunteer.last_location.is_not(None),
                    func.ST_DWithin(
                        cast(Volunteer.last_location, Geography),
                        cast(
                            func.ST_SetSRID(func.ST_MakePoint(event_lon, event_lat), 4326),
                            Geography,
                        ),
                        settings.VOLUNTEER_RADIUS_M,
                    ),
                )
            )
            vol_res = await db.execute(vol_query)
            volunteers = vol_res.scalars().all()
        except Exception as vol_err:
            logger.warning("Spatial volunteer query exception: %s. Falling back to available list.", vol_err)
            try:
                fallback_res = await db.execute(
                    select(Volunteer).where(
                        Volunteer.is_available.is_(True),
                        Volunteer.verified.is_(True),
                    )
                )
                volunteers = fallback_res.scalars().all()
            except Exception:
                volunteers = []

        push_tasks = []
        for vol in volunteers:
            push_tasks.append(
                notification_service.send_push(
                    recipient_id=str(vol.user_id),
                    title="🚨 Emergency SOS Alert Nearby",
                    body=f"Assistance needed within 1 km. Open SafeWay to respond.",
                    data={
                        "sos_id": str(event.id),
                        "lat": event_lat,
                        "lon": event_lon,
                        "tracking_url": tracking_url,
                    },
                )
            )

        # ── (c) Alert to Police / 112 with helpline references ────────────────
        police_meta = {
            "helplines": STANDARD_HELPLINES,
            "trigger_type": event.trigger_type,
            "confidence": event.confidence,
            "tracking_url": tracking_url,
        }
        police_task = notification_service.alert_police(
            event_id=str(event.id),
            lat=event_lat,
            lon=event_lon,
            user_name=user_name,
            phone=user_phone,
            metadata=police_meta,
        )

        # Run all notifications concurrently
        all_tasks = sms_tasks + push_tasks + [police_task]
        await asyncio.gather(*all_tasks, return_exceptions=True)
        logger.info("SOS %s auto-dispatch complete.", event_id)


async def cancel_sos(
    db: AsyncSession,
    sos_id: uuid.UUID,
    user_id: uuid.UUID,
    reason: str = "false_alarm",
    ip_address: str | None = None,
) -> SOSCancelResponse:
    """
    Cancel an SOS during the 5-second countdown window to prevent false alarms.
    """
    event = await db.get(SOSEvent, sos_id)
    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="SOS event not found.",
        )

    if event.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to cancel this SOS event.",
        )

    if event.status == SOSStatus.CANCELLED:
        return SOSCancelResponse(
            sos_id=event.id,
            status=SOSStatus.CANCELLED.value,
            message="SOS is already cancelled.",
        )

    if event.status != SOSStatus.COUNTDOWN:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel SOS in status '{event.status.value}'. Countdown has expired.",
        )

    event.status = SOSStatus.CANCELLED
    await _log_audit(
        db=db,
        user_id=user_id,
        action="sos_cancelled",
        resource_id=str(event.id),
        details={"reason": reason, "status": SOSStatus.CANCELLED.value},
        ip_address=ip_address,
    )
    await db.commit()

    redis = get_redis()
    if redis:
        try:
            redis.delete(f"sos:countdown:{event.id}")
        except Exception:
            pass

    # Broadcast cancellation over WebSocket
    await ws_manager.broadcast(
        event.id,
        {
            "type": "status_change",
            "status": SOSStatus.CANCELLED.value,
            "message": "SOS cancelled by user.",
        },
    )

    return SOSCancelResponse(
        sos_id=event.id,
        status=SOSStatus.CANCELLED.value,
        message="SOS cancelled successfully.",
    )


async def get_sos_status(
    db: AsyncSession,
    sos_id: uuid.UUID,
    user_id: uuid.UUID,
) -> SOSStatusResponse:
    """
    Owner-only status query for an SOS event.
    Returns status, timestamps, trigger_type, confidence, and tracking_url.
    """
    event = await db.get(SOSEvent, sos_id)
    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="SOS event not found.",
        )

    if event.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this SOS event.",
        )

    token = create_tracking_token(event.id)
    tracking_url = f"{settings.API_V1_PREFIX}/track/{token}"

    evidence_req = False
    redis = get_redis()
    if redis:
        try:
            evidence_req = redis.get(f"sos:{event.id}:evidence_capture") == "true"
        except Exception:
            pass

    if event.status == SOSStatus.ACTIVE:
        evidence_req = True

    return SOSStatusResponse(
        sos_id=event.id,
        status=event.status.value,
        triggered_at=event.triggered_at,
        resolved_at=event.resolved_at,
        trigger_type=event.trigger_type,
        confidence=event.confidence,
        tracking_url=tracking_url,
        tracking_token=token,
        evidence_capture_requested=evidence_req,
        notes=event.notes,
    )


async def update_sos_location(
    db: AsyncSession,
    sos_id: uuid.UUID,
    user_id: uuid.UUID,
    payload: SOSLocationUpdate,
) -> dict[str, Any]:
    """
    Append mobile location update and broadcast over WebSocket to live tracking clients.
    """
    event = await db.get(SOSEvent, sos_id)
    if not event:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="SOS event not found.")

    if event.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized.")

    if event.status not in (SOSStatus.COUNTDOWN, SOSStatus.ACTIVE):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot record location for SOS in status '{event.status.value}'.",
        )

    now = payload.timestamp or datetime.now(timezone.utc)
    geom = WKTElement(f"POINT({payload.lon} {payload.lat})", srid=4326)
    loc = SOSLocation(
        sos_event_id=event.id,
        geometry=geom,
        accuracy_m=payload.accuracy_m,
        recorded_at=now,
    )
    db.add(loc)
    await db.commit()

    # Cache battery in Redis
    redis = get_redis()
    if redis:
        try:
            if payload.battery_level is not None:
                redis.set(f"sos:{event.id}:battery", str(payload.battery_level))
            redis.set(f"sos:{event.id}:lat", str(payload.lat))
            redis.set(f"sos:{event.id}:lon", str(payload.lon))
        except Exception:
            pass

    # Broadcast location update to active WebSocket listeners
    broadcast_payload = {
        "type": "location_update",
        "lat": payload.lat,
        "lon": payload.lon,
        "accuracy_m": payload.accuracy_m,
        "battery_level": payload.battery_level,
        "recorded_at": now.isoformat(),
    }
    await ws_manager.broadcast(event.id, broadcast_payload)

    return {"status": "recorded", "location": broadcast_payload}


async def upload_evidence(
    db: AsyncSession,
    sos_id: uuid.UUID,
    user_id: uuid.UUID,
    file: UploadFile,
    file_type: str,
    duration_s: float | None = None,
    ip_address: str | None = None,
) -> EvidenceUploadResponse:
    """
    Upload and register audio, video, or photo evidence for an SOS event.
    """
    event = await db.get(SOSEvent, sos_id)
    if not event:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="SOS event not found.")

    if event.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized.")

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    clean_filename = Path(file.filename or "evidence.bin").name
    dest_filename = f"{event.id}_{uuid.uuid4().hex[:8]}_{clean_filename}"
    dest_path = EVIDENCE_DIR / dest_filename

    contents = await file.read()
    with open(dest_path, "wb") as f:
        f.write(contents)

    evidence = EvidenceFile(
        sos_event_id=event.id,
        file_type=file_type,
        file_path=str(dest_path),
        duration_s=duration_s,
    )
    db.add(evidence)
    await db.flush()

    await _log_audit(
        db=db,
        user_id=user_id,
        action="sos_evidence_uploaded",
        resource_id=str(event.id),
        details={"file_id": str(evidence.id), "file_type": file_type, "file_path": str(dest_path)},
        ip_address=ip_address,
    )
    await db.commit()

    return EvidenceUploadResponse.model_validate(evidence)


async def resolve_sos(
    db: AsyncSession,
    sos_id: uuid.UUID,
    user_id: uuid.UUID,
    notes: str | None = None,
    ip_address: str | None = None,
) -> SOSResolveResponse:
    """
    Mark SOS safe and resolved; notify contacts that user is safe.
    """
    event = await db.get(SOSEvent, sos_id)
    if not event:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="SOS event not found.")

    if event.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized.")

    if event.status == SOSStatus.RESOLVED:
        return SOSResolveResponse(
            sos_id=event.id,
            status=SOSStatus.RESOLVED.value,
            resolved_at=event.resolved_at or datetime.now(timezone.utc),
            message="SOS is already resolved.",
        )

    now = datetime.now(timezone.utc)
    event.status = SOSStatus.RESOLVED
    event.resolved_at = now
    if notes:
        event.notes = f"{event.notes or ''} [Resolved]: {notes}".strip()

    await _log_audit(
        db=db,
        user_id=user_id,
        action="sos_resolved",
        resource_id=str(event.id),
        details={"status": SOSStatus.RESOLVED.value, "resolved_at": now.isoformat()},
        ip_address=ip_address,
    )
    await db.commit()

    # Broadcast resolution to WebSocket
    await ws_manager.broadcast(
        event.id,
        {
            "type": "status_change",
            "status": SOSStatus.RESOLVED.value,
            "resolved_at": now.isoformat(),
            "message": "User marked themselves safe.",
        },
    )

    # Notify trusted contacts
    user_res = await db.execute(
        select(User).options(selectinload(User.trusted_contacts)).where(User.id == user_id)
    )
    user = user_res.scalar_one_or_none()
    if user and user.trusted_contacts:
        ns = get_notification_service()
        safe_msg = f"SAFE ALERT: {user.name} has marked themselves SAFE. The SOS emergency has ended."
        tasks = [ns.send_sms(contact.contact_phone, safe_msg) for contact in user.trusted_contacts]
        await asyncio.gather(*tasks, return_exceptions=True)

    return SOSResolveResponse(
        sos_id=event.id,
        status=SOSStatus.RESOLVED.value,
        resolved_at=now,
        message="SOS marked safe and resolved successfully.",
    )


async def get_tracking_data(
    db: AsyncSession,
    sos_id: uuid.UUID,
) -> SOSTrackResponse:
    """
    Fetch comprehensive tracking state for public signed token endpoint.
    """
    event_query = (
        select(SOSEvent)
        .options(selectinload(SOSEvent.locations))
        .where(SOSEvent.id == sos_id)
    )
    event_res = await db.execute(event_query)
    event = event_res.scalar_one_or_none()
    if not event:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="SOS event not found.")

    user_res = await db.execute(select(User).where(User.id == event.user_id))
    user = user_res.scalar_one_or_none()
    user_name = user.name if user else "SafeWay User"
    user_phone = user.phone if user else "Unknown"

    trail: list[LocationPoint] = []
    # Sort locations chronologically
    sorted_locs = sorted(event.locations, key=lambda l: l.recorded_at)
    for loc in sorted_locs:
        lat, lon = _coords_from_geom(loc.geometry)
        trail.append(
            LocationPoint(
                lat=lat,
                lon=lon,
                accuracy_m=loc.accuracy_m,
                recorded_at=loc.recorded_at,
            )
        )

    latest = trail[-1] if trail else None

    # Retrieve battery and evidence flag from Redis
    battery_val: float | None = None
    evidence_req = False
    redis = get_redis()
    if redis:
        try:
            bat_str = redis.get(f"sos:{event.id}:battery")
            if bat_str:
                battery_val = float(bat_str)
            evidence_req = redis.get(f"sos:{event.id}:evidence_capture") == "true"
        except Exception:
            pass

    if event.status == SOSStatus.ACTIVE:
        evidence_req = True

    return SOSTrackResponse(
        sos_id=event.id,
        status=event.status.value,
        user_name=user_name,
        user_phone=user_phone,
        trigger_type=event.trigger_type,
        confidence=event.confidence,
        triggered_at=event.triggered_at,
        resolved_at=event.resolved_at,
        latest_location=latest,
        battery_level=battery_val,
        route_trail=trail,
        evidence_capture_requested=evidence_req,
    )
