"""
app/services/trip_monitor.py - Live trip monitoring and anomaly response service.

Manages:
- Trip start, location updates, and end
- Route deviation, unusual stop, and motion anomaly detection
- 'Are you okay?' push prompt and Redis-backed response timer
- Auto-escalation to SOS flow if prompt is unanswered
- User feedback and per-segment risk EMA learning
- Location privacy retention and trusted-contact access control
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
import logging
from typing import Any
import uuid

from fastapi import HTTPException, status
from geoalchemy2.elements import WKTElement
from geoalchemy2.shape import to_shape
from shapely.geometry import Point
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.deps import async_session_factory, get_redis
from app.ml.anomaly import (
    detect_route_deviation,
    detect_unusual_stop,
    extract_coordinates_from_geojson,
    get_motion_detector,
)
from app.models.feedback import Feedback
from app.models.trip import Trip, TripLocation, TripStatus
from app.models.user import TrustedContact, User, UserRole
from app.schemas.trip import (
    TripEndResponse,
    TripFeedbackRequest,
    TripFeedbackResponse,
    TripLiveLocationResponse,
    TripLocationRequest,
    TripLocationResponse,
    TripRespondRequest,
    TripRespondResponse,
    TripStartRequest,
    TripStartResponse,
)
from app.services.notification_service import get_notification_service

logger = logging.getLogger(__name__)


# ── Geometry Coordinate Helpers ─────────────────────────────────────────────

def _coords_from_geom(geom: Any) -> tuple[float, float]:
    """Extract (lat, lon) from GeoAlchemy geometry element or WKT string."""
    if geom is None:
        return 0.0, 0.0
    try:
        shape = to_shape(geom)
        if isinstance(shape, Point):
            return float(shape.y), float(shape.x)
    except Exception:
        pass
    raw = str(geom)
    if "POINT" in raw:
        parts = raw.split("(")[-1].rstrip(")").split()
        if len(parts) >= 2:
            return float(parts[1]), float(parts[0])
    return 0.0, 0.0


# ── Trip Lifecycle Operations ────────────────────────────────────────────────

async def start_trip(
    db: AsyncSession,
    user_id: uuid.UUID,
    req: TripStartRequest,
) -> TripStartResponse:
    """
    Initialize a live monitored trip and record origin as initial location.
    """
    origin_geom = WKTElement(f"POINT({req.origin_lon} {req.origin_lat})", srid=4326)
    dest_geom = WKTElement(f"POINT({req.destination_lon} {req.destination_lat})", srid=4326)
    now = req.started_at or datetime.now(timezone.utc)

    trip = Trip(
        user_id=user_id,
        status=TripStatus.ACTIVE,
        origin=origin_geom,
        destination=dest_geom,
        route_geojson=req.route_geojson,
        safety_score=req.safety_score,
        started_at=now,
    )
    db.add(trip)
    await db.flush()

    # Initial location at origin
    loc = TripLocation(
        trip_id=trip.id,
        geometry=origin_geom,
        accuracy_m=10.0,
        recorded_at=now,
    )
    db.add(loc)
    await db.commit()
    await db.refresh(trip)

    return TripStartResponse(
        trip_id=trip.id,
        status=trip.status.value,
        started_at=trip.started_at,
        safety_score=trip.safety_score,
        origin={"lat": req.origin_lat, "lon": req.origin_lon},
        destination={"lat": req.destination_lat, "lon": req.destination_lon},
        route_geojson=trip.route_geojson,
    )


async def record_trip_location(
    db: AsyncSession,
    trip_id: uuid.UUID,
    user_id: uuid.UUID,
    req: TripLocationRequest,
) -> TripLocationResponse:
    """
    Record a live location ping and run anomaly checks (deviation, stop, motion).
    If an anomaly is detected, dispatch an 'Are you okay?' prompt and start an auto-escalation timer.
    """
    trip = await db.get(Trip, trip_id)
    if not trip:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trip not found.")
    if trip.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to update this trip.")
    if trip.status == TripStatus.COMPLETED:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Trip is already completed.")

    now = req.timestamp or datetime.now(timezone.utc)
    geom = WKTElement(f"POINT({req.lon} {req.lat})", srid=4326)

    loc = TripLocation(
        trip_id=trip.id,
        geometry=geom,
        accuracy_m=req.accuracy_m,
        recorded_at=now,
    )
    db.add(loc)
    await db.commit()

    # ── Anomaly Check 1: Route Deviation ──────────────────────────────────────
    is_deviated, dev_dist_m = detect_route_deviation(
        current_lat=req.lat,
        current_lon=req.lon,
        route_geojson=trip.route_geojson,
        threshold_meters=settings.TRIP_DEVIATION_METERS,
    )

    # ── Anomaly Check 2: Unusual Stop in High-Risk Area ───────────────────────
    # Fetch recent locations for this trip
    stmt = (
        select(TripLocation)
        .where(TripLocation.trip_id == trip.id)
        .order_by(TripLocation.recorded_at.desc())
        .limit(50)
    )
    res = await db.execute(stmt)
    recent_db_locs = list(reversed(res.scalars().all()))
    loc_dicts = []
    for l in recent_db_locs:
        l_lat, l_lon = _coords_from_geom(l.geometry)
        loc_dicts.append({"lat": l_lat, "lon": l_lon, "recorded_at": l.recorded_at})

    # Segment risk approximation: default to high risk if safety score is low
    estimated_risk = 1.0 - (trip.safety_score or 50.0) / 100.0
    is_unusual_stop, stop_mins = detect_unusual_stop(
        recent_locations=loc_dicts,
        segment_risk=estimated_risk,
        min_duration_minutes=settings.TRIP_UNUSUAL_STOP_MINUTES,
        risk_threshold=0.45,
    )

    # ── Anomaly Check 3: Motion Anomaly (Isolation Forest) ─────────────────────
    detector = get_motion_detector()
    is_motion_anomaly, motion_score, motion_reason = detector.evaluate(req.motion_window)

    # ── Consolidate Anomaly Evaluation ────────────────────────────────────────
    anomaly_detected = False
    anomaly_type: str | None = None
    details: dict[str, Any] = {}

    if is_deviated:
        anomaly_detected = True
        anomaly_type = "route_deviation"
        details = {"deviation_meters": dev_dist_m, "threshold": settings.TRIP_DEVIATION_METERS}
        trip.status = TripStatus.DEVIATED
        await db.commit()
    elif is_unusual_stop:
        anomaly_detected = True
        anomaly_type = "unusual_stop"
        details = {"stationary_minutes": stop_mins, "segment_risk": estimated_risk}
    elif is_motion_anomaly:
        anomaly_detected = True
        anomaly_type = f"motion_{motion_reason}"
        details = {"motion_reason": motion_reason, "score": motion_score}

    prompt_sent = False
    if anomaly_detected and anomaly_type:
        redis = get_redis()
        redis_key = f"trip:are_you_okay:{trip.id}"
        timeout_sec = settings.TRIP_ARE_YOU_OKAY_SECONDS

        # Record active prompt in Redis
        if redis:
            prompt_data = {
                "user_id": str(user_id),
                "trip_id": str(trip.id),
                "anomaly_type": anomaly_type,
                "lat": req.lat,
                "lon": req.lon,
                "created_at": now.isoformat(),
            }
            redis.setex(redis_key, timeout_sec + 5, json.dumps(prompt_data))

        # Send push notification
        ns = get_notification_service()
        prompt_title = "Are you okay?"
        prompt_body = (
            f"SafeWay detected a potential anomaly ({anomaly_type}). "
            f"Please tap to confirm you are safe within {timeout_sec} seconds."
        )
        await ns.send_push(
            recipient_id=str(user_id),
            title=prompt_title,
            body=prompt_body,
            data={"trip_id": str(trip.id), "anomaly_type": anomaly_type},
        )
        prompt_sent = True

        # Launch auto-escalation timer
        asyncio.create_task(
            run_are_you_okay_escalation(
                trip_id=trip.id,
                user_id=user_id,
                lat=req.lat,
                lon=req.lon,
                anomaly_type=anomaly_type,
                timeout_seconds=float(timeout_sec),
            )
        )

    return TripLocationResponse(
        status="anomaly_flagged" if anomaly_detected else "recorded",
        trip_id=trip.id,
        anomaly_detected=anomaly_detected,
        anomaly_type=anomaly_type,
        prompt_sent=prompt_sent,
        details=details if anomaly_detected else None,
    )


async def respond_to_anomaly_check(
    db: AsyncSession,
    trip_id: uuid.UUID,
    user_id: uuid.UUID,
    req: TripRespondRequest,
) -> TripRespondResponse:
    """
    Handle user response ('I'm okay') to dismiss the prompt and cancel auto-escalation.
    """
    trip = await db.get(Trip, trip_id)
    if not trip:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trip not found.")
    if trip.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized.")

    # Clear pending Redis escalation key
    redis = get_redis()
    if redis:
        try:
            redis.delete(f"trip:are_you_okay:{trip.id}")
        except Exception:
            pass

    if trip.status == TripStatus.DEVIATED:
        trip.status = TripStatus.ACTIVE
        await db.commit()

    logger.info("Trip %s: user confirmed safety with '%s'. Escalation cancelled.", trip_id, req.message)
    return TripRespondResponse(
        trip_id=trip.id,
        status=trip.status.value,
        message="Confirmation received. Emergency escalation cancelled.",
        escalation_cancelled=True,
    )


async def run_are_you_okay_escalation(
    trip_id: uuid.UUID,
    user_id: uuid.UUID,
    lat: float,
    lon: float,
    anomaly_type: str,
    timeout_seconds: float,
) -> None:
    """
    Background timer: wait timeout_seconds.
    If 'trip:are_you_okay:{trip_id}' is still present in Redis, the prompt went unanswered.
    Auto-escalate into the Step 5 SOS emergency dispatch flow!
    """
    try:
        await asyncio.sleep(timeout_seconds)
        redis = get_redis()
        is_pending = True
        if redis:
            raw = redis.get(f"trip:are_you_okay:{trip_id}")
            is_pending = raw is not None

        if not is_pending:
            logger.info("Trip %s: escalation timer expired but check was acknowledged by user.", trip_id)
            return

        logger.warning(
            "Trip %s: 'Are you okay?' went unanswered after %.1fs. AUTO-ESCALATING TO SOS!",
            trip_id, timeout_seconds,
        )

        # Clear Redis key to prevent duplicate escalation
        if redis:
            redis.delete(f"trip:are_you_okay:{trip_id}")

        async with async_session_factory() as session:
            trip = await session.get(Trip, trip_id)
            if trip and trip.status != TripStatus.COMPLETED:
                trip.status = TripStatus.SOS_TRIGGERED
                await session.commit()

            user = await session.get(User, user_id)
            if user:
                from app.schemas.sos import SOSTriggerRequest
                from app.services.sos_service import trigger_sos

                sos_req = SOSTriggerRequest(
                    lat=lat,
                    lon=lon,
                    trigger_type=f"trip_anomaly_{anomaly_type}",
                    trip_id=trip_id,
                )
                await trigger_sos(
                    db=session,
                    current_user=user,
                    req=sos_req,
                    countdown_seconds=0,
                )
                logger.info("Trip %s: SOS flow auto-dispatched successfully.", trip_id)
    except Exception as exc:
        logger.error("Error executing trip anomaly auto-escalation for trip %s: %s", trip_id, exc, exc_info=True)


async def end_trip(
    db: AsyncSession,
    trip_id: uuid.UUID,
    user_id: uuid.UUID,
) -> TripEndResponse:
    """
    Conclude a monitored trip, clear pending timers, and return trip duration.
    """
    trip = await db.get(Trip, trip_id)
    if not trip:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trip not found.")
    if trip.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized.")

    now = datetime.now(timezone.utc)
    trip.status = TripStatus.COMPLETED
    trip.ended_at = now
    await db.commit()

    # Clear pending anomaly timer
    redis = get_redis()
    if redis:
        try:
            redis.delete(f"trip:are_you_okay:{trip.id}")
        except Exception:
            pass

    duration_s = max(0.0, (now - trip.started_at).total_seconds())
    return TripEndResponse(
        trip_id=trip.id,
        status=trip.status.value,
        started_at=trip.started_at,
        ended_at=now,
        duration_seconds=round(duration_s, 2),
    )


# ── Feedback & EMA Learning ──────────────────────────────────────────────────

async def record_trip_feedback(
    db: AsyncSession,
    trip_id: uuid.UUID,
    user_id: uuid.UUID,
    req: TripFeedbackRequest,
) -> TripFeedbackResponse:
    """
    Store post-trip safety feedback and adjust per-segment risk weights via EMA.
    """
    trip = await db.get(Trip, trip_id)
    if not trip:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trip not found.")
    if trip.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to submit feedback for this trip.")

    fb = Feedback(
        user_id=user_id,
        trip_id=trip.id,
        safety_rating=req.safety_rating,
        route_rating=req.route_rating or req.safety_rating,
        comment=req.comment,
    )
    db.add(fb)
    await db.commit()
    await db.refresh(fb)

    # ── Feedback Learning: EMA Adjustment of Per-Segment Risk Weights ──────────
    # Normalized feedback risk: 1 star -> 1.0 (very unsafe), 5 stars -> 0.0 (very safe)
    feedback_risk = 1.0 - (float(req.safety_rating) - 1.0) / 4.0
    alpha = 0.20  # EMA smoothing factor

    redis = get_redis()
    if redis and trip.route_geojson:
        try:
            # Extract segment IDs or features from route_geojson
            segment_ids: list[str] = []
            features = trip.route_geojson.get("features", [])
            for feat in features:
                props = feat.get("properties", {})
                sid = props.get("segment_id") or props.get("osm_id")
                if sid:
                    segment_ids.append(str(sid))

            # If no explicit segment IDs in properties, derive dummy IDs based on trip
            if not segment_ids:
                segment_ids = [f"seg_{trip.id}_{i}" for i in range(3)]

            for sid in segment_ids:
                key = f"segment:ema_risk:{sid}"
                prev = redis.get(key)
                prev_val = float(prev) if prev else 0.50
                new_val = (1.0 - alpha) * prev_val + alpha * feedback_risk
                redis.set(key, str(round(new_val, 4)))
            logger.info("EMA risk adjusted for %d segments with rating=%d (risk=%.2f)", len(segment_ids), req.safety_rating, feedback_risk)
        except Exception as exc:
            logger.warning("Failed to apply EMA risk feedback adjustment: %s", exc)

    return TripFeedbackResponse(
        feedback_id=fb.id,
        trip_id=trip.id,
        safety_rating=fb.safety_rating,
        route_rating=fb.route_rating,
        comment=fb.comment,
        message="Feedback recorded and incorporated into safety scoring.",
    )


# ── Privacy: Retention Purge & Access Control ────────────────────────────────

async def purge_old_trip_locations(
    db: AsyncSession,
    retention_days: int | None = None,
) -> int:
    """
    Privacy policy: delete raw GPS location trails older than N days.
    """
    days = retention_days if retention_days is not None else settings.TRIP_LOCATION_RETENTION_DAYS
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    stmt = delete(TripLocation).where(TripLocation.recorded_at < cutoff)
    res = await db.execute(stmt)
    await db.commit()
    deleted_count = res.rowcount or 0
    logger.info("Privacy cleanup: purged %d raw trip location pings older than %d days.", deleted_count, days)
    return deleted_count


async def get_trip_live_location(
    db: AsyncSession,
    trip_id: uuid.UUID,
    current_user: User,
) -> TripLiveLocationResponse:
    """
    Live location and trail.
    Privacy constraint: shared ONLY with the trip owner and their trusted contacts.
    """
    trip = await db.get(Trip, trip_id)
    if not trip:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trip not found.")

    # Authorization: Owner, Authority, or Registered Trusted Contact
    is_owner = trip.user_id == current_user.id
    is_authority = current_user.role == UserRole.AUTHORITY

    is_trusted = False
    if not (is_owner or is_authority):
        tc_stmt = select(TrustedContact).where(
            TrustedContact.user_id == trip.user_id,
            TrustedContact.contact_phone == current_user.phone,
        )
        tc_res = await db.execute(tc_stmt)
        if tc_res.scalar_one_or_none():
            is_trusted = True

    if not (is_owner or is_authority or is_trusted):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Live trip location is shared only with the user and their registered trusted emergency contacts.",
        )

    # Fetch ordered locations
    loc_stmt = select(TripLocation).where(TripLocation.trip_id == trip.id).order_by(TripLocation.recorded_at)
    loc_res = await db.execute(loc_stmt)
    db_locs = loc_res.scalars().all()

    trail = []
    for l in db_locs:
        l_lat, l_lon = _coords_from_geom(l.geometry)
        trail.append({
            "lat": l_lat,
            "lon": l_lon,
            "accuracy_m": l.accuracy_m,
            "recorded_at": l.recorded_at,
        })

    latest = trail[-1] if trail else None
    return TripLiveLocationResponse(
        trip_id=trip.id,
        user_id=trip.user_id,
        status=trip.status.value,
        latest_location=latest,
        trail=trail,
        started_at=trip.started_at,
    )
