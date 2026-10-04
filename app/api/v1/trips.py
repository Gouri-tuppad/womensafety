"""
app/api/v1/trips.py – Live trip monitoring and safety anomaly management endpoints.

Endpoints:
  POST /api/v1/trips/start            – Start a monitored trip and store planned route
  POST /api/v1/trips/{id}/location     – Record GPS ping + motion window, evaluate anomalies
  POST /api/v1/trips/{id}/end          – Conclude a monitored trip
  POST /api/v1/trips/{id}/feedback     – Post-trip feedback ('Did you feel safe?' 1-5)
  POST /api/v1/trips/{id}/respond      – User response ('I'm okay') to dismiss anomaly prompt
  GET  /api/v1/trips/{id}/live         – Live location trail (trusted contacts only)
  POST /api/v1/trips/purge-locations   – Scheduled/maintenance purge of old GPS trails
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.deps import get_current_user, get_db
from app.models.user import User
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
from app.services import trip_monitor

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/start",
    response_model=TripStartResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start a live monitored trip",
    description="Initializes a live trip, stores planned route geometry, and begins monitoring.",
)
async def start_trip(
    req: TripStartRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> TripStartResponse:
    return await trip_monitor.start_trip(db=db, user_id=current_user.id, req=req)


@router.post(
    "/{id}/location",
    response_model=TripLocationResponse,
    summary="Record live GPS ping and evaluate motion/route anomalies",
    description=(
        "Appends a GPS location to the trip trail and checks for: "
        "1. Route deviation beyond threshold, "
        "2. Unusual prolonged stop in high-risk area, "
        "3. Motion anomaly (panic running or phone drop) via Isolation Forest. "
        "On anomaly, sends an 'Are you okay?' prompt and starts an auto-escalation timer."
    ),
)
async def record_location(
    id: uuid.UUID,
    req: TripLocationRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> TripLocationResponse:
    return await trip_monitor.record_trip_location(
        db=db,
        trip_id=id,
        user_id=current_user.id,
        req=req,
    )


@router.post(
    "/{id}/end",
    response_model=TripEndResponse,
    summary="Conclude a monitored trip",
    description="Marks the trip completed and cancels any active anomaly timers.",
)
async def end_trip(
    id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> TripEndResponse:
    return await trip_monitor.end_trip(db=db, trip_id=id, user_id=current_user.id)


@router.post(
    "/{id}/feedback",
    response_model=TripFeedbackResponse,
    summary="Submit post-trip safety rating and feedback",
    description="Records rating (1-5) and applies EMA adjustments to per-segment safety risk weights.",
)
async def submit_feedback(
    id: uuid.UUID,
    req: TripFeedbackRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> TripFeedbackResponse:
    return await trip_monitor.record_trip_feedback(
        db=db,
        trip_id=id,
        user_id=current_user.id,
        req=req,
    )


@router.post(
    "/{id}/respond",
    response_model=TripRespondResponse,
    summary="Respond 'I'm okay' to anomaly check",
    description="Acknowledges the safety prompt, clears the Redis countdown, and prevents SOS escalation.",
)
async def respond_to_check(
    id: uuid.UUID,
    req: TripRespondRequest = TripRespondRequest(),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> TripRespondResponse:
    return await trip_monitor.respond_to_anomaly_check(
        db=db,
        trip_id=id,
        user_id=current_user.id,
        req=req,
    )


@router.get(
    "/{id}/live",
    response_model=TripLiveLocationResponse,
    summary="Get live trip location (owner or trusted contacts only)",
    description="Privacy-restricted view of live trip location and trail.",
)
async def get_live_location(
    id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> TripLiveLocationResponse:
    return await trip_monitor.get_trip_live_location(
        db=db,
        trip_id=id,
        current_user=current_user,
    )


@router.post(
    "/purge-locations",
    summary="Purge location trails older than retention policy",
    description="Privacy compliance maintenance job.",
)
async def purge_locations(
    days: int = Query(default=7, ge=1, le=365),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    deleted_count = await trip_monitor.purge_old_trip_locations(db=db, retention_days=days)
    return {"status": "success", "deleted_location_records": deleted_count, "retention_days": days}
