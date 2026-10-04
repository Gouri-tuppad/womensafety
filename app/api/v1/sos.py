"""
app/api/v1/sos.py - Shake-Gesture SOS and Live Tracking API.

Endpoints:
  POST /sos/trigger       - Trigger emergency SOS (shake gesture or manual) with 5s countdown
  POST /sos/{id}/cancel   - Cancel SOS within countdown window (anti-false-alarm)
  POST /sos/{id}/location - Mobile client periodic location updates with WebSocket broadcast
  POST /sos/{id}/evidence - Upload audio, video, or photo evidence
  POST /sos/{id}/resolve  - Mark safe and notify emergency contacts
  GET  /track/{token}     - Public live tracking status via signed expiring token
  WS   /ws/track/{token}  - Public WebSocket for real-time tracking updates
"""

from __future__ import annotations

import logging
import uuid
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.deps import get_current_user, get_db
from app.models.user import User
from app.schemas.sos import (
    EvidenceUploadResponse,
    SOSCancelRequest,
    SOSCancelResponse,
    SOSLocationUpdate,
    SOSResolveRequest,
    SOSResolveResponse,
    SOSTrackResponse,
    SOSTriggerRequest,
    SOSTriggerResponse,
)
from app.services import sos_service

logger = logging.getLogger(__name__)

router = APIRouter()


# ── POST /sos/trigger ─────────────────────────────────────────────────────────

@router.post(
    "/sos/trigger",
    response_model=SOSTriggerResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Trigger Emergency SOS",
    description=(
        "Triggers an SOS event with status COUNTDOWN and starts a 5-second silent "
        "countdown. Anti-false-alarm shake verification runs on the optional accelerometer "
        "window without ever blocking emergency creation. Duplicate triggers within 30s "
        "map to the same event."
    ),
)
async def trigger_sos_endpoint(
    req: SOSTriggerRequest,
    request: Request,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> SOSTriggerResponse:
    ip = request.client.host if request.client else None
    return await sos_service.trigger_sos(
        db=db,
        current_user=current_user,
        req=req,
        ip_address=ip,
    )


# ── POST /sos/{id}/cancel ───────────────────────────────────────────────────

@router.post(
    "/sos/{id}/cancel",
    response_model=SOSCancelResponse,
    summary="Cancel SOS during countdown",
    description=(
        "Cancels an SOS during the 5-second countdown window to avoid false alarms. "
        "Aborts emergency dispatch."
    ),
)
async def cancel_sos_endpoint(
    id: uuid.UUID,
    req: SOSCancelRequest,
    request: Request,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> SOSCancelResponse:
    ip = request.client.host if request.client else None
    return await sos_service.cancel_sos(
        db=db,
        sos_id=id,
        user_id=current_user.id,
        reason=req.reason or "false_alarm",
        ip_address=ip,
    )


# ── POST /sos/{id}/location ─────────────────────────────────────────────────

@router.post(
    "/sos/{id}/location",
    summary="Push periodic SOS location update",
    description="Updates the active SOS location trail and broadcasts live via WebSocket.",
)
async def update_location_endpoint(
    id: uuid.UUID,
    payload: SOSLocationUpdate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    return await sos_service.update_sos_location(
        db=db,
        sos_id=id,
        user_id=current_user.id,
        payload=payload,
    )


# ── POST /sos/{id}/evidence ─────────────────────────────────────────────────

@router.post(
    "/sos/{id}/evidence",
    response_model=EvidenceUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload SOS evidence file",
    description="Upload audio, video, or photo evidence recorded by the client device.",
)
async def upload_evidence_endpoint(
    id: uuid.UUID,
    file: UploadFile = File(...),
    file_type: str = Form(..., description="audio, video, or image"),
    duration_s: float | None = Form(None),
    request: Request = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> EvidenceUploadResponse:
    ip = request.client.host if request and request.client else None
    return await sos_service.upload_evidence(
        db=db,
        sos_id=id,
        user_id=current_user.id,
        file=file,
        file_type=file_type,
        duration_s=duration_s,
        ip_address=ip,
    )


# ── POST /sos/{id}/resolve ──────────────────────────────────────────────────

@router.post(
    "/sos/{id}/resolve",
    response_model=SOSResolveResponse,
    summary="Mark SOS safe and resolved",
    description="Marks emergency safe, records resolution, and notifies trusted contacts.",
)
async def resolve_sos_endpoint(
    id: uuid.UUID,
    req: SOSResolveRequest,
    request: Request,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> SOSResolveResponse:
    ip = request.client.host if request.client else None
    return await sos_service.resolve_sos(
        db=db,
        sos_id=id,
        user_id=current_user.id,
        notes=req.notes,
        ip_address=ip,
    )


# ── GET /track/{token} ──────────────────────────────────────────────────────

@router.get(
    "/track/{token}",
    response_model=SOSTrackResponse,
    summary="Public Live SOS Tracking",
    description=(
        "Publicly viewable live location and status of an active SOS via a signed, "
        "expiring token provided to trusted contacts or emergency responders."
    ),
)
async def get_tracking_endpoint(
    token: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> SOSTrackResponse:
    sos_id = sos_service.decode_tracking_token(token)
    return await sos_service.get_tracking_data(db=db, sos_id=sos_id)


# ── WebSocket /ws/track/{token} ──────────────────────────────────────────────

@router.websocket("/ws/track/{token}")
async def tracking_websocket_endpoint(
    websocket: WebSocket,
    token: str,
    db: AsyncSession = Depends(get_db),
):
    """
    WebSocket endpoint for real-time tracking updates (location, battery, status).
    """
    try:
        sos_id = sos_service.decode_tracking_token(token)
    except Exception as exc:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await sos_service.ws_manager.connect(sos_id, websocket)

    # Immediately push initial state
    try:
        track_data = await sos_service.get_tracking_data(db=db, sos_id=sos_id)
        await websocket.send_json({
            "type": "initial_state",
            "data": track_data.model_dump(mode="json"),
        })
    except Exception as err:
        logger.warning("Error sending initial WebSocket state: %s", err)

    try:
        while True:
            # Client heartbeat / ping handling
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        await sos_service.ws_manager.disconnect(sos_id, websocket)
    except Exception:
        await sos_service.ws_manager.disconnect(sos_id, websocket)
