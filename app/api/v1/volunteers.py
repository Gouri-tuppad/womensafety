"""
app/api/v1/volunteers.py – Safety volunteer opt-in, location check-in, and verification API.

Endpoints:
  POST /api/v1/volunteers/opt-in    – Opt in to emergency response dispatch
  POST /api/v1/volunteers/opt-out   – Opt out / mark unavailable
  POST /api/v1/volunteers/location  – Update current GPS location for radius queries
  POST /api/v1/volunteers/{id}/verify – Authority verifies a volunteer
  GET  /api/v1/volunteers/me        – Current volunteer profile
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from geoalchemy2.elements import WKTElement
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.security import require_roles
from app.deps import get_current_user, get_db
from app.models.user import User, UserRole
from app.models.volunteer import Volunteer
from app.schemas.volunteer import (
    VolunteerLocationUpdateRequest,
    VolunteerLocationUpdateResponse,
    VolunteerOptInRequest,
    VolunteerOptInResponse,
    VolunteerVerifyResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/opt-in",
    response_model=VolunteerOptInResponse,
    summary="Opt in as an emergency response volunteer",
    description="Registers or marks volunteer available to receive emergency SOS dispatches within radius.",
)
async def volunteer_opt_in(
    req: VolunteerOptInRequest = VolunteerOptInRequest(),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> VolunteerOptInResponse:
    stmt = select(Volunteer).where(Volunteer.user_id == current_user.id)
    res = await db.execute(stmt)
    vol = res.scalar_one_or_none()

    if not vol:
        vol = Volunteer(
            user_id=current_user.id,
            is_available=True,
            radius_m=req.radius_m,
            verified=False,
        )
        db.add(vol)
    else:
        vol.is_available = True
        vol.radius_m = req.radius_m

    await db.commit()
    await db.refresh(vol)

    return VolunteerOptInResponse(
        volunteer_id=vol.id,
        user_id=vol.user_id,
        is_available=vol.is_available,
        verified=vol.verified,
        radius_m=vol.radius_m,
        message="Volunteer availability enabled.",
    )


@router.post(
    "/opt-out",
    response_model=VolunteerOptInResponse,
    summary="Opt out of emergency response volunteer dispatch",
    description="Marks volunteer unavailable so no emergency alerts are dispatched.",
)
async def volunteer_opt_out(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> VolunteerOptInResponse:
    stmt = select(Volunteer).where(Volunteer.user_id == current_user.id)
    res = await db.execute(stmt)
    vol = res.scalar_one_or_none()

    if not vol:
        vol = Volunteer(
            user_id=current_user.id,
            is_available=False,
            verified=False,
        )
        db.add(vol)
    else:
        vol.is_available = False

    await db.commit()
    await db.refresh(vol)

    return VolunteerOptInResponse(
        volunteer_id=vol.id,
        user_id=vol.user_id,
        is_available=vol.is_available,
        verified=vol.verified,
        radius_m=vol.radius_m,
        message="Volunteer availability disabled.",
    )


@router.post(
    "/location",
    response_model=VolunteerLocationUpdateResponse,
    summary="Update volunteer location for SOS proximity matching",
    description="Checks in volunteer's physical location used by PostGIS radius queries during Step 5 SOS dispatches.",
)
async def update_location(
    req: VolunteerLocationUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> VolunteerLocationUpdateResponse:
    stmt = select(Volunteer).where(Volunteer.user_id == current_user.id)
    res = await db.execute(stmt)
    vol = res.scalar_one_or_none()

    now = datetime.now(timezone.utc)
    geom = WKTElement(f"POINT({req.lon} {req.lat})", srid=4326)

    if not vol:
        vol = Volunteer(
            user_id=current_user.id,
            is_available=True,
            last_location=geom,
            verified=False,
        )
        db.add(vol)
    else:
        vol.last_location = geom

    await db.commit()
    await db.refresh(vol)

    return VolunteerLocationUpdateResponse(
        volunteer_id=vol.id,
        status="location_updated",
        lat=req.lat,
        lon=req.lon,
        updated_at=now,
    )


@router.post(
    "/{id}/verify",
    response_model=VolunteerVerifyResponse,
    summary="Verify volunteer status (authority only)",
    description="Authority endpoint to verify a community volunteer before they receive SOS dispatches.",
    dependencies=[Depends(require_roles("authority"))],
)
async def verify_volunteer(
    id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> VolunteerVerifyResponse:
    vol = await db.get(Volunteer, id)
    if not vol:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Volunteer record not found.")

    vol.verified = True
    await db.commit()

    return VolunteerVerifyResponse(
        volunteer_id=vol.id,
        verified=True,
        message="Volunteer successfully verified by authority.",
    )
