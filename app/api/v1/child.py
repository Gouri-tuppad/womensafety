"""
app/api/v1/child.py – Child safety, geofencing, and parent monitoring API.

Endpoints:
  POST /api/v1/child/link             – Link child account to parent
  POST /api/v1/child/{id}/geofences    – Define safe zones / corridors for child
  GET  /api/v1/child/{id}/geofences    – List child's active geofences
  POST /api/v1/child/{id}/location     – Child location ping (checks geofences and alerts parent)
  GET  /api/v1/child/{id}/suggested-route – Safest school route with 'child' profile
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.deps import get_current_user, get_db
from app.models.user import User
from app.schemas.child import (
    ChildLinkRequest,
    ChildLinkResponse,
    ChildLocationPingRequest,
    ChildLocationPingResponse,
    ChildSuggestedRouteRequest,
    ChildSuggestedRouteResponse,
    GeofenceCreateRequest,
    GeofenceResponse,
)
from app.services import geofence as geofence_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/link",
    response_model=ChildLinkResponse,
    summary="Link child account to parent",
    description="Parent guardian links a child account via email or phone.",
)
async def link_child(
    req: ChildLinkRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ChildLinkResponse:
    return await geofence_service.link_child_to_parent(
        db=db,
        parent_user=current_user,
        req=req,
    )


@router.post(
    "/{id}/geofences",
    response_model=GeofenceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a safe zone / corridor geofence for child",
    description="Defines a circular safe zone or transit corridor (Home, School, Tuition).",
)
async def create_geofence(
    id: uuid.UUID,
    req: GeofenceCreateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> GeofenceResponse:
    return await geofence_service.create_child_geofence(
        db=db,
        parent_user=current_user,
        child_id=id,
        req=req,
    )


@router.get(
    "/{id}/geofences",
    response_model=list[GeofenceResponse],
    summary="List child's active geofences",
    description="Fetches all active safe zones and corridors configured for a child.",
)
async def list_geofences(
    id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[GeofenceResponse]:
    return await geofence_service.list_child_geofences(db=db, child_id=id)


@router.post(
    "/{id}/location",
    response_model=ChildLocationPingResponse,
    summary="Record child location ping and evaluate geofences",
    description="Evaluates child coordinates against safe zones; sends SMS/Push to parent on breach.",
)
async def check_child_location(
    id: uuid.UUID,
    req: ChildLocationPingRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ChildLocationPingResponse:
    return await geofence_service.evaluate_child_location(
        db=db,
        child_id=id,
        req=req,
    )


@router.get(
    "/{id}/suggested-route",
    response_model=ChildSuggestedRouteResponse,
    summary="Compute suggested safest route with child safety profile",
    description="Computes safest route tailored for children (prioritizing lighting and safe havens) for parent approval.",
)
async def get_suggested_route(
    id: uuid.UUID,
    origin_lat: float | None = None,
    origin_lon: float | None = None,
    dest_lat: float | None = None,
    dest_lon: float | None = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ChildSuggestedRouteResponse:
    req = ChildSuggestedRouteRequest(
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        dest_lat=dest_lat,
        dest_lon=dest_lon,
    )
    return await geofence_service.get_child_suggested_route(
        db=db,
        child_id=id,
        req=req,
    )
