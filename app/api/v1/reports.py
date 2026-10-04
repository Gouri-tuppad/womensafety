"""
app/api/v1/reports.py – Community hazard & incident reporting API.

Endpoints:
  POST /api/v1/reports          – Submit a community safety hazard report
  GET  /api/v1/reports          – List recent reports
  POST /api/v1/reports/{id}/moderate – Moderate report (authority / reviewer)
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import require_roles
from app.deps import get_current_user, get_db
from app.models.user import User, UserRole
from app.schemas.report import (
    ReportCreateRequest,
    ReportModerateRequest,
    ReportModerateResponse,
    ReportResponse,
)
from app.services import report_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "",
    response_model=ReportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a community safety hazard report",
    description=(
        "Records crowdsourced safety hazard (dark street, harassment, felt unsafe, broken light, isolated area). "
        "Applies reporter trust scoring, de-duplicates nearby occurrences, and updates street risk in real time."
    ),
)
async def create_report(
    req: ReportCreateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ReportResponse:
    return await report_service.submit_report(
        db=db,
        current_user=current_user,
        req=req,
    )


@router.get(
    "",
    response_model=list[ReportResponse],
    summary="List recent community reports",
    description="Fetches recent community safety reports for visualization and analysis.",
)
async def get_reports(
    limit: int = Query(default=50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[ReportResponse]:
    return await report_service.list_reports(db=db, limit=limit)


@router.post(
    "/{id}/moderate",
    response_model=ReportModerateResponse,
    summary="Moderate a community report (authority)",
    description="Authority verdict ('verified' or 'rejected'), adjusting the reporter's trust score accordingly.",
    dependencies=[Depends(require_roles("authority"))],
)
async def moderate_report_endpoint(
    id: uuid.UUID,
    req: ReportModerateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ReportModerateResponse:
    return await report_service.moderate_report(
        db=db,
        report_id=id,
        current_user=current_user,
        req=req,
    )
