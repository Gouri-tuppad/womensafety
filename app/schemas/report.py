"""
app/schemas/report.py – Pydantic schemas for community safety reports and moderation.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ReportType(str, Enum):
    DARK_STREET = "dark_street"
    HARASSMENT = "harassment"
    FELT_UNSAFE = "felt_unsafe"
    BROKEN_LIGHT = "broken_light"
    ISOLATED_AREA = "isolated_area"


class ReportCreateRequest(BaseModel):
    """Payload to submit a crowdsourced safety hazard or incident report."""

    report_type: ReportType = Field(..., description="Category of hazard / incident")
    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude of the hazard")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Longitude of the hazard")
    timestamp: datetime | None = Field(default=None, description="Observation timestamp")
    title: str | None = Field(default=None, max_length=200, description="Optional brief title")
    description: str | None = Field(default=None, max_length=2000, description="Optional description of the situation")


class ReportResponse(BaseModel):
    """Outcome of submitting or querying a safety report."""

    report_id: uuid.UUID
    user_id: uuid.UUID
    report_type: str
    title: str
    description: str
    status: str
    lat: float
    lon: float
    created_at: datetime
    corroborated: bool = False
    corroboration_count: int = 0
    reporter_trust_score: float
    affected_segments: list[int] = []
    message: str


class ReportModerateRequest(BaseModel):
    """Payload for authority or peer moderation of a report."""

    status: str = Field(..., pattern="^(verified|rejected)$", description="'verified' or 'rejected'")
    action_reason: str | None = Field(default=None, max_length=500, description="Reason for verdict")


class ReportModerateResponse(BaseModel):
    """Result of moderating a report."""

    report_id: uuid.UUID
    status: str
    reporter_new_trust: float
    message: str
