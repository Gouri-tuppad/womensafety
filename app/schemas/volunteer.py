"""
app/schemas/volunteer.py – Pydantic schemas for safety volunteer opt-in, location updates, and verification.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class VolunteerOptInRequest(BaseModel):
    """Payload to opt into emergency volunteer dispatch."""

    radius_m: float = Field(default=1000.0, ge=100.0, le=5000.0, description="Response radius in metres")


class VolunteerOptInResponse(BaseModel):
    """Outcome of volunteer opt-in/opt-out."""

    volunteer_id: uuid.UUID
    user_id: uuid.UUID
    is_available: bool
    verified: bool
    radius_m: float
    message: str


class VolunteerLocationUpdateRequest(BaseModel):
    """Payload to update volunteer's current physical coordinates."""

    lat: float = Field(..., ge=-90.0, le=90.0, description="Current latitude")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Current longitude")


class VolunteerLocationUpdateResponse(BaseModel):
    """Outcome of volunteer location check-in."""

    volunteer_id: uuid.UUID
    status: str
    lat: float
    lon: float
    updated_at: datetime
    message: str = "Volunteer location updated successfully."


class VolunteerVerifyResponse(BaseModel):
    """Result of authority verification for volunteer."""

    volunteer_id: uuid.UUID
    verified: bool
    message: str
