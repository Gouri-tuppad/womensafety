"""
app/schemas/trip.py – Pydantic schemas for live trip monitoring, locations, feedback, and anomaly response.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class TripStartRequest(BaseModel):
    """Request payload to start a monitored trip."""

    origin_lat: float = Field(..., ge=-90.0, le=90.0, description="Origin latitude")
    origin_lon: float = Field(..., ge=-180.0, le=180.0, description="Origin longitude")
    destination_lat: float = Field(..., ge=-90.0, le=90.0, description="Destination latitude")
    destination_lon: float = Field(..., ge=-180.0, le=180.0, description="Destination longitude")
    route_geojson: dict[str, Any] | None = Field(
        default=None,
        description="Optional planned route GeoJSON LineString / FeatureCollection",
    )
    safety_score: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Initial estimated safety score (0-100)",
    )
    mode: str = Field(default="safest", description="Route mode: safest, fastest, or balanced")
    started_at: datetime | None = Field(default=None, description="Optional start timestamp for simulation/testing")


class TripStartResponse(BaseModel):
    """Response returned upon trip initialization."""

    trip_id: uuid.UUID
    status: str
    started_at: datetime
    safety_score: float | None = None
    origin: dict[str, float]
    destination: dict[str, float]
    route_geojson: dict[str, Any] | None = None


class TripLocationRequest(BaseModel):
    """Periodic GPS ping and optional motion window from user's device."""

    lat: float = Field(..., ge=-90.0, le=90.0, description="Current latitude")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Current longitude")
    accuracy_m: float | None = Field(default=None, ge=0.0, description="GPS accuracy in metres")
    timestamp: datetime | None = Field(default=None, description="Observation timestamp")
    motion_window: list[list[float]] | None = Field(
        default=None,
        description="Optional accel/gyro window: [[ax, ay, az, (gx, gy, gz), (t)], ...]",
    )
    speed_mps: float | None = Field(default=None, ge=0.0, description="Current speed in m/s")


class TripLocationResponse(BaseModel):
    """Response for a recorded trip location, indicating anomaly status."""

    status: str
    trip_id: uuid.UUID
    anomaly_detected: bool = False
    anomaly_type: str | None = None
    prompt_sent: bool = False
    details: dict[str, Any] | None = None


class TripEndResponse(BaseModel):
    """Response returned when a trip is concluded."""

    trip_id: uuid.UUID
    status: str
    started_at: datetime
    ended_at: datetime
    duration_seconds: float


class TripFeedbackRequest(BaseModel):
    """Post-trip user feedback."""

    safety_rating: int = Field(..., ge=1, le=5, description="Did you feel safe? (1=very unsafe, 5=very safe)")
    route_rating: int | None = Field(default=None, ge=1, le=5, description="1-5 rating for route quality")
    comment: str | None = Field(default=None, max_length=1000, description="Optional qualitative feedback")


class TripFeedbackResponse(BaseModel):
    """Confirmation of recorded trip feedback."""

    feedback_id: uuid.UUID
    trip_id: uuid.UUID
    safety_rating: int
    route_rating: int
    comment: str | None = None
    message: str = "Feedback recorded and incorporated into safety scoring."


class TripRespondRequest(BaseModel):
    """User response to 'Are you okay?' anomaly prompt."""

    message: str = Field(default="I'm okay", description="Confirmation message")


class TripRespondResponse(BaseModel):
    """Outcome of acknowledging the anomaly prompt."""

    trip_id: uuid.UUID
    status: str
    message: str
    escalation_cancelled: bool = True


class TripLiveLocationResponse(BaseModel):
    """Live location and trail accessible only to owner and trusted contacts."""

    trip_id: uuid.UUID
    user_id: uuid.UUID
    status: str
    latest_location: dict[str, Any] | None = None
    trail: list[dict[str, Any]] = []
    started_at: datetime
