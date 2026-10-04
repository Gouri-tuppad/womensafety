"""
app/schemas/child.py – Pydantic schemas for child/parent account linking, geofencing, and monitoring.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ChildLinkRequest(BaseModel):
    """Payload for parent linking child account by email or phone."""

    child_email: str | None = Field(default=None, description="Child's registered email")
    child_phone: str | None = Field(default=None, description="Child's registered phone number")


class ChildLinkResponse(BaseModel):
    """Outcome of linking child account to parent."""

    parent_id: uuid.UUID
    child_id: uuid.UUID
    child_name: str
    status: str
    message: str


class GeofenceCreateRequest(BaseModel):
    """Payload to create a safe zone or transit corridor geofence for a child."""

    name: str = Field(..., max_length=120, description="e.g. Home, School, Tuition, Daily Route")
    geofence_type: str = Field(default="safe_zone", pattern="^(safe_zone|corridor)$", description="'safe_zone' or 'corridor'")
    center_lat: float | None = Field(default=None, ge=-90.0, le=90.0, description="Latitude for circular safe zone")
    center_lon: float | None = Field(default=None, ge=-180.0, le=180.0, description="Longitude for circular safe zone")
    radius_m: float = Field(default=150.0, ge=10.0, le=5000.0, description="Radius in metres for circular safe zone")
    polygon_coords: list[list[float]] | None = Field(
        default=None,
        description="Optional explicit polygon vertices [[lon, lat], ...]",
    )
    alert_on_exit: bool = Field(default=True, description="Alert parent when child exits this zone")


class GeofenceResponse(BaseModel):
    """Details of a registered geofence."""

    geofence_id: uuid.UUID
    user_id: uuid.UUID
    name: str
    alert_on_exit: bool
    is_active: bool
    created_at: datetime


class ChildLocationPingRequest(BaseModel):
    """Live GPS ping from child's device."""

    lat: float = Field(..., ge=-90.0, le=90.0, description="Current latitude")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Current longitude")
    accuracy_m: float | None = Field(default=None, ge=0.0, description="GPS accuracy in metres")
    timestamp: datetime | None = Field(default=None, description="Observation timestamp")


class ChildLocationPingResponse(BaseModel):
    """Outcome of evaluating child position against active geofences and motion stops."""

    child_id: uuid.UUID
    inside_safe_zone: bool
    breach_detected: bool
    breached_geofence_name: str | None = None
    unexpected_stop_detected: bool = False
    alert_sent: bool
    message: str


class ChildSuggestedRouteRequest(BaseModel):
    """Query parameters to compute the safest child school/tuition route."""

    origin_lat: float | None = Field(default=None, description="Optional start latitude (defaults to Home geofence)")
    origin_lon: float | None = Field(default=None, description="Optional start longitude")
    dest_lat: float | None = Field(default=None, description="Optional destination latitude (defaults to School geofence)")
    dest_lon: float | None = Field(default=None, description="Optional destination longitude")


class ChildSuggestedRouteResponse(BaseModel):
    """Safest route computed with child profile and ready for parent approval."""

    child_id: uuid.UUID
    profile: str = "child"
    origin: dict[str, float]
    destination: dict[str, float]
    safety_score: float
    route_geojson: dict[str, Any]
    safety_explanations: list[str] = []
    message: str
