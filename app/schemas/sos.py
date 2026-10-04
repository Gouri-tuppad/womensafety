"""
app/schemas/sos.py - Pydantic models for SOS trigger, tracking, evidence, and updates.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class SOSTriggerRequest(BaseModel):
    """Payload for triggering an SOS event."""
    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude of emergency origin.")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Longitude of emergency origin.")
    timestamp: Optional[datetime] = Field(None, description="Client trigger timestamp.")
    trigger_type: str = Field(default="shake", description="'shake', 'button', 'voice', or 'auto'.")
    accel_window: Optional[List[Any]] = Field(
        None,
        description="Optional accelerometer window samples [[x, y, z, t], ...] or dicts.",
    )
    trip_id: Optional[uuid.UUID] = Field(None, description="Optional active trip ID.")
    notes: Optional[str] = Field(None, description="Optional trigger metadata.")


class SOSTriggerResponse(BaseModel):
    """Response returned upon SOS creation or duplicate trigger identification."""
    sos_id: uuid.UUID
    status: str
    countdown_seconds: int
    trigger_type: str
    confidence: float
    triggered_at: datetime
    tracking_token: str
    tracking_url: str
    evidence_capture_requested: bool = False
    is_duplicate: bool = False

    model_config = ConfigDict(from_attributes=True)


class SOSCancelRequest(BaseModel):
    """Payload to cancel an SOS during the 5-second countdown window."""
    reason: Optional[str] = Field(default="false_alarm", description="Reason for cancellation.")


class SOSCancelResponse(BaseModel):
    """Response returned upon cancelling an SOS."""
    sos_id: uuid.UUID
    status: str
    message: str


class SOSLocationUpdate(BaseModel):
    """Payload sent by mobile client every few seconds during active SOS."""
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    accuracy_m: Optional[float] = Field(None, ge=0.0)
    battery_level: Optional[float] = Field(None, ge=0.0, le=100.0, description="Battery percentage.")
    timestamp: Optional[datetime] = None


class SOSResolveRequest(BaseModel):
    """Payload to resolve an SOS and notify contacts user is safe."""
    notes: Optional[str] = Field(None, description="Resolution notes.")


class SOSResolveResponse(BaseModel):
    """Response returned upon resolving an active SOS."""
    sos_id: uuid.UUID
    status: str
    resolved_at: datetime
    message: str


class LocationPoint(BaseModel):
    """Single geo point in the SOS route trail."""
    lat: float
    lon: float
    accuracy_m: Optional[float] = None
    recorded_at: datetime


class SOSTrackResponse(BaseModel):
    """Public tracking state returned via signed token."""
    sos_id: uuid.UUID
    status: str
    user_name: str
    user_phone: str
    trigger_type: str
    confidence: float
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    latest_location: Optional[LocationPoint] = None
    battery_level: Optional[float] = None
    route_trail: List[LocationPoint] = []
    evidence_capture_requested: bool = False


class EvidenceUploadResponse(BaseModel):
    """Response returned after uploading evidence file."""
    id: uuid.UUID
    sos_event_id: uuid.UUID
    file_type: str
    file_path: str
    duration_s: Optional[float] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class SOSStatusResponse(BaseModel):
    """Response returned for GET /api/v1/sos/{id} (owner status check)."""
    sos_id: uuid.UUID
    status: str
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    trigger_type: str
    confidence: float
    tracking_url: str
    tracking_token: Optional[str] = None
    evidence_capture_requested: bool = False
    notes: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)

