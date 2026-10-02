"""
app/schemas/route.py - Request and response schemas for routing and safety heatmap.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class RouteProfile(str, Enum):
    ADULT_DAY = "adult_day"
    WOMAN_NIGHT = "woman_night"
    CHILD = "child"


class SegmentRisk(str, Enum):
    GREEN = "green"    # safety_score >= 0.60
    YELLOW = "yellow"  # 0.30 <= safety_score < 0.60
    RED = "red"        # safety_score < 0.30


# ── Request ───────────────────────────────────────────────────────────────────

class LatLon(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)


class RouteRequest(BaseModel):
    origin: LatLon
    destination: LatLon
    depart_time: datetime = Field(
        ...,
        description="Departure time (ISO-8601 with timezone). Used to compute darkness and crime time-slot.",
        examples=["2024-10-15T23:00:00+05:30"],
    )
    profile: RouteProfile = Field(
        RouteProfile.ADULT_DAY,
        description="Weight profile controlling alpha/beta/gamma/delta.",
    )

    @field_validator("depart_time")
    @classmethod
    def must_have_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("depart_time must include timezone information")
        return v


# ── Response ──────────────────────────────────────────────────────────────────

class SegmentInfo(BaseModel):
    osm_id: int
    risk: SegmentRisk
    safety_score: float = Field(..., ge=0.0, le=1.0)
    crime_risk: float
    darkness_risk: float
    isolation_risk: float
    activity_score: float
    lit_ratio: float
    lamp_count: int


class SafeHaven(BaseModel):
    name: str
    category: str
    lat: float
    lon: float
    dist_from_route_m: float


class RouteResult(BaseModel):
    polyline: list[list[float]]  # [[lat, lon], ...]
    distance_m: float
    duration_s: float
    safety_score: float = Field(..., ge=0, le=100, description="0-100; higher is safer")
    segments: list[SegmentInfo]
    node_ids: list[int]


class RouteComparison(BaseModel):
    distance_pct_longer: float = Field(description="% longer than fastest (0 if safest IS fastest)")
    safety_score_gain: float = Field(description="safety_score of safest minus fastest")
    lit_ratio_fastest: float
    lit_ratio_safest: float
    lit_ratio_multiplier: float = Field(description="how many times more lit the safest route is")
    police_stations_near_safest: int
    summary: str = Field(description="Human-readable one-line comparison, e.g. '+12% longer, 3x more lit'")


class RouteResponse(BaseModel):
    zone: str
    profile: RouteProfile
    depart_time: datetime
    fastest: RouteResult
    safest: RouteResult
    routes_differ: bool = Field(description="True if safest and fastest follow different paths")
    safe_havens_near_safest: list[SafeHaven]
    comparison: RouteComparison


# ── Heatmap ───────────────────────────────────────────────────────────────────

class HeatmapCell(BaseModel):
    h3_index: str
    lat: float
    lon: float
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Composite risk 0-1; 1 = highest risk")
    crime_risk: float
    darkness_risk: float
    isolation_risk: float
    crowd_count: int


class HeatmapResponse(BaseModel):
    zone: str
    hour: int = Field(..., ge=0, le=23)
    is_dark: bool
    is_weekend: bool
    cells: list[HeatmapCell]
    cell_count: int
