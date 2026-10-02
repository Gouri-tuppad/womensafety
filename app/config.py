"""
SafeWay – application configuration via ``pydantic-settings``.

Every value is overridable through an env var or ``.env`` file.
Zone presets define bounding boxes for the two demo locations in Bengaluru.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


# ── Zone presets ─────────────────────────────────────────────────────────────
# Bounding boxes approximate; verified against OpenStreetMap extents.
# Centre coordinates feed ``astral`` for sunset/sunrise calculation.

@dataclass(frozen=True)
class ZonePreset:
    """Geographic bounding box and metadata for a demo zone."""

    display_name: str
    bbox_north: float
    bbox_south: float
    bbox_east: float
    bbox_west: float
    center_lat: float
    center_lon: float
    description: str


ZONE_PRESETS: dict[str, ZonePreset] = {
    "peenya": ZonePreset(
        display_name="Peenya Industrial Area",
        bbox_north=13.05,
        bbox_south=13.00,
        bbox_east=77.55,
        bbox_west=77.48,
        center_lat=13.025,
        center_lon=77.515,
        description=(
            "Large industrial belt: factories, warehouses, garment units. "
            "Busy at shift changes (~06-07, 14-15, 22-23), very empty and "
            "dark at night with long compound walls and few shops."
        ),
    ),
    "subramanyapura": ZonePreset(
        display_name="Subramanyapura",
        bbox_north=12.91,
        bbox_south=12.86,
        bbox_east=77.58,
        bbox_west=77.52,
        center_lat=12.885,
        center_lon=77.55,
        description=(
            "Mostly residential with a metro station, Kanakapura Road as the "
            "main artery, interior lanes, vacant plots and lake/park edges. "
            "Thin crowds after ~22:00."
        ),
    ),
}


# ── Application settings ────────────────────────────────────────────────────

class Settings(BaseSettings):
    """Central configuration – every field can be overridden via env var."""

    # App
    APP_NAME: str = "SafeWay"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"

    # Zone
    ZONE: Literal["peenya", "subramanyapura"] = "peenya"
    TIMEZONE: str = "Asia/Kolkata"

    # PostgreSQL + PostGIS (async URL for the application)
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://safeway:safeway@localhost:5432/safeway",
        description="Async database URL (SQLAlchemy async engine).",
    )
    # Sync URL used by Alembic & seeding scripts
    DATABASE_URL_SYNC: str = Field(
        default="postgresql+psycopg://safeway:safeway@localhost:5432/safeway",
        description="Sync database URL (Alembic, seed scripts).",
    )

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── Auth / JWT ───────────────────────────────────────────────────────
    JWT_SECRET: str = "CHANGE_ME_IN_PRODUCTION_use_openssl_rand_hex_32"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60

    # ── Safety-cost formula weights ──────────────────────────────────────
    # cost = distance × clamp(1 + α·Crime + β·Darkness + γ·Isolation − δ·Activity, min=0.5)
    ALPHA_CRIME: float = 0.4
    BETA_DARKNESS: float = 0.3
    GAMMA_ISOLATION: float = 0.2
    DELTA_ACTIVITY: float = 0.1

    # ── SOS ──────────────────────────────────────────────────────────────
    SOS_COUNTDOWN_SECONDS: int = 5

    # ── Volunteers ───────────────────────────────────────────────────────
    VOLUNTEER_RADIUS_M: int = 1000

    # ── Trip monitoring ──────────────────────────────────────────────────
    TRIP_STALE_MINUTES: int = 5
    TRIP_DEVIATION_METERS: float = 200.0

    # ── Provider switches ────────────────────────────────────────────────
    # "mock" implementations need no API keys.
    SMS_PROVIDER: Literal["mock", "twilio"] = "mock"
    PUSH_PROVIDER: Literal["mock", "fcm"] = "mock"
    LLM_PROVIDER: Literal["mock", "openai", "anthropic"] = "mock"

    # ── Rate limiting ────────────────────────────────────────────────────
    RATE_LIMIT_PER_MINUTE: int = 60

    # ── Graph cache ──────────────────────────────────────────────────────
    GRAPH_CACHE_DIR: str = ".graph_cache"

    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
    }

    # ── Derived helpers ──────────────────────────────────────────────────

    @property
    def zone_preset(self) -> ZonePreset:
        """Return the currently-active zone preset."""
        return ZONE_PRESETS[self.ZONE]


settings = Settings()
