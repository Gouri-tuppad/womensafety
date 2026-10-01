"""
Trip and TripLocation models.

Trip stores the planned route; TripLocation records live GPS pings.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import DateTime, Enum, Float, ForeignKey, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from geoalchemy2 import Geometry

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class TripStatus(str, PyEnum):
    PLANNED = "planned"
    ACTIVE = "active"
    COMPLETED = "completed"
    SOS_TRIGGERED = "sos_triggered"
    DEVIATED = "deviated"
    STALE = "stale"


class Trip(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "trips"

    user_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True,
    )

    status: Mapped[TripStatus] = mapped_column(
        Enum(TripStatus, name="trip_status", create_constraint=True),
        default=TripStatus.PLANNED, nullable=False,
    )

    # Origin / destination as PostGIS Points
    origin: Mapped[str] = mapped_column(
        Geometry(geometry_type="POINT", srid=4326), nullable=False,
    )
    destination: Mapped[str] = mapped_column(
        Geometry(geometry_type="POINT", srid=4326), nullable=False,
    )

    route_geojson: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    safety_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    locations: Mapped[list["TripLocation"]] = relationship(
        "TripLocation", back_populates="trip", cascade="all, delete-orphan",
        order_by="TripLocation.recorded_at",
    )

    def __repr__(self) -> str:
        return f"<Trip {self.id} status={self.status.value}>"


class TripLocation(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Periodic GPS ping during a live trip."""

    __tablename__ = "trip_locations"

    trip_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("trips.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )

    geometry: Mapped[str] = mapped_column(
        Geometry(geometry_type="POINT", srid=4326, spatial_index=True),
        nullable=False,
    )
    accuracy_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )

    trip: Mapped[Trip] = relationship("Trip", back_populates="locations")

    def __repr__(self) -> str:
        return f"<TripLocation trip={self.trip_id}>"
