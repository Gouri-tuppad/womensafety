"""
SOSEvent, SOSLocation, and EvidenceFile models.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import DateTime, Enum, Float, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from geoalchemy2 import Geometry

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class SOSStatus(str, PyEnum):
    COUNTDOWN = "countdown"
    ACTIVE = "active"
    CANCELLED = "cancelled"
    RESOLVED = "resolved"


class SOSEvent(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "sos_events"

    user_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True,
    )
    trip_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("trips.id"), nullable=True,
    )

    status: Mapped[SOSStatus] = mapped_column(
        Enum(
            SOSStatus,
            name="sos_status",
            create_constraint=True,
            values_callable=lambda e: [m.value for m in e],
        ),
        default=SOSStatus.COUNTDOWN, nullable=False,
    )
    trigger_type: Mapped[str] = mapped_column(
        String(30), nullable=False,
        doc="shake, button, voice, auto",
    )
    confidence: Mapped[float] = mapped_column(
        Float, default=1.0, nullable=False,
        doc="0-1 confidence that the SOS is genuine (sensor-based).",
    )

    triggered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    locations: Mapped[list["SOSLocation"]] = relationship(
        "SOSLocation", back_populates="sos_event", cascade="all, delete-orphan",
    )
    evidence_files: Mapped[list["EvidenceFile"]] = relationship(
        "EvidenceFile", back_populates="sos_event", cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<SOSEvent {self.id} status={self.status.value}>"


class SOSLocation(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Location trail during an active SOS event."""

    __tablename__ = "sos_locations"

    sos_event_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("sos_events.id", ondelete="CASCADE"),
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

    sos_event: Mapped[SOSEvent] = relationship("SOSEvent", back_populates="locations")


class EvidenceFile(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Audio/video/image evidence attached to an SOS event or report."""

    __tablename__ = "evidence_files"

    sos_event_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("sos_events.id", ondelete="SET NULL"),
        nullable=True,
    )
    report_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("reports.id", ondelete="SET NULL"),
        nullable=True,
    )

    file_type: Mapped[str] = mapped_column(
        String(20), nullable=False, doc="audio, video, image",
    )
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)

    sos_event: Mapped[SOSEvent | None] = relationship("SOSEvent", back_populates="evidence_files")
