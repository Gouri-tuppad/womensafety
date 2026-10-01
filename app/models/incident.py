"""
Incident and Report models.

Geometry: PostGIS Point (SRID 4326) with spatial index.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import (
    Boolean, DateTime, Enum, Float, ForeignKey, Index,
    Integer, String, Text,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from geoalchemy2 import Geometry

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class IncidentSource(str, PyEnum):
    OFFICIAL = "official"
    NEWS = "news"
    USER = "user"
    SYNTHETIC = "synthetic"


class Incident(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "incidents"

    # PostGIS Point (WGS-84)
    geometry: Mapped[str] = mapped_column(
        Geometry(geometry_type="POINT", srid=4326, spatial_index=True),
        nullable=False,
    )

    incident_type: Mapped[str] = mapped_column(
        String(60), nullable=False,
        doc="harassment, robbery, assault, stalking, chain_snatching, eve_teasing, …",
    )
    severity: Mapped[int] = mapped_column(
        Integer, nullable=False, doc="1 (minor) to 5 (critical)",
    )
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source: Mapped[IncidentSource] = mapped_column(
        Enum(IncidentSource, name="incident_source", create_constraint=True),
        nullable=False,
    )
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    zone: Mapped[str] = mapped_column(String(30), index=True, nullable=False)
    verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    __table_args__ = (
        Index("ix_incident_zone_type", "zone", "incident_type"),
        Index("ix_incident_occurred", "occurred_at"),
    )

    def __repr__(self) -> str:
        return f"<Incident {self.incident_type} sev={self.severity}>"


class ReportStatus(str, PyEnum):
    PENDING = "pending"
    REVIEWED = "reviewed"
    VERIFIED = "verified"
    REJECTED = "rejected"


class Report(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "reports"

    user_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True,
    )
    incident_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("incidents.id"), nullable=True,
    )

    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)

    # Optional location for the report itself
    geometry: Mapped[str | None] = mapped_column(
        Geometry(geometry_type="POINT", srid=4326, spatial_index=False),
        nullable=True,
    )

    status: Mapped[ReportStatus] = mapped_column(
        Enum(ReportStatus, name="report_status", create_constraint=True),
        default=ReportStatus.PENDING,
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<Report {self.title!r} status={self.status.value}>"
