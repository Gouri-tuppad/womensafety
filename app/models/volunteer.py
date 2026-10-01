"""
Volunteer model – registered safety volunteers with availability tracking.
"""

from __future__ import annotations

import uuid

from sqlalchemy import Boolean, Float, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column
from geoalchemy2 import Geometry

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class Volunteer(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "volunteers"

    user_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, unique=True, index=True,
    )

    is_available: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Last known location (nullable until first check-in)
    last_location: Mapped[str | None] = mapped_column(
        Geometry(geometry_type="POINT", srid=4326, spatial_index=True),
        nullable=True,
    )

    radius_m: Mapped[float] = mapped_column(
        Float, default=1000.0, nullable=False,
        doc="Response radius in metres.",
    )
    verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    def __repr__(self) -> str:
        return f"<Volunteer user={self.user_id} avail={self.is_available}>"
