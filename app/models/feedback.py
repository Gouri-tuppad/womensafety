"""
Feedback model – post-trip safety and route quality ratings.
"""

from __future__ import annotations

import uuid

from sqlalchemy import ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class Feedback(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "feedback"

    user_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    trip_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("trips.id", ondelete="SET NULL"),
        nullable=True,
    )

    route_rating: Mapped[int] = mapped_column(
        Integer, nullable=False, doc="1-5 stars for route quality",
    )
    safety_rating: Mapped[int] = mapped_column(
        Integer, nullable=False, doc="1-5 stars for perceived safety",
    )
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<Feedback user={self.user_id} route={self.route_rating} safety={self.safety_rating}>"
