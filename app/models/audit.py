"""
AuditLog model – immutable record of security-relevant actions.
"""

from __future__ import annotations

import uuid

from sqlalchemy import ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class AuditLog(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "audit_log"

    user_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True, index=True,
    )

    action: Mapped[str] = mapped_column(
        String(60), nullable=False, doc="e.g. login, sos_trigger, route_request",
    )
    resource_type: Mapped[str] = mapped_column(
        String(60), nullable=False, doc="e.g. user, trip, sos_event",
    )
    resource_id: Mapped[str | None] = mapped_column(String(60), nullable=True)

    details: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)

    def __repr__(self) -> str:
        return f"<AuditLog {self.action} on {self.resource_type}>"
