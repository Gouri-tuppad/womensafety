"""
User and TrustedContact models.
"""

from __future__ import annotations

import uuid
from enum import Enum as PyEnum

from sqlalchemy import Boolean, Enum, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class UserRole(str, PyEnum):
    """Roles a user can hold in the system."""
    WOMAN = "woman"
    CHILD = "child"
    PARENT = "parent"
    VOLUNTEER = "volunteer"
    AUTHORITY = "authority"


class User(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    phone: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)

    role: Mapped[UserRole] = mapped_column(
        Enum(
            UserRole,
            name="user_role",
            create_constraint=True,
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
        default=UserRole.WOMAN,
    )
    trust_score: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    profile_type: Mapped[str | None] = mapped_column(String(60), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Relationships
    trusted_contacts: Mapped[list["TrustedContact"]] = relationship(
        "TrustedContact", back_populates="user", cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<User {self.email!r} role={self.role.value}>"


class TrustedContact(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "trusted_contacts"

    user_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    contact_name: Mapped[str] = mapped_column(String(120), nullable=False)
    contact_phone: Mapped[str] = mapped_column(String(20), nullable=False)
    relationship_label: Mapped[str] = mapped_column(
        String(60), nullable=False, doc="e.g. mother, friend, colleague",
    )
    priority: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    user: Mapped[User] = relationship("User", back_populates="trusted_contacts")

    def __repr__(self) -> str:
        return f"<TrustedContact {self.contact_name!r} for user={self.user_id}>"
