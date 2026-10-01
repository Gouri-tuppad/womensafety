"""Pydantic schemas for user profile and trusted-contacts CRUD."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


# ── Profile ──────────────────────────────────────────────────────────────────

class ProfileUpdate(BaseModel):
    """Partial update for a user's profile."""

    name: str | None = Field(None, min_length=1, max_length=120)
    phone: str | None = Field(None, min_length=10, max_length=20)
    profile_type: str | None = Field(None, max_length=60)


# ── Trusted Contacts ─────────────────────────────────────────────────────────

class TrustedContactCreate(BaseModel):
    contact_name: str = Field(..., min_length=1, max_length=120)
    contact_phone: str = Field(..., min_length=10, max_length=20)
    relationship_label: str = Field(..., min_length=1, max_length=60)
    priority: int = Field(default=1, ge=1, le=10)


class TrustedContactUpdate(BaseModel):
    contact_name: str | None = None
    contact_phone: str | None = None
    relationship_label: str | None = None
    priority: int | None = Field(None, ge=1, le=10)


class TrustedContactOut(BaseModel):
    id: uuid.UUID
    contact_name: str
    contact_phone: str
    relationship_label: str
    priority: int
    created_at: datetime

    model_config = {"from_attributes": True}
