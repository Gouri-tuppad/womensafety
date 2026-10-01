"""Pydantic schemas for authentication and registration."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    """Body for ``POST /auth/register``."""

    email: str = Field(..., min_length=5, max_length=255)
    phone: str = Field(..., min_length=10, max_length=20)
    password: str = Field(..., min_length=8, max_length=128)
    name: str = Field(..., min_length=1, max_length=120)
    role: str = Field(
        default="woman",
        pattern="^(woman|child|parent|volunteer|authority)$",
    )


class LoginRequest(BaseModel):
    """Body for ``POST /auth/login`` (JSON variant)."""

    email: str
    password: str


class TokenResponse(BaseModel):
    """JWT token response."""

    access_token: str
    token_type: str = "bearer"
    user_id: uuid.UUID
    role: str
    expires_in_minutes: int


class UserOut(BaseModel):
    """Public user representation."""

    id: uuid.UUID
    email: str
    phone: str
    name: str
    role: str
    trust_score: float
    profile_type: str | None
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}
