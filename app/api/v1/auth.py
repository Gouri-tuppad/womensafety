"""
auth.py – Register, login (JWT), and token endpoints.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.security import create_access_token, hash_password, verify_password
from app.deps import get_db
from app.models.user import User, UserRole
from app.schemas.auth import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserOut,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/register",
    response_model=UserOut,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """Create a new user account with hashed password."""
    # Check for duplicate email
    existing = await db.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered.",
        )

    # Check for duplicate phone
    existing_phone = await db.execute(select(User).where(User.phone == body.phone))
    if existing_phone.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Phone number already registered.",
        )

    user = User(
        email=body.email,
        phone=body.phone,
        hashed_password=hash_password(body.password),
        name=body.name,
        role=UserRole(body.role),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    logger.info("User registered: %s (%s)", user.email, user.role.value)
    return user


async def _authenticate(email: str, password: str, db: AsyncSession) -> TokenResponse:
    """Shared auth logic for both login endpoints."""
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if user is None or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled.",
        )

    token = create_access_token(user.id, user.role.value)
    logger.info("Login: %s", user.email)
    return TokenResponse(
        access_token=token,
        user_id=user.id,
        role=user.role.value,
        expires_in_minutes=settings.JWT_EXPIRE_MINUTES,
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login and receive a JWT",
    description="Accepts JSON body with email and password.",
)
async def login(
    body: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    """JSON-body login for API clients."""
    return await _authenticate(body.email, body.password, db)


@router.post(
    "/token",
    response_model=TokenResponse,
    summary="OAuth2 form token endpoint",
    description="OAuth2-compatible endpoint for Swagger Authorize button.",
)
async def login_token(
    form: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    """OAuth2-compatible form login."""
    return await _authenticate(form.username, form.password, db)

