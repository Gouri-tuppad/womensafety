"""
users.py – Profile endpoints + trusted-contacts CRUD.
"""

from __future__ import annotations

import uuid
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.deps import get_current_user, get_db
from app.models.user import TrustedContact, User
from app.schemas.auth import UserOut
from app.schemas.user import (
    ProfileUpdate,
    TrustedContactCreate,
    TrustedContactOut,
    TrustedContactUpdate,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Profile ──────────────────────────────────────────────────────────────────

@router.get("/me", response_model=UserOut, summary="Get current user profile")
async def get_profile(user: User = Depends(get_current_user)):
    return user


@router.patch("/me", response_model=UserOut, summary="Update current user profile")
async def update_profile(
    body: ProfileUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if body.name is not None:
        user.name = body.name
    if body.phone is not None:
        user.phone = body.phone
    if body.profile_type is not None:
        user.profile_type = body.profile_type
    await db.commit()
    await db.refresh(user)
    return user


# ── Trusted Contacts ─────────────────────────────────────────────────────────

@router.get(
    "/me/contacts",
    response_model=list[TrustedContactOut],
    summary="List trusted contacts",
)
async def list_contacts(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(TrustedContact)
        .where(TrustedContact.user_id == user.id)
        .order_by(TrustedContact.priority)
    )
    return result.scalars().all()


@router.post(
    "/me/contacts",
    response_model=TrustedContactOut,
    status_code=status.HTTP_201_CREATED,
    summary="Add a trusted contact",
)
async def create_contact(
    body: TrustedContactCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    contact = TrustedContact(
        user_id=user.id,
        contact_name=body.contact_name,
        contact_phone=body.contact_phone,
        relationship_label=body.relationship_label,
        priority=body.priority,
    )
    db.add(contact)
    await db.commit()
    await db.refresh(contact)
    return contact


@router.patch(
    "/me/contacts/{contact_id}",
    response_model=TrustedContactOut,
    summary="Update a trusted contact",
)
async def update_contact(
    contact_id: uuid.UUID,
    body: TrustedContactUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(TrustedContact).where(
            TrustedContact.id == contact_id,
            TrustedContact.user_id == user.id,
        )
    )
    contact = result.scalar_one_or_none()
    if not contact:
        raise HTTPException(status_code=404, detail="Contact not found.")

    if body.contact_name is not None:
        contact.contact_name = body.contact_name
    if body.contact_phone is not None:
        contact.contact_phone = body.contact_phone
    if body.relationship_label is not None:
        contact.relationship_label = body.relationship_label
    if body.priority is not None:
        contact.priority = body.priority

    await db.commit()
    await db.refresh(contact)
    return contact


@router.delete(
    "/me/contacts/{contact_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a trusted contact",
)
async def delete_contact(
    contact_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(TrustedContact).where(
            TrustedContact.id == contact_id,
            TrustedContact.user_id == user.id,
        )
    )
    contact = result.scalar_one_or_none()
    if not contact:
        raise HTTPException(status_code=404, detail="Contact not found.")

    await db.delete(contact)
    await db.commit()
