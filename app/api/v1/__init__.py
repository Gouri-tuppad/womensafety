"""API v1 router – aggregates sub-routers for Step 1."""

from fastapi import APIRouter

from app.api.v1.auth import router as auth_router
from app.api.v1.health import router as health_router
from app.api.v1.users import router as users_router

router = APIRouter()
router.include_router(auth_router, prefix="/auth", tags=["Auth"])
router.include_router(health_router, prefix="", tags=["System"])
router.include_router(users_router, prefix="/users", tags=["Users"])
