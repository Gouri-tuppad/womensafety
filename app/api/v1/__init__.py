"""API v1 router – aggregates sub-routers for Step 1."""

from fastapi import APIRouter

from app.api.v1.auth import router as auth_router
from app.api.v1.health import router as health_router
from app.api.v1.route import router as route_router
from app.api.v1.users import router as users_router
from app.api.v1.dashboard import router as dashboard_router
from app.api.v1.sos import router as sos_router
from app.api.v1.trips import router as trips_router
from app.api.v1.reports import router as reports_router
from app.api.v1.child import router as child_router
from app.api.v1.volunteers import router as volunteers_router

router = APIRouter()
router.include_router(auth_router, prefix="/auth", tags=["Auth"])
router.include_router(health_router, prefix="", tags=["System"])
router.include_router(route_router, prefix="", tags=["Routing"])
router.include_router(users_router, prefix="/users", tags=["Users"])
router.include_router(dashboard_router, prefix="", tags=["Dashboard"])
router.include_router(sos_router, prefix="", tags=["SOS"])
router.include_router(trips_router, prefix="/trips", tags=["Trips"])
router.include_router(reports_router, prefix="/reports", tags=["Reports"])
router.include_router(child_router, prefix="/child", tags=["Child Safety"])
router.include_router(volunteers_router, prefix="/volunteers", tags=["Volunteers"])



