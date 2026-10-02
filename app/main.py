"""
SafeWay – main FastAPI application (Step 1: foundation).

Entry point: ``uvicorn app.main:app``
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.core.exceptions import SafeWayError
from app.core.middleware import RateLimitMiddleware
from app.api.v1 import router as v1_router

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "🚀 SafeWay starting | zone=%s (%s) | debug=%s",
        settings.ZONE, settings.zone_preset.display_name, settings.DEBUG,
    )
    yield
    logger.info("🛑 SafeWay shutting down")


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "SafeWay returns the **safest** walking route instead of the fastest, "
        "using crime history, street lighting, crowd activity, and time of day. "
        "Includes shake-gesture SOS, live trip monitoring, and an authority dashboard."
    ),
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────────────────────

app.add_middleware(RateLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Exception handlers ──────────────────────────────────────────────────────

@app.exception_handler(SafeWayError)
async def safeway_error_handler(request: Request, exc: SafeWayError):
    status_map = {
        "ROUTE_NOT_FOUND": 404,
        "ZONE_OUT_OF_BOUNDS": 400,
        "SOS_COOLDOWN": 429,
        "TRIP_NOT_FOUND": 404,
        "TRIP_ALREADY_ENDED": 400,
        "DUPLICATE": 409,
    }
    return JSONResponse(
        status_code=status_map.get(exc.code, 500),
        content={"error": exc.code, "detail": exc.detail},
    )

# ── Routers ──────────────────────────────────────────────────────────────────

app.include_router(v1_router, prefix=settings.API_V1_PREFIX)
from app.api.v1.route import router as route_router
app.include_router(route_router, prefix="", tags=["Routing"])


# ── Root ─────────────────────────────────────────────────────────────────────

@app.get("/", tags=["System"])
def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "zone": settings.ZONE,
        "zone_display": settings.zone_preset.display_name,
    }
