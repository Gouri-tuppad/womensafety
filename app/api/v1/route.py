"""
app/api/v1/route.py - Routing and safety heatmap endpoints.

POST /route           - compute fastest + safest routes
GET  /safety/heatmap  - H3 risk grid for a zone and hour
GET  /safety/precompute - trigger edge weight precomputation for one time slot
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.deps import get_db, get_redis
from app.schemas.route import HeatmapCell, HeatmapResponse, RouteRequest, RouteResponse
from app.services.routing_service import compute_heatmap, compute_routes
from app.services.safety_scoring import TimeContext

log = logging.getLogger(__name__)
router = APIRouter()


# ── POST /route ───────────────────────────────────────────────────────────────

@router.post(
    "/route",
    response_model=RouteResponse,
    summary="Compute fastest and safest routes",
    description=(
        "Returns both the **fastest** (shortest travel time) and **safest** "
        "(lowest weighted crime/darkness/isolation risk) routes between two points. "
        "The response includes colour-coded segments, safe havens within 300 m of the "
        "safest route, and a comparison summary."
    ),
)
async def route(
    req: RouteRequest,
    redis=Depends(get_redis),
):
    # Validate that origin and destination are in the active zone
    preset = settings.zone_preset
    for name, point in [("origin", req.origin), ("destination", req.destination)]:
        if not (preset.bbox_south <= point.lat <= preset.bbox_north and
                preset.bbox_west <= point.lon <= preset.bbox_east):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"{name} ({point.lat}, {point.lon}) is outside the active zone "
                    f"'{settings.ZONE}' bounding box."
                ),
            )

    time_ctx = TimeContext.from_datetime(req.depart_time, settings.ZONE)

    try:
        # Routing is CPU-bound + sync DB: run in a thread pool
        result = await asyncio.to_thread(
            compute_routes,
            origin_lat=req.origin.lat,
            origin_lon=req.origin.lon,
            dest_lat=req.destination.lat,
            dest_lon=req.destination.lon,
            time_ctx=time_ctx,
            profile_name=req.profile.value,
            db_url=settings.DATABASE_URL_SYNC,
            redis_client=redis,
            zone=settings.ZONE,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except Exception as exc:
        log.exception("Routing error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Routing computation failed. See server logs.",
        )

    return RouteResponse(
        zone=result["zone"],
        profile=req.profile,
        depart_time=req.depart_time,
        fastest=result["fastest"],
        safest=result["safest"],
        routes_differ=result["routes_differ"],
        safe_havens_near_safest=result["safe_havens_near_safest"],
        comparison=result["comparison"],
    )


# ── GET /safety/heatmap ───────────────────────────────────────────────────────

@router.get(
    "/safety/heatmap",
    response_model=HeatmapResponse,
    summary="H3 risk heatmap for a zone and hour",
    description=(
        "Returns per-H3-cell (resolution 9) composite risk scores for the active zone "
        "at the specified hour. Intended to power a time-slider UI overlay on the map."
    ),
)
async def heatmap(
    hour: int = Query(..., ge=0, le=23, description="Local hour (0-23, IST)"),
    zone: str = Query(default="peenya", description="Zone name, default peenya"),
    redis=Depends(get_redis),
):
    if zone not in ("peenya", settings.ZONE):
        zone = settings.ZONE
    preset = settings.zone_preset

    try:
        from datetime import date
        import pytz
        _ist = pytz.timezone("Asia/Kolkata")
        today = date.today()
        ref_dt = _ist.localize(
            __import__("datetime").datetime(today.year, today.month, today.day, hour, 0, 0)
        )
        time_ctx = TimeContext.from_datetime(ref_dt, zone)
    except Exception as exc:
        log.warning("TimeContext build failed for hour=%d: %s", hour, exc)
        is_dark = 19 <= hour or hour <= 6  # simple fallback
        is_weekend = False
        time_ctx = None  # used for response only

    cells_raw = await asyncio.to_thread(
        compute_heatmap,
        zone=zone,
        hour=hour,
        db_url=settings.DATABASE_URL_SYNC,
        redis_client=redis,
    )

    cells = [
        HeatmapCell(
            h3_index=c["h3_index"],
            lat=c["lat"],
            lon=c["lon"],
            risk_score=c["risk_score"],
            crime_risk=c["crime_risk"],
            darkness_risk=c["darkness_risk"],
            isolation_risk=c["isolation_risk"],
            crowd_count=c["crowd_count"],
        )
        for c in cells_raw
    ]

    return HeatmapResponse(
        zone=zone,
        hour=hour,
        is_dark=time_ctx.is_dark if time_ctx else (19 <= hour or hour <= 6),
        is_weekend=time_ctx.is_weekend if time_ctx else False,
        cells=cells,
        cell_count=len(cells),
    )


# ── GET /safety/precompute ────────────────────────────────────────────────────

@router.get(
    "/safety/precompute",
    summary="Precompute edge weights for a time slot",
    description=(
        "Triggers synchronous edge-weight precomputation for the active zone at a given "
        "hour and day type. Results are cached in Redis. "
        "Runs in a background thread (returns immediately on launch)."
    ),
)
async def precompute(
    hour: int = Query(..., ge=0, le=23),
    is_weekend: bool = Query(False),
    background: bool = Query(False, description="If true, return immediately without waiting"),
    redis=Depends(get_redis),
):
    from app.services.safety_scoring import precompute_zone_weights

    zone = settings.ZONE

    async def _run():
        n = await asyncio.to_thread(
            precompute_zone_weights,
            zone=zone,
            hour=hour,
            is_weekend=is_weekend,
            db_url=settings.DATABASE_URL_SYNC,
            redis_url=settings.REDIS_URL,
        )
        log.info("Precompute finished: %d segments, zone=%s, hour=%d", n, zone, hour)
        return n

    if background:
        asyncio.create_task(_run())
        return {"status": "launched", "zone": zone, "hour": hour, "is_weekend": is_weekend}

    n = await _run()
    return {"status": "done", "zone": zone, "hour": hour, "segments_processed": n}
