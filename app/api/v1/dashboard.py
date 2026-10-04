"""
app/api/v1/dashboard.py - Authority dashboard endpoints.

GET /dashboard/fairness  - fairness audit report (authority role only)
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.config import settings
from app.core.security import require_roles

log = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/dashboard/fairness",
    summary="Fairness audit report",
    description=(
        "Returns the ML risk fairness audit comparing predicted risk across "
        "neighborhoods, grouped by affluence proxy. Authority role required. "
        "Triggers a fresh audit if no cached report exists."
    ),
    dependencies=[Depends(require_roles("authority"))],
)
async def fairness_dashboard():
    """Return the fairness audit report (authority role only)."""
    from pathlib import Path
    import json

    report_path = Path("ml/reports/fairness.json")

    # Try cached report first
    if report_path.exists():
        with open(report_path) as f:
            return json.load(f)

    # Run audit on demand
    try:
        from app.ml.fairness_audit import run_fairness_audit
        report = await asyncio.to_thread(
            run_fairness_audit,
            zone=settings.ZONE,
            db_url=settings.DATABASE_URL_SYNC,
        )
        return report
    except Exception as exc:
        log.exception("Fairness audit failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fairness audit failed: {exc}",
        )
