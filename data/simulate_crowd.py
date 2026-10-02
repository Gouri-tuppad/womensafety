"""
data/simulate_crowd.py - Simulate anonymous per-H3-cell hourly crowd counts.

DATA SOURCE: SIMULATED - NOT REAL. All crowd counts are synthetic estimates
based on documented zone characteristics. Clearly flagged in all storage.

H3 resolution 9 cells (~174 m hexagon edge length) are used.
Counts stored in:
  - Redis: key crowd:{zone}:{h3_index}:{YYYY-MM-DD}:{HH} = count (TTL 7 days)
  - PostgreSQL: crowd_counts table (created by this script if absent)

Zone-specific patterns (documented assumptions):
  Peenya (industrial):
    - Peaks: 06:00-07:00 and 14:00-15:00 and 22:00-23:00 (shift changes)
    - Very low: 00:00-05:00 and 23:00+ outside shift hours
    - Near-empty residential at night: <5 people per cell

  Subramanyapura (residential + metro):
    - Peaks: 08:00-10:00 (morning commute on Kanakapura Rd + metro)
             17:00-20:00 (evening return)
    - Thin after 22:00 in interior lanes
    - Metro stop cells have distinct double peak

Usage:
    python -m data.simulate_crowd               # ZONE from env
    python -m data.simulate_crowd peenya        # single zone
    python -m data.simulate_crowd subramanyapura
    python -m data.simulate_crowd --all         # both zones
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import ZONE_PRESETS, Settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("simulate_crowd")

# ── H3 import with fallback ───────────────────────────────────────────────────

try:
    import h3

    def _latlng_to_h3(lat: float, lng: float, res: int = 9) -> str:
        return h3.latlng_to_cell(lat, lng, res)

    def _h3_cells_in_bbox(
        lat_min: float, lat_max: float, lon_min: float, lon_max: float, res: int = 9
    ) -> list[str]:
        """Return H3 cells that cover the bounding box."""
        # Use h3.geo_to_cells with a polygon covering the bbox
        polygon_coords = [
            [lat_min, lon_min],
            [lat_min, lon_max],
            [lat_max, lon_max],
            [lat_max, lon_min],
        ]
        try:
            # h3-py v4+ API
            cells = h3.geo_to_cells({"type": "Polygon", "coordinates": [
                [[lon_min, lat_min], [lon_max, lat_min],
                 [lon_max, lat_max], [lon_min, lat_max], [lon_min, lat_min]]
            ]}, res)
        except AttributeError:
            # h3-py v3 API
            cells = h3.polyfill_geojson({"type": "Polygon", "coordinates": [
                [[lon_min, lat_min], [lon_max, lat_min],
                 [lon_max, lat_max], [lon_min, lat_max], [lon_min, lat_min]]
            ]}, res)
        return list(cells)

    H3_AVAILABLE = True
except ImportError:
    H3_AVAILABLE = False
    log.warning("h3 library not available - will use grid approximation")

    def _latlng_to_h3(lat: float, lng: float, res: int = 9) -> str:  # type: ignore[misc]
        """Fallback: encode as a fixed-precision grid string."""
        return f"grid_{lat:.3f}_{lng:.3f}_r{res}"

    def _h3_cells_in_bbox(
        lat_min: float, lat_max: float, lon_min: float, lon_max: float, res: int = 9
    ) -> list[str]:
        """Fallback: create a regular grid of pseudo-cells."""
        step = 0.002  # ~220 m
        cells = []
        lat = lat_min
        while lat < lat_max:
            lng = lon_min
            while lng < lon_max:
                cells.append(_latlng_to_h3(lat, lng, res))
                lng += step
            lat += step
        return cells


# ── Crowd pattern functions ───────────────────────────────────────────────────

def _peenya_hourly_weight(hour: int, is_near_factory: bool = False) -> float:
    """
    Peenya industrial crowd weight for a given hour (0-23).
    SIMULATED - based on known shift-change patterns.
    """
    # Shift change peaks: 06-07, 14-15, 22-23
    base = [
        0.05, 0.04, 0.03, 0.03, 0.04, 0.10,  # 00-05
        0.85, 0.70, 0.30, 0.20, 0.20, 0.20,  # 06-11
        0.20, 0.20, 0.80, 0.70, 0.35, 0.25,  # 12-17
        0.20, 0.20, 0.20, 0.20, 0.90, 0.75,  # 18-23
    ]
    w = base[hour]
    if is_near_factory:
        # Near factory gates: additional uplift at shift changes
        if hour in (6, 7, 14, 15, 22, 23):
            w = min(1.0, w * 1.3)
    return w


def _subramanyapura_hourly_weight(hour: int, is_near_metro: bool = False) -> float:
    """
    Subramanyapura residential/metro crowd weight for a given hour.
    SIMULATED - based on commuter and residential patterns.
    """
    base = [
        0.04, 0.03, 0.03, 0.03, 0.05, 0.10,  # 00-05
        0.25, 0.55, 0.80, 0.75, 0.40, 0.35,  # 06-11
        0.35, 0.30, 0.30, 0.35, 0.45, 0.65,  # 12-17
        0.85, 0.80, 0.55, 0.30, 0.12, 0.06,  # 18-23
    ]
    w = base[hour]
    if is_near_metro:
        # Metro station cells: sharper double peak
        if 8 <= hour <= 10:
            w = min(1.0, w * 1.5)
        elif 17 <= hour <= 20:
            w = min(1.0, w * 1.5)
    return w


# Metro cell proximity (approximate lat/lon for each zone's metro stations)
METRO_COORDS = {
    "peenya": [
        (13.0285, 77.5086),
        (13.0218, 77.5143),
        (13.0136, 77.5201),
        (13.0280, 77.5007),
    ],
    "subramanyapura": [
        (12.8825, 77.5523),
        (12.8739, 77.5497),
    ],
}

# Industrial hotspot areas in Peenya (for near-factory flag)
PEENYA_INDUSTRIAL_BBOX = (13.010, 13.040, 77.490, 77.530)


def _is_near_metro(h3_cell: str, zone: str, threshold_deg: float = 0.005) -> bool:
    """Rough proximity check to metro stations using cell lat/lng."""
    if not H3_AVAILABLE:
        return False
    try:
        clat, clng = h3.cell_to_latlng(h3_cell)
    except Exception:
        return False
    for mlat, mlng in METRO_COORDS.get(zone, []):
        dist = math.hypot(clat - mlat, clng - mlng)
        if dist < threshold_deg:
            return True
    return False


def _is_near_factory(h3_cell: str) -> bool:
    """Check if H3 cell is within Peenya industrial bbox."""
    if not H3_AVAILABLE:
        return False
    try:
        clat, clng = h3.cell_to_latlng(h3_cell)
    except Exception:
        return False
    lat_min, lat_max, lon_min, lon_max = PEENYA_INDUSTRIAL_BBOX
    return lat_min <= clat <= lat_max and lon_min <= clng <= lon_max


# ── Simulation ────────────────────────────────────────────────────────────────

def _simulate_cell_hour(
    zone: str,
    h3_cell: str,
    hour: int,
    rng: random.Random,
) -> int:
    """
    Simulate crowd count for one H3 cell and one hour.
    SIMULATED DATA - see module docstring for assumptions.
    """
    near_metro = _is_near_metro(h3_cell, zone)
    near_factory = _is_near_factory(h3_cell) if zone == "peenya" else False

    if zone == "peenya":
        weight = _peenya_hourly_weight(hour, is_near_factory=near_factory)
        base_max = 120 if near_factory else 50
    else:  # subramanyapura
        weight = _subramanyapura_hourly_weight(hour, is_near_metro=near_metro)
        base_max = 200 if near_metro else 80

    # Poisson-like: count drawn from Poisson(lambda = weight * base_max)
    lam = weight * base_max
    # Approximate Poisson with Normal for speed (lam > 5)
    if lam < 1:
        count = 0 if rng.random() > lam else 1
    else:
        count = max(0, int(rng.gauss(lam, math.sqrt(lam))))

    return count


# ── Database: crowd_counts table ──────────────────────────────────────────────

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS crowd_counts (
    id          SERIAL PRIMARY KEY,
    zone        VARCHAR(30)  NOT NULL,
    h3_index    VARCHAR(20)  NOT NULL,
    hour_utc    SMALLINT     NOT NULL CHECK (hour_utc BETWEEN 0 AND 23),
    date_utc    DATE         NOT NULL,
    count       INTEGER      NOT NULL DEFAULT 0,
    is_simulated BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (zone, h3_index, date_utc, hour_utc)
);
CREATE INDEX IF NOT EXISTS ix_crowd_zone_date ON crowd_counts (zone, date_utc);
CREATE INDEX IF NOT EXISTS ix_crowd_h3 ON crowd_counts (h3_index);
"""

UPSERT_SQL = """
INSERT INTO crowd_counts (zone, h3_index, hour_utc, date_utc, count, is_simulated)
VALUES (:zone, :h3, :hour, :date, :count, TRUE)
ON CONFLICT (zone, h3_index, date_utc, hour_utc)
DO UPDATE SET count = EXCLUDED.count, is_simulated = TRUE;
"""


def _ensure_table(db_url: str) -> None:
    """Create crowd_counts table if it doesn't exist."""
    from sqlalchemy import create_engine, text
    engine = create_engine(db_url, echo=False)
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))
    log.info("crowd_counts table ensured.")


def _store_db(db_url: str, rows: list[dict]) -> int:
    """Bulk-upsert crowd rows into PostgreSQL. Returns count."""
    from sqlalchemy import create_engine, text
    engine = create_engine(db_url, echo=False)
    written = 0
    with engine.begin() as conn:
        for i in range(0, len(rows), 1000):
            batch = rows[i : i + 1000]
            conn.execute(text(UPSERT_SQL), batch)
            written += len(batch)
    return written


def _store_redis(redis_url: str, rows: list[dict], ttl_seconds: int = 7 * 86400) -> int:
    """Store crowd counts in Redis. Key: crowd:{zone}:{h3}:{date}:{hour}."""
    try:
        import redis as redis_lib
    except ImportError:
        log.warning("redis library not available - skipping Redis storage")
        return 0

    r = redis_lib.from_url(redis_url, decode_responses=True)
    pipeline = r.pipeline(transaction=False)
    for row in rows:
        key = f"crowd:{row['zone']}:{row['h3']}:{row['date']}:{row['hour']:02d}"
        pipeline.set(key, row["count"], ex=ttl_seconds)
    pipeline.execute()
    log.info("Stored %d crowd entries in Redis", len(rows))
    return len(rows)


# ── Main simulation pipeline ──────────────────────────────────────────────────

def simulate_zone(zone: str, db_url: str, redis_url: str, days: int = 7, seed: int = 99) -> dict:
    """
    Simulate crowd for the past `days` days (default 7) for all H3 cells in zone.
    Stores in both PostgreSQL and Redis.
    Returns summary dict.
    """
    if zone not in ZONE_PRESETS:
        raise ValueError(f"Unknown zone: '{zone}'")

    preset = ZONE_PRESETS[zone]
    rng = random.Random(seed)

    # Get all H3 cells in bbox
    cells = _h3_cells_in_bbox(
        lat_min=preset.bbox_south,
        lat_max=preset.bbox_north,
        lon_min=preset.bbox_west,
        lon_max=preset.bbox_east,
        res=9,
    )
    log.info("Zone %s: %d H3 cells to simulate", zone, len(cells))

    # Simulate past `days` x 24 hours
    today = date.today()
    rows: list[dict] = []

    for day_offset in range(days):
        sim_date = today - timedelta(days=day_offset)
        for cell in cells:
            for hour in range(24):
                count = _simulate_cell_hour(zone, cell, hour, rng)
                rows.append({
                    "zone": zone,
                    "h3": cell,
                    "hour": hour,
                    "date": sim_date.isoformat(),
                    "count": count,
                })

    log.info("Simulated %d crowd data points", len(rows))

    # Ensure DB table exists
    _ensure_table(db_url)

    # Store
    n_db = _store_db(db_url, rows)
    log.info("DB rows written: %d", n_db)

    n_redis = _store_redis(redis_url, rows)

    summary = {
        "zone": zone,
        "h3_cells": len(cells),
        "days_simulated": days,
        "total_rows": len(rows),
        "db_rows": n_db,
        "redis_rows": n_redis,
    }
    _print_summary(summary)
    return summary


def _print_summary(s: dict) -> None:
    print("\n" + "=" * 60)
    print(f"  CROWD SIMULATION SUMMARY - zone={s['zone']}")
    print("=" * 60)
    print(f"  H3 cells simulated  : {s['h3_cells']:,}")
    print(f"  Days simulated      : {s['days_simulated']}")
    print(f"  Total data points   : {s['total_rows']:,}")
    print(f"  DB rows written     : {s['db_rows']:,}")
    print(f"  Redis entries       : {s['redis_rows']:,}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate crowd data for SafeWay zones.")
    parser.add_argument("zone", nargs="?", default=None)
    parser.add_argument("--all", action="store_true", help="Simulate all zones")
    parser.add_argument("--days", type=int, default=7, help="Number of past days to simulate")
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args()

    s = Settings()
    zones = list(ZONE_PRESETS.keys()) if args.all else [args.zone or s.ZONE]

    for zone in zones:
        simulate_zone(
            zone=zone,
            db_url=s.DATABASE_URL_SYNC,
            redis_url=s.REDIS_URL,
            days=args.days,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
