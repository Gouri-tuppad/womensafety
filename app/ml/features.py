"""
app/ml/features.py - Build the ML feature table per (segment, time_slot).

Each row represents one road segment evaluated at a specific time slot.  The
feature set is intentionally rich in **environmental / spatial** features so
that segments with zero incident history still receive meaningful risk scores.

Features:
  Temporal:    hour, day_of_week, is_weekend, is_holiday, is_dark
  Lighting:    lamp_density (lamps per 100 m), lit_ratio
  POI counts:  shops/ATMs within 150 m, petrol within 200 m, metro within 300 m,
               police within 300 m, hospital within 300 m, bus_stop within 150 m
  Infrastructure: distance to main road, distance to nearest police station,
                  distance to nearest hospital, distance to nearest metro,
                  road type (encoded), building/landuse density proxy
  Crime:       recency-decayed past-incident count within 500 m
  Crowd:       crowd_count for the H3 cell at that hour

Target:
  risk_label   - composite risk score [0, 1] from the heuristic scorer, used
                 as the regression target for the ML model.

DATA SOURCE: The features are derived from seeded (SYNTHETIC) data.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shapely.geometry as sg
from shapely import STRtree, box
from sqlalchemy import create_engine, text

from app.config import ZONE_PRESETS, settings
from app.services.safety_scoring import (
    CRIME_RADIUS_M,
    IncidentDensityProvider,
    SegmentFeatures,
    SegmentScorer,
    TimeContext,
)

log = logging.getLogger(__name__)

# ── Feature column definitions ────────────────────────────────────────────────

TEMPORAL_COLS = ["hour", "day_of_week", "is_weekend", "is_holiday", "is_dark"]
LIGHTING_COLS = ["lamp_density", "lit_ratio"]
POI_COLS = [
    "shop_count_150m", "atm_count_150m", "petrol_count_200m",
    "bus_stop_count_150m", "metro_count_300m", "police_count_300m",
    "hospital_count_300m", "factory_gate_count_150m",
]
INFRA_COLS = [
    "dist_main_road_m", "dist_police_m", "dist_hospital_m", "dist_metro_m",
    "road_type_encoded", "building_density_proxy",
]
CROWD_COLS = ["crowd_count"]
CRIME_COLS = ["incident_count_500m_decayed"]
TARGET_COL = "risk_label"

ALL_FEATURE_COLS = TEMPORAL_COLS + LIGHTING_COLS + POI_COLS + INFRA_COLS + CROWD_COLS + CRIME_COLS
FEATURE_COLS_WITH_TARGET = ALL_FEATURE_COLS + [TARGET_COL]

# ── Road type encoding ────────────────────────────────────────────────────────

_ROAD_TYPE_MAP = {
    "motorway": 0, "motorway_link": 0,
    "trunk": 1, "trunk_link": 1,
    "primary": 2, "primary_link": 2,
    "secondary": 3, "secondary_link": 3,
    "tertiary": 4, "tertiary_link": 4,
    "residential": 5, "living_street": 5,
    "unclassified": 6,
    "service": 7,
    "industrial": 8,
    "track": 9, "path": 9, "footway": 9, "cycleway": 9, "steps": 9,
}


def _encode_road_type(rt: str) -> int:
    return _ROAD_TYPE_MAP.get(rt.lower().strip(), 6)


# ── Distance helpers ──────────────────────────────────────────────────────────

def _mpd(lat: float) -> float:
    """Metres per degree at given latitude."""
    return (
        111_132.954
        - 559.822 * math.cos(2 * math.radians(lat))
        + 1.175 * math.cos(4 * math.radians(lat))
    )


def _nearest_distance_m(
    tree: STRtree | None,
    points: list[sg.Point],
    pt: sg.Point,
    mpd_val: float,
) -> float:
    """Return distance in metres to nearest point in tree, or 9999.0 if empty."""
    if tree is None or not points:
        return 9999.0
    idx = tree.nearest(pt)
    if idx is None:
        return 9999.0
    return float(pt.distance(points[idx])) * mpd_val


def _is_db_reachable(db_url: str) -> bool:
    import socket
    from urllib.parse import urlparse
    try:
        clean = db_url.replace("postgresql+psycopg://", "http://").replace("postgresql://", "http://")
        parsed = urlparse(clean)
        host = parsed.hostname or "localhost"
        port = parsed.port or 5432
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def generate_synthetic_feature_table(
    zone: str = "peenya",
    hours: list[int] | None = None,
    include_weekends: bool = True,
    n_segments: int = 150,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a realistic synthetic feature table for training/testing when DB is offline."""
    np.random.seed(seed)
    if hours is None:
        hours = list(range(24))

    slots = []
    for h in hours:
        slots.append((h, 0, 0))  # weekday
        if include_weekends:
            slots.append((h, 1, 5))  # weekend

    n_slots = len(slots)
    n_rows = n_segments * n_slots

    osm_ids = np.tile(np.arange(1001, 1001 + n_segments), n_slots)
    time_slots = []
    slot_hours = []
    slot_is_weekends = []
    slot_days = []
    slot_is_darks = []

    for h, is_wk, dow in slots:
        slot_str = f"{h:02d}{'W' if is_wk else 'D'}"
        is_dark = 1 if (h >= 19 or h < 6) else 0
        time_slots.extend([slot_str] * n_segments)
        slot_hours.extend([h] * n_segments)
        slot_is_weekends.extend([is_wk] * n_segments)
        slot_days.extend([dow] * n_segments)
        slot_is_darks.extend([is_dark] * n_segments)

    # Lighting
    lamp_density = np.random.uniform(0.0, 4.5, n_rows)
    lit_ratio = np.random.uniform(0.0, 1.0, n_rows)

    # POIs
    poi_data = {
        "shop_count_150m": np.random.randint(0, 8, n_rows),
        "atm_count_150m": np.random.randint(0, 5, n_rows),
        "petrol_count_200m": np.random.randint(0, 3, n_rows),
        "bus_stop_count_150m": np.random.randint(0, 4, n_rows),
        "metro_count_300m": np.random.randint(0, 2, n_rows),
        "police_count_300m": np.random.randint(0, 2, n_rows),
        "hospital_count_300m": np.random.randint(0, 2, n_rows),
        "factory_gate_count_150m": np.random.randint(0, 6, n_rows),
    }

    # Infra
    dist_main_road_m = np.random.uniform(10.0, 1500.0, n_rows)
    dist_police_m = np.random.uniform(100.0, 4000.0, n_rows)
    dist_hospital_m = np.random.uniform(100.0, 4000.0, n_rows)
    dist_metro_m = np.random.uniform(100.0, 3500.0, n_rows)
    road_type_encoded = np.random.randint(0, 10, n_rows)
    building_density_proxy = np.random.randint(0, 25, n_rows)

    # Crowd & Crime
    crowd_count = np.random.randint(0, 100, n_rows)
    incident_count_500m_decayed = np.random.uniform(0.0, 4.0, n_rows)

    # Synthetic risk label
    risk = (
        0.30 * np.array(slot_is_darks)
        + 0.25 * (1.0 - lit_ratio)
        + 0.15 * (incident_count_500m_decayed / 4.0)
        + 0.10 * (road_type_encoded / 9.0)
        + 0.10 * (1.0 - crowd_count / 100.0)
        + 0.10 * np.clip(dist_police_m / 4000.0, 0.0, 1.0)
        + np.random.normal(0.0, 0.03, n_rows)
    )
    risk_label = np.clip(risk, 0.0, 1.0)

    data = {
        "osm_id": osm_ids,
        "time_slot": time_slots,
        "hour": slot_hours,
        "day_of_week": slot_days,
        "is_weekend": slot_is_weekends,
        "is_holiday": np.zeros(n_rows, dtype=int),
        "is_dark": slot_is_darks,
        "lamp_density": np.round(lamp_density, 3),
        "lit_ratio": np.round(lit_ratio, 3),
        **poi_data,
        "dist_main_road_m": np.round(dist_main_road_m, 1),
        "dist_police_m": np.round(dist_police_m, 1),
        "dist_hospital_m": np.round(dist_hospital_m, 1),
        "dist_metro_m": np.round(dist_metro_m, 1),
        "road_type_encoded": road_type_encoded,
        "building_density_proxy": building_density_proxy,
        "crowd_count": crowd_count,
        "incident_count_500m_decayed": np.round(incident_count_500m_decayed, 4),
        TARGET_COL: np.round(risk_label, 4),
    }
    df = pd.DataFrame(data)
    log.info("Generated synthetic feature table: %d rows (%d segments × %d slots)", len(df), n_segments, n_slots)
    return df


# ── Main feature builder ─────────────────────────────────────────────────────

def build_feature_table(
    zone: str,
    db_url: str,
    hours: list[int] | None = None,
    include_weekends: bool = True,
    reference_date: date | None = None,
) -> pd.DataFrame:
    """
    Build a feature DataFrame for all (segment, time_slot) combinations.

    Args:
        zone: zone name ('peenya' or 'subramanyapura')
        db_url: sync SQLAlchemy DB URL
        hours: which hours to generate features for (default: all 24)
        include_weekends: whether to include weekend slots
        reference_date: base date for temporal features (default: today)

    Returns:
        DataFrame with columns: osm_id, time_slot, <ALL_FEATURE_COLS>, risk_label
    """
    import pytz

    if hours is None:
        hours = list(range(24))
    if reference_date is None:
        reference_date = date.today()

    if not _is_db_reachable(db_url):
        log.warning("Database at %s not reachable; using synthetic feature table for zone=%s", db_url, zone)
        return generate_synthetic_feature_table(zone=zone, hours=hours, include_weekends=include_weekends)

    _ist = pytz.timezone("Asia/Kolkata")
    engine = create_engine(db_url, echo=False)

    # ── Load data from DB ─────────────────────────────────────────────────

    try:
        with engine.connect() as conn:
            # 1. Road segments
            seg_rows = conn.execute(text(
                "SELECT osm_id, road_type, length_m, lit_ratio, lamp_count, "
                "ST_X(ST_Centroid(geometry)) AS lon, ST_Y(ST_Centroid(geometry)) AS lat "
                "FROM road_segments WHERE zone = :z"
            ), {"z": zone}).fetchall()
            segments = [dict(r._mapping) for r in seg_rows]
            if not segments:
                log.warning("No road segments in DB for zone=%s; falling back to synthetic", zone)
                return generate_synthetic_feature_table(zone=zone, hours=hours, include_weekends=include_weekends)
            log.info("Loaded %d segments for zone=%s", len(segments), zone)
    except Exception as exc:
        log.warning("Database query failed (%s); using synthetic feature table", exc)
        return generate_synthetic_feature_table(zone=zone, hours=hours, include_weekends=include_weekends)

    with engine.connect() as conn:
        # 1. Road segments
        seg_rows = conn.execute(text(
            "SELECT osm_id, road_type, length_m, lit_ratio, lamp_count, "
            "ST_X(ST_Centroid(geometry)) AS lon, ST_Y(ST_Centroid(geometry)) AS lat "
            "FROM road_segments WHERE zone = :z"
        ), {"z": zone}).fetchall()
        segments = [dict(r._mapping) for r in seg_rows]
        log.info("Loaded %d segments for zone=%s", len(segments), zone)

        # 2. POIs by category
        poi_rows = conn.execute(text(
            "SELECT category, ST_X(geometry) AS lon, ST_Y(geometry) AS lat "
            "FROM pois WHERE zone = :z"
        ), {"z": zone}).fetchall()
        pois = [dict(r._mapping) for r in poi_rows]

        # 3. Incidents (past 365 days)
        inc_rows = conn.execute(text(
            "SELECT ST_X(geometry) AS lon, ST_Y(geometry) AS lat, "
            "severity, occurred_at, "
            "EXTRACT(HOUR FROM occurred_at AT TIME ZONE 'Asia/Kolkata') AS hour "
            "FROM incidents WHERE zone = :z "
            "AND occurred_at >= NOW() - INTERVAL '365 days'"
        ), {"z": zone}).fetchall()
        incidents_raw = [dict(r._mapping) for r in inc_rows]

        # 4. Crowd counts (all hours, most recent day per hour)
        crowd_rows = conn.execute(text(
            "SELECT h3_index, hour_utc, count FROM crowd_counts "
            "WHERE zone = :z ORDER BY date_utc DESC LIMIT 500000"
        ), {"z": zone}).fetchall()

        # 5. User Feedback (training signal)
        try:
            feedback_rows = conn.execute(text(
                "SELECT f.safety_rating, t.route_geojson "
                "FROM feedback f JOIN trips t ON f.trip_id = t.id "
                "WHERE t.route_geojson IS NOT NULL"
            )).fetchall()
        except Exception:
            feedback_rows = []

    # Build crowd lookup: {(h3_index, hour): count}
    crowd_lookup: dict[tuple[str, int], int] = {}
    for r in crowd_rows:
        m = r._mapping
        key = (m["h3_index"], int(m["hour_utc"]))
        if key not in crowd_lookup:
            crowd_lookup[key] = int(m["count"])

    # Build feedback lookup: {osm_id: [normalized_risk, ...]}
    feedback_by_segment: dict[int, list[float]] = {}
    for fb in feedback_rows:
        try:
            s_rating = fb._mapping.get("safety_rating", 3)
            # 1 star = 1.0 (very unsafe), 5 stars = 0.0 (very safe)
            fb_risk = 1.0 - (float(s_rating) - 1.0) / 4.0
            r_json = fb._mapping.get("route_geojson")
            if isinstance(r_json, str):
                import json as _json
                r_json = _json.loads(r_json)
            if isinstance(r_json, dict):
                feats = r_json.get("features", [])
                for f in feats:
                    props = f.get("properties", {})
                    osm_val = props.get("osm_id") or props.get("segment_id")
                    if osm_val is not None:
                        feedback_by_segment.setdefault(int(osm_val), []).append(fb_risk)
        except Exception:
            pass

    # ── Build spatial trees ───────────────────────────────────────────────

    poi_by_cat: dict[str, list[sg.Point]] = {}
    for p in pois:
        cat = str(p["category"]).lower()
        poi_by_cat.setdefault(cat, []).append(sg.Point(p["lon"], p["lat"]))

    cat_trees: dict[str, tuple[STRtree, list[sg.Point]]] = {}
    for cat, pts in poi_by_cat.items():
        cat_trees[cat] = (STRtree(pts), pts)

    # Incident spatial tree
    inc_pts = [sg.Point(i["lon"], i["lat"]) for i in incidents_raw]
    inc_tree = STRtree(inc_pts) if inc_pts else None

    # Main road segments for distance calculation
    main_pts = [
        sg.Point(s["lon"], s["lat"])
        for s in segments
        if s["road_type"].lower() in ("primary", "secondary", "trunk")
    ]
    main_tree = STRtree(main_pts) if main_pts else None

    # H3 helper
    try:
        import h3 as _h3
        def _h3_cell(lat: float, lon: float) -> str:
            return _h3.latlng_to_cell(lat, lon, 9)
    except ImportError:
        def _h3_cell(lat: float, lon: float) -> str:
            return f"grid_{lat:.3f}_{lon:.3f}_r9"

    # ── Crime density provider for target ─────────────────────────────────
    scorer = SegmentScorer()

    # ── Iterate over segments × time slots ────────────────────────────────

    rows: list[dict] = []
    day_types = [False]  # weekday
    if include_weekends:
        day_types.append(True)

    for is_wkend in day_types:
        # Find a reference date for this day type
        if is_wkend:
            days_to_sat = (5 - reference_date.weekday()) % 7
            ref_date = reference_date + timedelta(days=days_to_sat)
        else:
            days_to_mon = (0 - reference_date.weekday()) % 7
            ref_date = reference_date + timedelta(days=days_to_mon)

        for hour in hours:
            ref_dt = _ist.localize(
                datetime(ref_date.year, ref_date.month, ref_date.day, hour, 0, 0)
            )
            time_ctx = TimeContext.from_datetime(ref_dt, zone)
            time_slot = time_ctx.time_slot

            for seg in segments:
                osm_id = int(seg["osm_id"])
                lat, lon = float(seg["lat"]), float(seg["lon"])
                mpd_val = _mpd(lat)
                pt = sg.Point(lon, lat)
                length_m = float(seg["length_m"])

                # ── Temporal features ─────────────────────────────────────
                feat: dict = {
                    "osm_id": osm_id,
                    "time_slot": time_slot,
                    "hour": hour,
                    "day_of_week": ref_dt.weekday(),
                    "is_weekend": int(is_wkend),
                    "is_holiday": int(time_ctx.is_holiday),
                    "is_dark": int(time_ctx.is_dark),
                }

                # ── Lighting features ─────────────────────────────────────
                feat["lit_ratio"] = float(seg["lit_ratio"])
                feat["lamp_density"] = float(seg["lamp_count"]) / max(length_m / 100.0, 0.01)

                # ── POI counts within buffers ─────────────────────────────
                def _count_within(cat: str, buffer_m: float) -> int:
                    tree_pts = cat_trees.get(cat)
                    if tree_pts is None:
                        return 0
                    tree, _ = tree_pts
                    d = buffer_m / mpd_val
                    return len(tree.query(box(lon - d, lat - d, lon + d, lat + d)))

                feat["shop_count_150m"] = _count_within("shop", 150.0)
                feat["atm_count_150m"] = _count_within("atm", 150.0)
                feat["petrol_count_200m"] = _count_within("petrol", 200.0)
                feat["bus_stop_count_150m"] = _count_within("bus_stop", 150.0)
                feat["metro_count_300m"] = _count_within("metro", 300.0)
                feat["police_count_300m"] = _count_within("police", 300.0)
                feat["hospital_count_300m"] = _count_within("hospital", 300.0)
                feat["factory_gate_count_150m"] = _count_within("factory_gate", 150.0)

                # ── Infrastructure distances ──────────────────────────────
                feat["dist_main_road_m"] = _nearest_distance_m(main_tree, main_pts, pt, mpd_val)

                police_tree_pts = cat_trees.get("police")
                feat["dist_police_m"] = _nearest_distance_m(
                    police_tree_pts[0] if police_tree_pts else None,
                    police_tree_pts[1] if police_tree_pts else [],
                    pt, mpd_val,
                )

                hospital_tree_pts = cat_trees.get("hospital")
                feat["dist_hospital_m"] = _nearest_distance_m(
                    hospital_tree_pts[0] if hospital_tree_pts else None,
                    hospital_tree_pts[1] if hospital_tree_pts else [],
                    pt, mpd_val,
                )

                metro_tree_pts = cat_trees.get("metro")
                feat["dist_metro_m"] = _nearest_distance_m(
                    metro_tree_pts[0] if metro_tree_pts else None,
                    metro_tree_pts[1] if metro_tree_pts else [],
                    pt, mpd_val,
                )

                feat["road_type_encoded"] = _encode_road_type(seg["road_type"])

                # Building density proxy: total POI count within 200 m
                total_poi_200m = 0
                for cat_name in poi_by_cat:
                    total_poi_200m += _count_within(cat_name, 200.0)
                feat["building_density_proxy"] = total_poi_200m

                # ── Crowd count ───────────────────────────────────────────
                h3_cell = _h3_cell(lat, lon)
                feat["crowd_count"] = crowd_lookup.get((h3_cell, hour), 0)

                # ── Incident count (recency-decayed) ──────────────────────
                d_crime = CRIME_RADIUS_M / mpd_val
                nearby_incidents: list[dict] = []
                if inc_tree is not None:
                    nearby_idxs = inc_tree.query(
                        box(lon - d_crime, lat - d_crime, lon + d_crime, lat + d_crime)
                    )
                    nearby_incidents = [incidents_raw[i] for i in nearby_idxs]

                # Compute recency-decayed count
                now_utc = datetime.now(tz=timezone.utc)
                decayed_count = 0.0
                HALF_LIFE_DAYS = 60.0
                for inc in nearby_incidents:
                    occurred = inc.get("occurred_at")
                    if occurred is not None and hasattr(occurred, "timestamp"):
                        days_ago = max(0, (now_utc - occurred.astimezone(timezone.utc)).days)
                    else:
                        days_ago = 180
                    decay = math.exp(-0.693 * days_ago / HALF_LIFE_DAYS)
                    sev = max(1, min(5, inc.get("severity", 3)))
                    decayed_count += decay * (sev / 5.0)
                feat["incident_count_500m_decayed"] = round(decayed_count, 4)

                # ── Target: heuristic composite risk ──────────────────────
                seg_features = SegmentFeatures(
                    osm_id=osm_id,
                    lat=lat, lon=lon,
                    road_type=str(seg["road_type"]),
                    length_m=length_m,
                    lit_ratio=float(seg["lit_ratio"]),
                    lamp_count=int(seg["lamp_count"]),
                    shop_count_150m=feat["shop_count_150m"],
                    petrol_count_200m=feat["petrol_count_200m"],
                    any_metro_300m=feat["metro_count_300m"] > 0,
                    dist_main_m=feat["dist_main_road_m"],
                    crowd_count=feat["crowd_count"],
                    incidents=nearby_incidents,
                )
                scores = scorer.score(seg_features, time_ctx)
                base_risk = scores.composite
                if osm_id in feedback_by_segment:
                    fb_mean = float(np.mean(feedback_by_segment[osm_id]))
                    feat["risk_label"] = round(0.85 * base_risk + 0.15 * fb_mean, 4)
                else:
                    feat["risk_label"] = round(base_risk, 4)

                rows.append(feat)

    df = pd.DataFrame(rows)
    log.info(
        "Feature table built: %d rows, %d segments × %d time slots",
        len(df), len(segments), len(df) // max(len(segments), 1),
    )
    return df


def save_feature_table(df: pd.DataFrame, path: str | Path) -> None:
    """Save the feature DataFrame to parquet (or CSV fallback)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except (ImportError, ValueError):
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        log.info("Feature table saved to CSV fallback: %s (%d rows)", csv_path, len(df))
        return
    log.info("Feature table saved to %s (%d rows)", path, len(df))


def load_feature_table(path: str | Path) -> pd.DataFrame:
    """Load the feature DataFrame from parquet (or CSV fallback)."""
    path = Path(path)
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.read_parquet(path)
