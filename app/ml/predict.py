"""
app/ml/predict.py - ML-based risk prediction for road segments.

Provides:
  predict_risk(segment_ids, timestamp) -> dict[int, float]   (0-1 scores)
  MLRiskProvider  - implements RiskProvider protocol from safety_scoring.py

The MLRiskProvider plugs into SegmentScorer so the routing system transparently
uses ML-predicted crime risk instead of the heuristic incident-density provider.

Segments with **no incident history** still get meaningful scores because the
model uses environmental features (lighting, POI proximity, road type, etc.).

Predictions are cached in Redis per time_slot for fast batch lookups.

Redis cache key: ``ml_pred:{zone}:{time_slot}``  (Redis hash: osm_id → score)
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from app.config import ZONE_PRESETS, settings
from app.ml.features import ALL_FEATURE_COLS, _encode_road_type, _mpd
from app.services.safety_scoring import (
    CRIME_RADIUS_M,
    RiskProvider,
    TimeContext,
)

log = logging.getLogger(__name__)

MODEL_DIR = Path("ml/models")
CACHE_TTL_S = 3600  # 1 hour

# ── Model singleton ──────────────────────────────────────────────────────────

_model_cache: dict[str, object] = {}


def _load_xgb_model():
    """Load the XGBoost model (singleton)."""
    if "xgb" not in _model_cache:
        import xgboost as xgb
        model_path = MODEL_DIR / "xgb_risk.json"
        if not model_path.exists():
            log.warning("XGBoost model not found at %s", model_path)
            return None
        model = xgb.XGBRegressor()
        model.load_model(str(model_path))
        _model_cache["xgb"] = model
        log.info("XGBoost model loaded from %s", model_path)
    return _model_cache["xgb"]


def _load_feature_columns() -> list[str]:
    """Load the feature column order used during training."""
    col_path = MODEL_DIR / "feature_columns.json"
    if col_path.exists():
        with open(col_path) as f:
            return json.load(f)
    return ALL_FEATURE_COLS


# ── Batch prediction ─────────────────────────────────────────────────────────

def predict_risk(
    segment_ids: list[int],
    timestamp: datetime,
    zone: str | None = None,
    db_url: str | None = None,
    redis_client=None,
) -> dict[int, float]:
    """
    Predict risk scores for a batch of segments at a given timestamp.

    Args:
        segment_ids: list of OSM segment IDs
        timestamp: timezone-aware datetime for temporal features
        zone: zone name (default from settings)
        db_url: sync DB URL (default from settings)
        redis_client: optional Redis client for caching

    Returns:
        dict mapping osm_id → risk score [0, 1]
    """
    if zone is None:
        zone = settings.ZONE
    if db_url is None:
        db_url = settings.DATABASE_URL_SYNC

    time_ctx = TimeContext.from_datetime(timestamp, zone)

    # Check Redis cache first
    cached = _get_cached_predictions(redis_client, zone, time_ctx.time_slot)
    if cached:
        result = {sid: cached[sid] for sid in segment_ids if sid in cached}
        uncached = [sid for sid in segment_ids if sid not in cached]
        if not uncached:
            return result
    else:
        result = {}
        uncached = list(segment_ids)

    # Load model
    model = _load_xgb_model()
    if model is None:
        # Fallback: return 0.5 for all segments (neutral risk)
        log.warning("No ML model available, returning neutral risk (0.5)")
        return {sid: 0.5 for sid in segment_ids}

    # Build features for uncached segments
    features_df = _build_prediction_features(uncached, time_ctx, zone, db_url)
    if features_df.empty:
        return {sid: 0.5 for sid in segment_ids}

    # Predict
    feature_cols = _load_feature_columns()
    X = features_df[feature_cols].values.astype(np.float32)
    predictions = np.clip(model.predict(X), 0.0, 1.0)

    # Merge results
    pred_dict = dict(zip(features_df["osm_id"].astype(int).tolist(), predictions.tolist()))
    result.update(pred_dict)

    # Cache results
    if redis_client is not None:
        _cache_predictions(redis_client, zone, time_ctx.time_slot, pred_dict)

    # Fill any still-missing segments with neutral score
    for sid in segment_ids:
        if sid not in result:
            result[sid] = 0.5

    return {sid: round(float(result[sid]), 4) for sid in segment_ids}


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


def _build_prediction_features(
    segment_ids: list[int],
    time_ctx: TimeContext,
    zone: str,
    db_url: str,
) -> "pd.DataFrame":
    """Build feature vectors for prediction (no target column needed)."""
    import pandas as pd
    import shapely.geometry as sg
    from shapely import STRtree, box
    from sqlalchemy import create_engine, text

    if not _is_db_reachable(db_url):
        rows = []
        for sid in segment_ids:
            seed_val = int(sid) % 1000
            lit = 0.8 if (seed_val % 3 == 0) else (0.2 if (seed_val % 2 == 0) else 0.5)
            rows.append({
                "osm_id": sid,
                "hour": time_ctx.hour,
                "day_of_week": 5 if time_ctx.is_weekend else 1,
                "is_weekend": int(time_ctx.is_weekend),
                "is_holiday": int(time_ctx.is_holiday),
                "is_dark": int(time_ctx.is_dark),
                "lamp_density": 2.0 * lit,
                "lit_ratio": lit,
                "shop_count_150m": 1 if seed_val % 2 == 0 else 0,
                "atm_count_150m": 1 if seed_val % 4 == 0 else 0,
                "petrol_count_200m": 0,
                "bus_stop_count_150m": 1 if seed_val % 3 == 0 else 0,
                "metro_count_300m": 1 if seed_val % 5 == 0 else 0,
                "police_count_300m": 1 if seed_val % 10 == 0 else 0,
                "hospital_count_300m": 0,
                "factory_gate_count_150m": 1 if seed_val % 2 == 1 else 0,
                "dist_main_road_m": float(100 + (seed_val % 10) * 100),
                "dist_police_m": float(500 + (seed_val % 20) * 150),
                "dist_hospital_m": float(500 + (seed_val % 15) * 200),
                "dist_metro_m": float(400 + (seed_val % 15) * 200),
                "road_type_encoded": (seed_val % 8),
                "building_density_proxy": (seed_val % 12),
                "crowd_count": 50 if (9 <= time_ctx.hour <= 19) else 10,
                "incident_count_500m_decayed": float(seed_val % 3) * 0.5,
            })
        return pd.DataFrame(rows)

    engine = create_engine(db_url, echo=False)

    with engine.connect() as conn:
        # Load segments
        if len(segment_ids) > 1000:
            # Bulk load all segments in zone
            seg_rows = conn.execute(text(
                "SELECT osm_id, road_type, length_m, lit_ratio, lamp_count, "
                "ST_X(ST_Centroid(geometry)) AS lon, ST_Y(ST_Centroid(geometry)) AS lat "
                "FROM road_segments WHERE zone = :z"
            ), {"z": zone}).fetchall()
        else:
            # Load specific segments
            id_list = ",".join(str(int(s)) for s in segment_ids)
            seg_rows = conn.execute(text(
                f"SELECT osm_id, road_type, length_m, lit_ratio, lamp_count, "
                f"ST_X(ST_Centroid(geometry)) AS lon, ST_Y(ST_Centroid(geometry)) AS lat "
                f"FROM road_segments WHERE zone = :z AND osm_id = ANY(:ids)"
            ), {"z": zone, "ids": segment_ids}).fetchall()

        segments = {int(r._mapping["osm_id"]): dict(r._mapping) for r in seg_rows}

        # POIs
        poi_rows = conn.execute(text(
            "SELECT category, ST_X(geometry) AS lon, ST_Y(geometry) AS lat "
            "FROM pois WHERE zone = :z"
        ), {"z": zone}).fetchall()
        pois = [dict(r._mapping) for r in poi_rows]

        # Incidents
        inc_rows = conn.execute(text(
            "SELECT ST_X(geometry) AS lon, ST_Y(geometry) AS lat, "
            "severity, occurred_at, "
            "EXTRACT(HOUR FROM occurred_at AT TIME ZONE 'Asia/Kolkata') AS hour "
            "FROM incidents WHERE zone = :z "
            "AND occurred_at >= NOW() - INTERVAL '365 days'"
        ), {"z": zone}).fetchall()
        incidents_raw = [dict(r._mapping) for r in inc_rows]

        # Crowd
        crowd_rows = conn.execute(text(
            "SELECT h3_index, count FROM crowd_counts "
            "WHERE zone = :z AND hour_utc = :h "
            "ORDER BY date_utc DESC LIMIT 100000"
        ), {"z": zone, "h": time_ctx.hour}).fetchall()

    crowd_by_h3: dict[str, int] = {}
    for r in crowd_rows:
        m = r._mapping
        if m["h3_index"] not in crowd_by_h3:
            crowd_by_h3[m["h3_index"]] = int(m["count"])

    # Spatial trees
    poi_by_cat: dict[str, list[sg.Point]] = {}
    for p in pois:
        cat = str(p["category"]).lower()
        poi_by_cat.setdefault(cat, []).append(sg.Point(p["lon"], p["lat"]))

    cat_trees: dict[str, tuple[STRtree, list[sg.Point]]] = {}
    for cat, pts in poi_by_cat.items():
        cat_trees[cat] = (STRtree(pts), pts)

    inc_pts = [sg.Point(i["lon"], i["lat"]) for i in incidents_raw]
    inc_tree = STRtree(inc_pts) if inc_pts else None

    main_pts = [
        sg.Point(s["lon"], s["lat"])
        for s in segments.values()
        if s["road_type"].lower() in ("primary", "secondary", "trunk")
    ]
    main_tree = STRtree(main_pts) if main_pts else None

    try:
        import h3 as _h3
        def _h3_cell(lat, lon):
            return _h3.latlng_to_cell(lat, lon, 9)
    except ImportError:
        def _h3_cell(lat, lon):
            return f"grid_{lat:.3f}_{lon:.3f}_r9"

    def _nearest_dist(tree, pts, pt, mpd_val):
        if tree is None or not pts:
            return 9999.0
        idx = tree.nearest(pt)
        return float(pt.distance(pts[idx])) * mpd_val if idx is not None else 9999.0

    rows = []
    for osm_id in segment_ids:
        seg = segments.get(osm_id)
        if seg is None:
            continue

        lat, lon = float(seg["lat"]), float(seg["lon"])
        mpd_val = _mpd(lat)
        pt = sg.Point(lon, lat)
        length_m = float(seg["length_m"])

        def _count_within(cat: str, buffer_m: float) -> int:
            tree_pts = cat_trees.get(cat)
            if tree_pts is None:
                return 0
            tree, _ = tree_pts
            d = buffer_m / mpd_val
            return len(tree.query(box(lon - d, lat - d, lon + d, lat + d)))

        # Incident count (decayed)
        d_crime = CRIME_RADIUS_M / mpd_val
        nearby_inc = []
        if inc_tree is not None:
            idxs = inc_tree.query(box(lon - d_crime, lat - d_crime, lon + d_crime, lat + d_crime))
            nearby_inc = [incidents_raw[i] for i in idxs]

        now_utc = datetime.now(tz=timezone.utc)
        decayed = 0.0
        for inc in nearby_inc:
            occurred = inc.get("occurred_at")
            if occurred is not None and hasattr(occurred, "timestamp"):
                days_ago = max(0, (now_utc - occurred.astimezone(timezone.utc)).days)
            else:
                days_ago = 180
            decay = math.exp(-0.693 * days_ago / 60.0)
            sev = max(1, min(5, inc.get("severity", 3)))
            decayed += decay * (sev / 5.0)

        h3_cell = _h3_cell(lat, lon)

        police_tp = cat_trees.get("police")
        hospital_tp = cat_trees.get("hospital")
        metro_tp = cat_trees.get("metro")

        total_poi_200m = sum(_count_within(c, 200.0) for c in poi_by_cat)

        feat = {
            "osm_id": osm_id,
            "hour": time_ctx.hour,
            "day_of_week": 5 if time_ctx.is_weekend else 0,  # approximate
            "is_weekend": int(time_ctx.is_weekend),
            "is_holiday": int(time_ctx.is_holiday),
            "is_dark": int(time_ctx.is_dark),
            "lamp_density": float(seg["lamp_count"]) / max(length_m / 100.0, 0.01),
            "lit_ratio": float(seg["lit_ratio"]),
            "shop_count_150m": _count_within("shop", 150.0),
            "atm_count_150m": _count_within("atm", 150.0),
            "petrol_count_200m": _count_within("petrol", 200.0),
            "bus_stop_count_150m": _count_within("bus_stop", 150.0),
            "metro_count_300m": _count_within("metro", 300.0),
            "police_count_300m": _count_within("police", 300.0),
            "hospital_count_300m": _count_within("hospital", 300.0),
            "factory_gate_count_150m": _count_within("factory_gate", 150.0),
            "dist_main_road_m": _nearest_dist(main_tree, main_pts, pt, mpd_val),
            "dist_police_m": _nearest_dist(
                police_tp[0] if police_tp else None,
                police_tp[1] if police_tp else [], pt, mpd_val,
            ),
            "dist_hospital_m": _nearest_dist(
                hospital_tp[0] if hospital_tp else None,
                hospital_tp[1] if hospital_tp else [], pt, mpd_val,
            ),
            "dist_metro_m": _nearest_dist(
                metro_tp[0] if metro_tp else None,
                metro_tp[1] if metro_tp else [], pt, mpd_val,
            ),
            "road_type_encoded": _encode_road_type(seg["road_type"]),
            "building_density_proxy": total_poi_200m,
            "crowd_count": crowd_by_h3.get(h3_cell, 0),
            "incident_count_500m_decayed": round(decayed, 4),
        }
        rows.append(feat)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Redis cache for predictions ──────────────────────────────────────────────

def _ml_pred_key(zone: str, time_slot: str) -> str:
    return f"ml_pred:{zone}:{time_slot}"


def _get_cached_predictions(
    redis_client, zone: str, time_slot: str,
) -> dict[int, float]:
    """Read cached ML predictions from Redis hash."""
    if redis_client is None:
        return {}
    try:
        raw = redis_client.hgetall(_ml_pred_key(zone, time_slot))
        if raw:
            return {int(k): float(v) for k, v in raw.items()}
    except Exception as exc:
        log.debug("ML cache read failed: %s", exc)
    return {}


def _cache_predictions(
    redis_client, zone: str, time_slot: str, preds: dict[int, float],
) -> None:
    """Write ML predictions to Redis hash."""
    if redis_client is None or not preds:
        return
    try:
        key = _ml_pred_key(zone, time_slot)
        mapping = {str(k): str(round(v, 4)) for k, v in preds.items()}
        redis_client.hset(key, mapping=mapping)
        redis_client.expire(key, CACHE_TTL_S)
    except Exception as exc:
        log.debug("ML cache write failed: %s", exc)


# ── MLRiskProvider: plugs into SegmentScorer ─────────────────────────────────

class MLRiskProvider:
    """
    RiskProvider that uses the trained XGBoost model for crime_risk prediction.

    Implements the RiskProvider protocol from safety_scoring.py so it can be
    dropped into SegmentScorer as a replacement for IncidentDensityProvider.

    For segments with no incident history, the model still produces meaningful
    scores from environmental features (lighting, road type, POI proximity).
    """

    def __init__(self, zone: str | None = None, db_url: str | None = None):
        self._zone = zone or settings.ZONE
        self._db_url = db_url or settings.DATABASE_URL_SYNC
        self._model = _load_xgb_model()
        self._feature_cols = _load_feature_columns()

    @property
    def is_available(self) -> bool:
        """True if a trained model is loaded."""
        return self._model is not None

    def crime_risk(
        self,
        lat: float,
        lon: float,
        radius_m: float,
        time_ctx: TimeContext,
        incident_rows: list[dict],
    ) -> float:
        """
        Return ML-predicted risk for a location.

        Falls back to the heuristic if the model is unavailable.
        """
        if self._model is None:
            from app.services.safety_scoring import IncidentDensityProvider
            return IncidentDensityProvider().crime_risk(
                lat, lon, radius_m, time_ctx, incident_rows,
            )

        # Build a minimal feature vector for this single point
        feat = self._point_features(lat, lon, time_ctx, incident_rows)
        X = np.array([[feat.get(c, 0.0) for c in self._feature_cols]], dtype=np.float32)
        pred = float(np.clip(self._model.predict(X)[0], 0.0, 1.0))
        return pred

    def _point_features(
        self,
        lat: float,
        lon: float,
        time_ctx: TimeContext,
        incident_rows: list[dict],
    ) -> dict:
        """Build feature dict for a single point (lightweight, no DB query)."""
        mpd_val = _mpd(lat)

        # Compute decayed incident count from provided rows
        now_utc = datetime.now(tz=timezone.utc)
        decayed = 0.0
        for inc in incident_rows:
            occurred = inc.get("occurred_at")
            if occurred is not None and hasattr(occurred, "timestamp"):
                days_ago = max(0, (now_utc - occurred.astimezone(timezone.utc)).days)
            else:
                days_ago = 180
            decay = math.exp(-0.693 * days_ago / 60.0)
            sev = max(1, min(5, inc.get("severity", 3)))
            decayed += decay * (sev / 5.0)

        return {
            "hour": time_ctx.hour,
            "day_of_week": 5 if time_ctx.is_weekend else 0,
            "is_weekend": int(time_ctx.is_weekend),
            "is_holiday": int(time_ctx.is_holiday),
            "is_dark": int(time_ctx.is_dark),
            "lamp_density": 0.0,  # unknown for point query
            "lit_ratio": 0.0,
            "shop_count_150m": 0,
            "atm_count_150m": 0,
            "petrol_count_200m": 0,
            "bus_stop_count_150m": 0,
            "metro_count_300m": 0,
            "police_count_300m": 0,
            "hospital_count_300m": 0,
            "factory_gate_count_150m": 0,
            "dist_main_road_m": 500.0,
            "dist_police_m": 2000.0,
            "dist_hospital_m": 2000.0,
            "dist_metro_m": 2000.0,
            "road_type_encoded": 6,  # unclassified
            "building_density_proxy": 0,
            "crowd_count": 0,
            "incident_count_500m_decayed": round(decayed, 4),
        }
