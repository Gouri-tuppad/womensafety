"""
app/ml/anomaly.py - Trip anomaly detection:
1. Route deviation: distance from planned polyline (in metres) exceeding threshold.
2. Unusual stop: stationary longer than N minutes on a segment whose risk is above threshold.
3. Abnormal motion: Isolation Forest on accel/gyro window features (magnitude mean/std, jerk, peak count).
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest

from app.config import settings

log = logging.getLogger(__name__)


# ── Distance & Geometry Helpers ───────────────────────────────────────────────

def _mpd(lat: float) -> tuple[float, float]:
    """Return (metres_per_degree_lat, metres_per_degree_lon) at given latitude."""
    lat_rad = math.radians(lat)
    m_lat = 111132.954 - 559.822 * math.cos(2 * lat_rad) + 1.175 * math.cos(4 * lat_rad)
    m_lon = 111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3 * lat_rad)
    return m_lat, m_lon


def point_to_segment_distance_m(
    p_lat: float, p_lon: float,
    a_lat: float, a_lon: float,
    b_lat: float, b_lon: float,
) -> float:
    """Distance in metres from point P to line segment AB."""
    mid_lat = (p_lat + a_lat + b_lat) / 3.0
    m_lat, m_lon = _mpd(mid_lat)

    # Project to local metric Cartesian coordinates
    px, py = p_lon * m_lon, p_lat * m_lat
    ax, ay = a_lon * m_lon, a_lat * m_lat
    bx, by = b_lon * m_lon, b_lat * m_lat

    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-6:
        return math.hypot(px - ax, py - ay)

    # Parametric projection t in [0, 1]
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
    proj_x = ax + t * dx
    proj_y = ay + t * dy

    return math.hypot(px - proj_x, py - proj_y)


def extract_coordinates_from_geojson(geojson_obj: dict[str, Any] | None) -> list[tuple[float, float]]:
    """
    Extract list of (lat, lon) coordinates from GeoJSON LineString or FeatureCollection.
    GeoJSON coordinates are [lon, lat]. We convert to (lat, lon).
    """
    if not geojson_obj:
        return []

    coords: list[tuple[float, float]] = []
    geom_type = geojson_obj.get("type")

    if geom_type == "FeatureCollection":
        features = geojson_obj.get("features", [])
        for feat in features:
            geom = feat.get("geometry", {})
            if geom.get("type") == "LineString":
                for pt in geom.get("coordinates", []):
                    if len(pt) >= 2:
                        coords.append((float(pt[1]), float(pt[0])))
    elif geom_type == "Feature":
        geom = geojson_obj.get("geometry", {})
        if geom.get("type") == "LineString":
            for pt in geom.get("coordinates", []):
                if len(pt) >= 2:
                    coords.append((float(pt[1]), float(pt[0])))
    elif geom_type == "LineString":
        for pt in geojson_obj.get("coordinates", []):
            if len(pt) >= 2:
                coords.append((float(pt[1]), float(pt[0])))

    return coords


def detect_route_deviation(
    current_lat: float,
    current_lon: float,
    route_geojson: dict[str, Any] | None,
    threshold_meters: float | None = None,
) -> tuple[bool, float]:
    """
    Check if current location deviates beyond threshold_meters from planned route.
    Returns (is_deviated, distance_meters).
    """
    threshold = threshold_meters or settings.TRIP_DEVIATION_METERS
    coords = extract_coordinates_from_geojson(route_geojson)
    if not coords or len(coords) < 2:
        return False, 0.0

    min_dist = float("inf")
    for i in range(len(coords) - 1):
        a_lat, a_lon = coords[i]
        b_lat, b_lon = coords[i + 1]
        dist = point_to_segment_distance_m(current_lat, current_lon, a_lat, a_lon, b_lat, b_lon)
        if dist < min_dist:
            min_dist = dist

    is_deviated = min_dist > threshold
    return is_deviated, round(min_dist, 2)


def _normalize_dt(dt_val: Any) -> datetime:
    if isinstance(dt_val, str):
        dt = datetime.fromisoformat(dt_val)
    elif isinstance(dt_val, datetime):
        dt = dt_val
    else:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def detect_unusual_stop(
    recent_locations: list[dict[str, Any]],
    segment_risk: float,
    min_duration_minutes: float | None = None,
    stationary_radius_m: float = 30.0,
    risk_threshold: float = 0.45,
) -> tuple[bool, float]:
    """
    Detect if user has stayed stationary for >= min_duration_minutes on a high-risk segment.
    Walks backwards from the latest location to compute continuous stationary duration.
    Returns (is_unusual_stop, stationary_minutes).
    """
    min_mins = min_duration_minutes or settings.TRIP_UNUSUAL_STOP_MINUTES
    if not recent_locations or len(recent_locations) < 2:
        return False, 0.0

    if segment_risk < risk_threshold:
        return False, 0.0

    # Ensure chronological order
    locs = sorted(
        recent_locations,
        key=lambda x: _normalize_dt(x["recorded_at"]),
    )

    curr_lat = locs[-1]["lat"]
    curr_lon = locs[-1]["lon"]
    t_end = _normalize_dt(locs[-1]["recorded_at"])

    m_lat, m_lon = _mpd(curr_lat)

    # Walk backwards from the latest position
    stationary_start = t_end
    for loc in reversed(locs[:-1]):
        t_loc = _normalize_dt(loc["recorded_at"])
        d_lat = (loc["lat"] - curr_lat) * m_lat
        d_lon = (loc["lon"] - curr_lon) * m_lon
        dist = math.hypot(d_lat, d_lon)
        if dist <= stationary_radius_m:
            stationary_start = t_loc
        else:
            break

    stationary_duration_mins = max(0.0, (t_end - stationary_start).total_seconds() / 60.0)
    is_stop = stationary_duration_mins >= min_mins
    return is_stop, round(stationary_duration_mins, 2)


# ── Motion Anomaly Detection (Isolation Forest) ────────────────────────────────

FEATURE_NAMES = [
    "mag_mean",
    "mag_std",
    "mag_max",
    "mag_min",
    "jerk_mean",
    "jerk_max",
    "peak_count",
    "gyro_mean",
    "gyro_std",
]


def extract_motion_features(motion_window: list[list[float]]) -> np.ndarray:
    """
    Extract feature vector from accelerometer / gyroscope time series window.
    Supports records of shape:
      [ax, ay, az]
      [ax, ay, az, t]
      [ax, ay, az, gx, gy, gz]
      [ax, ay, az, gx, gy, gz, t]
    """
    arr = np.asarray(motion_window, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 3:
        # Fallback neutral feature vector (normal stationary/walking)
        return np.array([9.8, 1.2, 11.0, 8.5, 2.0, 5.0, 1.0, 0.1, 0.05], dtype=np.float32)

    accel = arr[:, :3]
    mags = np.linalg.norm(accel, axis=1)

    mag_mean = float(np.mean(mags))
    mag_std = float(np.std(mags))
    mag_max = float(np.max(mags))
    mag_min = float(np.min(mags))

    # Jerk (discrete difference between consecutive magnitudes)
    diffs = np.abs(np.diff(mags))
    jerk_mean = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
    jerk_max = float(np.max(diffs)) if len(diffs) > 0 else 0.0

    # Strong acceleration peaks (sudden force > 15 m/s^2)
    peak_count = float(np.sum(mags > 15.0))

    # Gyroscope features if present
    if arr.shape[1] >= 6:
        gyro = arr[:, 3:6]
        gyro_mags = np.linalg.norm(gyro, axis=1)
        gyro_mean = float(np.mean(gyro_mags))
        gyro_std = float(np.std(gyro_mags))
    else:
        gyro_mean = 0.0
        gyro_std = 0.0

    return np.array([
        mag_mean, mag_std, mag_max, mag_min,
        jerk_mean, jerk_max, peak_count,
        gyro_mean, gyro_std,
    ], dtype=np.float32)


def generate_synthetic_motion_training_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for Isolation Forest motion anomaly training:
    - Normal walking (700 samples): steady 1.5-2.0 Hz gait, moderate std, low jerk.
    - Abnormal motion (150 samples): running sprints (high cadence/std) & drops/struggles.
    """
    rng = np.random.default_rng(42)
    features_list = []
    labels_list = []  # 1 for normal, -1 for anomaly

    # 1. Normal walking
    for _ in range(700):
        n_pts = rng.integers(20, 50)
        t = np.linspace(0, 2.0, n_pts)
        step_freq = rng.uniform(1.4, 2.0)
        base_accel = 9.8 + rng.uniform(1.0, 2.0) * np.sin(2 * np.pi * step_freq * t)
        noise = rng.normal(0, 0.4, (n_pts, 3))
        ax = noise[:, 0]
        ay = base_accel + noise[:, 1]
        az = noise[:, 2]
        sample = np.column_stack([ax, ay, az])
        features_list.append(extract_motion_features(sample.tolist()))
        labels_list.append(1)

    # 2. Running / Panic sprint
    for _ in range(80):
        n_pts = rng.integers(25, 60)
        t = np.linspace(0, 2.0, n_pts)
        step_freq = rng.uniform(2.8, 4.2)
        base_accel = 9.8 + rng.uniform(4.0, 9.0) * np.sin(2 * np.pi * step_freq * t)
        noise = rng.normal(0, 2.0, (n_pts, 3))
        sample = np.column_stack([noise[:, 0], base_accel + noise[:, 1], noise[:, 2]])
        features_list.append(extract_motion_features(sample.tolist()))
        labels_list.append(-1)

    # 3. Phone drop / violent struggle / impact
    for _ in range(70):
        n_pts = rng.integers(20, 40)
        sample = rng.normal(0, 0.5, (n_pts, 3))
        sample[:, 1] += 9.8
        # Free fall dip to near 0g
        dip_idx = n_pts // 3
        sample[dip_idx:dip_idx+3, :] = rng.normal(0, 0.2, (3, 3))
        # Violent impact spike
        impact_idx = dip_idx + 4
        if impact_idx < n_pts:
            sample[impact_idx, :] = rng.uniform(30.0, 60.0, 3) * rng.choice([-1, 1], 3)
        features_list.append(extract_motion_features(sample.tolist()))
        labels_list.append(-1)

    return np.array(features_list, dtype=np.float32), np.array(labels_list, dtype=np.int32)


class MotionAnomalyDetector:
    """
    Isolation Forest anomaly detector for smartphone sensor windows.
    Detects abnormal motion patterns such as panic running, struggles, or sudden drops.
    """

    _instance: MotionAnomalyDetector | None = None

    def __init__(self) -> None:
        X, _ = generate_synthetic_motion_training_data()
        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.18,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X)
        log.info("MotionAnomalyDetector initialized and fitted with %d synthetic motion samples.", len(X))

    @classmethod
    def get_instance(cls) -> MotionAnomalyDetector:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def evaluate(self, motion_window: list[list[float]] | None) -> tuple[bool, float, str]:
        """
        Evaluate a motion window for anomalies.
        Returns:
            (is_anomaly: bool, anomaly_score: float, reason: str)
        """
        if not motion_window or len(motion_window) < 5:
            return False, 0.0, "insufficient_samples"

        try:
            feats = extract_motion_features(motion_window)
            pred = self.model.predict(feats.reshape(1, -1))[0]
            # score_samples returns opposite of anomaly score (lower is more anomalous)
            raw_score = float(self.model.score_samples(feats.reshape(1, -1))[0])
            norm_anomaly_score = max(0.0, min(1.0, 0.5 - raw_score))

            is_anomaly = bool(pred == -1)

            # Classify specific anomaly type
            mag_max = float(feats[2])
            jerk_max = float(feats[5])
            peak_count = float(feats[6])
            mag_std = float(feats[1])

            if mag_max > 25.0 or jerk_max > 80.0:
                reason = "phone_drop_or_impact"
            elif peak_count >= 5 or mag_std > 3.5:
                reason = "rapid_running_or_struggle"
            else:
                reason = "abnormal_motion_pattern"

            return is_anomaly, norm_anomaly_score, reason if is_anomaly else "normal"
        except Exception as exc:
            log.warning("Motion anomaly detection error: %s", exc)
            return False, 0.0, "evaluation_error"


def get_motion_detector() -> MotionAnomalyDetector:
    return MotionAnomalyDetector.get_instance()
