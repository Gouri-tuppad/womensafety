"""
app/ml/shake_validator.py - Shake gesture validation using synthetic accelerometer classifier.

Accepts an accelerometer window [[x, y, z, t], ...] or [{"x":..,"y":..,"z":..,"t":..}, ...],
computes magnitudes, counts strong peaks above a threshold within ~2 seconds, and uses a
scikit-learn classifier trained on synthetic shake vs. walking vs. running data to return
a confidence score (0.0 to 1.0).

CRITICAL RULE:
If validation fails or no window is sent, NEVER block the SOS. Fall back to confidence 1.0.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

# Accelerometer threshold for strong shake peak (in m/s^2)
PEAK_THRESHOLD: float = 18.0
SAMPLING_RATE_HZ: float = 50.0  # assumed default if dt is missing


def _extract_window_arrays(
    accel_window: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse accel_window into (magnitudes, timestamps).

    Supports formats:
      - [[x, y, z, t], ...] or [[x, y, z], ...]
      - [{"x": x, "y": y, "z": z, "t": t}, ...]
    """
    if accel_window is None or len(accel_window) == 0:
        raise ValueError("Empty accelerometer window")

    records: list[tuple[float, float, float, float]] = []
    default_t = 0.0
    dt = 1.0 / SAMPLING_RATE_HZ

    for i, item in enumerate(accel_window):
        if isinstance(item, (list, tuple, np.ndarray)):
            if len(item) >= 4:
                x, y, z, t = float(item[0]), float(item[1]), float(item[2]), float(item[3])
            elif len(item) == 3:
                x, y, z = float(item[0]), float(item[1]), float(item[2])
                t = default_t
            else:
                continue
        elif isinstance(item, dict):
            x = float(item.get("x", 0.0))
            y = float(item.get("y", 0.0))
            z = float(item.get("z", 0.0))
            t = float(item.get("t", item.get("timestamp", default_t)))
        else:
            continue

        records.append((x, y, z, t))
        default_t += dt

    if len(records) < 3:
        raise ValueError("Insufficient accelerometer samples")

    arr = np.array(records, dtype=float)
    x, y, z, t = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    magnitudes = np.sqrt(x**2 + y**2 + z**2)
    return magnitudes, t


def extract_features(accel_window: Sequence[Any]) -> np.ndarray:
    """
    Extract feature vector for classifier:
      [peak_count, max_mag, mean_mag, std_mag, jerk_mean, peak_rate]
    """
    magnitudes, timestamps = _extract_window_arrays(accel_window)
    n = len(magnitudes)

    # Duration calculation
    duration = timestamps[-1] - timestamps[0]
    if duration <= 0.0 or duration > 100.0:
        duration = n / SAMPLING_RATE_HZ

    # Peak detection (local maxima above PEAK_THRESHOLD)
    peaks = 0
    for i in range(1, n - 1):
        if (
            magnitudes[i] > magnitudes[i - 1]
            and magnitudes[i] > magnitudes[i + 1]
            and magnitudes[i] >= PEAK_THRESHOLD
        ):
            peaks += 1

    max_mag = float(np.max(magnitudes))
    mean_mag = float(np.mean(magnitudes))
    std_mag = float(np.std(magnitudes))

    # Mean absolute difference (jerk proxy)
    jerk_mean = float(np.mean(np.abs(np.diff(magnitudes)))) if n > 1 else 0.0
    peak_rate = float(peaks / max(duration, 0.5))

    return np.array([peaks, max_mag, mean_mag, std_mag, jerk_mean, peak_rate], dtype=float)


def generate_synthetic_data(
    n_samples_per_class: int = 150,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic accelerometer windows for:
      - Class 1: Shake gesture (3 strong back-and-forth spikes, high energy > 20 m/s^2)
      - Class 0: Walking (cadence 1.8 Hz, mild oscillations 9-13 m/s^2)
      - Class 0: Running (cadence 3.0 Hz, medium oscillations 12-18 m/s^2)
    """
    rng = np.random.default_rng(random_state)
    features_list: list[np.ndarray] = []
    labels: list[int] = []

    # 1. Shake windows (Target = 1)
    for _ in range(n_samples_per_class):
        duration = rng.uniform(1.8, 2.5)
        num_points = int(duration * SAMPLING_RATE_HZ)
        t = np.linspace(0, duration, num_points)

        # Baseline gravity (z-axis or tilted)
        gx = rng.normal(0, 1.0)
        gy = rng.normal(0, 1.0)
        gz = math.sqrt(max(0.0, 9.81**2 - gx**2 - gy**2))

        # Three strong shakes: burst envelopes centered at t1, t2, t3
        shake_centers = np.sort(rng.uniform(0.3, duration - 0.3, size=3))
        shake_signal_x = np.zeros(num_points)
        shake_signal_y = np.zeros(num_points)

        for c in shake_centers:
            burst_width = rng.uniform(0.12, 0.22)
            burst = np.exp(-0.5 * ((t - c) / burst_width) ** 2)
            freq = rng.uniform(4.0, 7.0)
            amp = rng.uniform(18.0, 35.0)
            shake_signal_x += amp * burst * np.sin(2 * np.pi * freq * (t - c))
            shake_signal_y += amp * 0.5 * burst * np.cos(2 * np.pi * freq * (t - c))

        noise_x = rng.normal(0, 1.5, num_points)
        noise_y = rng.normal(0, 1.5, num_points)
        noise_z = rng.normal(0, 1.5, num_points)

        x = gx + shake_signal_x + noise_x
        y = gy + shake_signal_y + noise_y
        z = gz + noise_z

        window = np.column_stack([x, y, z, t])
        features_list.append(extract_features(window))
        labels.append(1)

    # 2. Walking windows (Target = 0)
    for _ in range(n_samples_per_class // 2):
        duration = rng.uniform(1.8, 2.5)
        num_points = int(duration * SAMPLING_RATE_HZ)
        t = np.linspace(0, duration, num_points)

        cadence = rng.uniform(1.6, 2.0)  # steps/s
        amp_z = rng.uniform(2.0, 3.8)
        amp_y = rng.uniform(1.0, 2.0)

        x = rng.normal(0, 0.5, num_points)
        y = amp_y * np.sin(2 * np.pi * (cadence / 2) * t) + rng.normal(0, 0.5, num_points)
        z = 9.81 + amp_z * np.sin(2 * np.pi * cadence * t) + rng.normal(0, 0.5, num_points)

        window = np.column_stack([x, y, z, t])
        features_list.append(extract_features(window))
        labels.append(0)

    # 3. Running windows (Target = 0)
    for _ in range(n_samples_per_class // 2):
        duration = rng.uniform(1.8, 2.5)
        num_points = int(duration * SAMPLING_RATE_HZ)
        t = np.linspace(0, duration, num_points)

        cadence = rng.uniform(2.6, 3.4)  # steps/s
        amp_z = rng.uniform(5.0, 8.5)
        amp_y = rng.uniform(2.5, 4.5)

        x = rng.normal(0, 1.0, num_points)
        y = amp_y * np.sin(2 * np.pi * (cadence / 2) * t) + rng.normal(0, 1.0, num_points)
        z = 9.81 + amp_z * np.sin(2 * np.pi * cadence * t) + rng.normal(0, 1.0, num_points)

        window = np.column_stack([x, y, z, t])
        features_list.append(extract_features(window))
        labels.append(0)

    return np.array(features_list), np.array(labels)


class ShakeClassifier:
    """Lightweight in-memory model trained on synthetic accelerometer profiles."""

    def __init__(self) -> None:
        self.model: RandomForestClassifier | None = None
        self._train()

    def _train(self) -> None:
        X, y = generate_synthetic_data(n_samples_per_class=160, random_state=42)
        clf = RandomForestClassifier(n_estimators=35, max_depth=5, random_state=42)
        clf.fit(X, y)
        self.model = clf
        logger.info("ShakeClassifier initialized and trained on synthetic gesture data.")

    def predict_confidence(self, accel_window: Sequence[Any]) -> float:
        """Return confidence score in [0.0, 1.0] that window represents a genuine shake."""
        features = extract_features(accel_window).reshape(1, -1)
        if self.model is None:
            return 1.0

        proba = self.model.predict_proba(features)[0]
        # proba[1] is probability of shake (class 1)
        shake_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])

        # If strong peaks >= 3 and max magnitude > 20 m/s^2, reinforce high confidence
        peaks = features[0][0]
        max_mag = features[0][1]
        if peaks >= 3 and max_mag >= 20.0:
            shake_prob = max(shake_prob, 0.85)

        return float(np.clip(shake_prob, 0.0, 1.0))


# Module-level singleton
_classifier: ShakeClassifier | None = None


def get_shake_classifier() -> ShakeClassifier:
    global _classifier
    if _classifier is None:
        _classifier = ShakeClassifier()
    return _classifier


def validate_shake(accel_window: Sequence[Any] | None) -> float:
    """
    Validate shake gesture from accelerometer window.

    CRITICAL SAFETY REQUIREMENT:
    If validation fails or no window is sent, NEVER block the SOS.
    Return 1.0 (or high confidence fallback) on error or missing data.
    """
    if accel_window is None:
        logger.info("No accelerometer window provided; defaulting SOS confidence to 1.0")
        return 1.0

    try:
        classifier = get_shake_classifier()
        confidence = classifier.predict_confidence(accel_window)
        logger.info("Shake validation completed with confidence=%.3f", confidence)
        return round(confidence, 3)
    except Exception as exc:
        logger.warning(
            "Shake validation failed with exception: %s. Never blocking SOS, defaulting to 1.0",
            exc,
        )
        return 1.0
