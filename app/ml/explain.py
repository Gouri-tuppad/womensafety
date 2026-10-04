"""
app/ml/explain.py - SHAP-based explanations for risk predictions.

Uses SHAP TreeExplainer on the trained XGBoost model to generate
human-readable explanations for why a route or segment is risky/safe.

Example output for a route segment:
  "Well lit: 92% of road lit"
  "Metro station within 300 m"
  "Very low foot traffic after 10 PM"
  "Near police station (280 m)"
  "Industrial road with long compound walls"

Explanations are aggregated per route: the top N contributing factors
across all segments are returned, with natural-language descriptions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from app.ml.features import ALL_FEATURE_COLS
from app.ml.predict import _load_xgb_model, _load_feature_columns

log = logging.getLogger(__name__)


# ── SHAP explainer singleton ─────────────────────────────────────────────────

_shap_explainer_cache: dict[str, object] = {}


def _get_shap_explainer():
    """Get or create a SHAP TreeExplainer for the XGBoost model."""
    if "explainer" not in _shap_explainer_cache:
        model = _load_xgb_model()
        if model is None:
            return None
        import shap
        explainer = shap.TreeExplainer(model)
        _shap_explainer_cache["explainer"] = explainer
        log.info("SHAP TreeExplainer created")
    return _shap_explainer_cache["explainer"]


# ── Feature → human-readable mapping ────────────────────────────────────────

def _feature_explanation(
    feature_name: str,
    feature_value: float,
    shap_value: float,
    is_positive: bool,
) -> str | None:
    """
    Convert a (feature, value, SHAP contribution) into a plain-English string.

    Returns None if the feature doesn't have a meaningful human description.
    ``is_positive`` means the feature increases risk (bad for safety).
    """
    v = feature_value

    # ── Lighting ──────────────────────────────────────────────────────────
    if feature_name == "lit_ratio":
        if v >= 0.8:
            return f"Well lit: {v*100:.0f}% of road lit"
        elif v >= 0.4:
            return f"Partially lit: {v*100:.0f}% of road lit"
        else:
            return f"Poorly lit: only {v*100:.0f}% of road lit"

    if feature_name == "lamp_density":
        if v >= 3.0:
            return "Dense street lighting"
        elif v >= 1.0:
            return "Moderate street lighting"
        elif v < 0.5:
            return "Very few street lamps"

    # ── Darkness / time ───────────────────────────────────────────────────
    if feature_name == "is_dark":
        if v > 0.5:
            return "After sunset: reduced visibility"
        else:
            return "Daytime: good natural visibility"

    if feature_name == "hour":
        if v >= 22 or v <= 4:
            return f"Late night ({int(v):02d}:00): very few people around"
        elif v >= 19:
            return f"Evening ({int(v):02d}:00): declining foot traffic"

    # ── POI proximity ─────────────────────────────────────────────────────
    if feature_name == "metro_count_300m":
        if v >= 1:
            return "Metro station within 300 m"
        elif is_positive:
            return "No metro station nearby"

    if feature_name == "police_count_300m":
        if v >= 1:
            return f"Police station within 300 m"
        elif is_positive:
            return "No police station nearby"

    if feature_name == "hospital_count_300m":
        if v >= 1:
            return "Hospital/clinic within 300 m"

    if feature_name == "shop_count_150m":
        if v >= 3:
            return f"{int(v)} shops/retail within 150 m (well-populated area)"
        elif v == 0 and is_positive:
            return "No shops nearby: isolated stretch"

    if feature_name == "petrol_count_200m":
        if v >= 1:
            return "24h petrol station nearby (always lit)"
        elif is_positive:
            return "No 24h petrol station nearby"

    if feature_name == "bus_stop_count_150m":
        if v >= 1:
            return f"Bus stop within 150 m"

    # ── Infrastructure ────────────────────────────────────────────────────
    if feature_name == "dist_main_road_m":
        if v < 100:
            return "On/near main road (well-connected)"
        elif v > 500:
            return f"Far from main road ({v:.0f} m): less connected"

    if feature_name == "dist_police_m":
        if v < 500:
            return f"Near police station ({v:.0f} m)"
        elif v > 2000 and is_positive:
            return f"Far from police station ({v:.0f} m)"

    if feature_name == "dist_hospital_m":
        if v < 500:
            return f"Near hospital ({v:.0f} m)"

    if feature_name == "dist_metro_m":
        if v < 500:
            return f"Near metro station ({v:.0f} m)"
        elif v > 2000 and is_positive:
            return "Far from metro/transit"

    if feature_name == "road_type_encoded":
        road_labels = {
            0: "Highway/motorway",
            1: "Trunk road",
            2: "Primary road (well-connected)",
            3: "Secondary road",
            4: "Tertiary road",
            5: "Residential street",
            6: "Unclassified road",
            7: "Service road (often isolated)",
            8: "Industrial road with long compound walls",
            9: "Footpath/track (isolated)",
        }
        label = road_labels.get(int(v), f"Road type {int(v)}")
        return label

    # ── Crowd / activity ──────────────────────────────────────────────────
    if feature_name == "crowd_count":
        if v >= 40:
            return f"High foot traffic ({int(v)} people nearby)"
        elif v >= 15:
            return f"Moderate foot traffic ({int(v)} people nearby)"
        elif v < 5:
            return "Very low foot traffic"

    # ── Crime history ─────────────────────────────────────────────────────
    if feature_name == "incident_count_500m_decayed":
        if v >= 3.0:
            return f"High incident history nearby (weighted: {v:.1f})"
        elif v >= 1.0:
            return f"Some past incidents nearby (weighted: {v:.1f})"
        elif v < 0.1:
            return "No recent incident history nearby"

    # ── Weekend / holiday ─────────────────────────────────────────────────
    if feature_name == "is_weekend":
        if v > 0.5:
            return "Weekend: different activity patterns"

    if feature_name == "is_holiday":
        if v > 0.5:
            return "Holiday: reduced regular commuter traffic"

    # ── Building density ──────────────────────────────────────────────────
    if feature_name == "building_density_proxy":
        if v >= 15:
            return "Dense urban area with many amenities"
        elif v < 3 and is_positive:
            return "Sparse area: few buildings and amenities"

    return None


# ── Segment-level explanation ────────────────────────────────────────────────

def explain_segment(
    feature_values: dict[str, float],
    top_n: int = 5,
) -> list[dict]:
    """
    Explain the risk prediction for a single segment.

    Args:
        feature_values: dict of feature_name → value
        top_n: number of top contributing factors to return

    Returns:
        List of dicts with keys: feature, value, shap_value, direction, explanation
    """
    explainer = _get_shap_explainer()
    if explainer is None:
        return []

    feature_cols = _load_feature_columns()
    X = np.array([[feature_values.get(c, 0.0) for c in feature_cols]], dtype=np.float32)

    shap_values = explainer.shap_values(X)[0]  # shape: (n_features,)

    # Pair features with SHAP values and sort by absolute contribution
    pairs = []
    for i, col in enumerate(feature_cols):
        sv = float(shap_values[i])
        fv = float(feature_values.get(col, 0.0))
        is_positive = sv > 0  # increases risk
        explanation = _feature_explanation(col, fv, sv, is_positive)
        if explanation is not None:
            pairs.append({
                "feature": col,
                "value": round(fv, 3),
                "shap_value": round(sv, 5),
                "direction": "increases_risk" if is_positive else "decreases_risk",
                "explanation": explanation,
            })

    # Sort by absolute SHAP value (most impactful first)
    pairs.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    return pairs[:top_n]


# ── Route-level explanation ──────────────────────────────────────────────────

def explain_route(
    segment_features: list[dict[str, float]],
    top_n: int = 5,
) -> list[str]:
    """
    Generate top contributing factors across an entire route.

    Aggregates SHAP values across all segments in the route, then picks
    the top_n most impactful factors with plain-English descriptions.

    Args:
        segment_features: list of feature dicts (one per route segment)
        top_n: number of top explanations to return

    Returns:
        List of plain-English strings like:
          "Well lit: 92% of road lit"
          "Metro station within 300 m"
    """
    explainer = _get_shap_explainer()
    if explainer is None or not segment_features:
        return []

    feature_cols = _load_feature_columns()

    # Build feature matrix
    X = np.array([
        [sf.get(c, 0.0) for c in feature_cols]
        for sf in segment_features
    ], dtype=np.float32)

    shap_values = explainer.shap_values(X)  # shape: (n_segments, n_features)

    # Aggregate: mean absolute SHAP value per feature
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    mean_shap = np.mean(shap_values, axis=0)
    mean_feat = np.mean(X, axis=0)

    # Build explanations from the most impactful features
    indexed = list(enumerate(mean_abs_shap))
    indexed.sort(key=lambda x: x[1], reverse=True)

    explanations: list[str] = []
    seen_categories = set()  # avoid redundant explanations

    for idx, abs_sv in indexed:
        if len(explanations) >= top_n:
            break
        col = feature_cols[idx]
        fv = float(mean_feat[idx])
        sv = float(mean_shap[idx])
        is_positive = sv > 0

        explanation = _feature_explanation(col, fv, sv, is_positive)
        if explanation is None:
            continue

        # Deduplicate similar categories
        category = col.split("_")[0]
        if category in seen_categories and len(explanations) >= 3:
            continue
        seen_categories.add(category)

        explanations.append(explanation)

    return explanations
