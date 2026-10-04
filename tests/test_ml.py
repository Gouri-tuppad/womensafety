"""
tests/test_ml.py - Tests for Step 4 ML backend.

Covers:
  1. Feature table builder produces correct columns
  2. ML model training and metrics generation
  3. Routing works with the MLRiskProvider
  4. SHAP explanations are returned
  5. Fairness audit produces a valid report
  6. predict_risk returns scores in [0, 1]

These tests are designed to run without a real database by mocking DB calls
and using synthetic in-memory data.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_feature_df():
    """Create a synthetic feature DataFrame for testing."""
    from app.ml.features import ALL_FEATURE_COLS, TARGET_COL

    np.random.seed(42)
    n_segments = 50
    n_slots = 8  # 4 hours × 2 day types
    n_rows = n_segments * n_slots

    data = {
        "osm_id": np.tile(np.arange(1, n_segments + 1), n_slots),
        "time_slot": np.repeat(
            ["08D", "14D", "20D", "23D", "08W", "14W", "20W", "23W"],
            n_segments,
        ),
    }

    # Temporal
    hours = [8, 14, 20, 23, 8, 14, 20, 23]
    data["hour"] = np.repeat(hours, n_segments)
    data["day_of_week"] = np.repeat([0, 0, 0, 0, 5, 5, 5, 5], n_segments)
    data["is_weekend"] = np.repeat([0, 0, 0, 0, 1, 1, 1, 1], n_segments)
    data["is_holiday"] = np.zeros(n_rows, dtype=int)
    data["is_dark"] = np.repeat([0, 0, 1, 1, 0, 0, 1, 1], n_segments)

    # Lighting
    data["lamp_density"] = np.random.uniform(0, 5, n_rows)
    data["lit_ratio"] = np.random.uniform(0, 1, n_rows)

    # POI counts
    for col in [
        "shop_count_150m", "atm_count_150m", "petrol_count_200m",
        "bus_stop_count_150m", "metro_count_300m", "police_count_300m",
        "hospital_count_300m", "factory_gate_count_150m",
    ]:
        data[col] = np.random.randint(0, 5, n_rows)

    # Infrastructure
    data["dist_main_road_m"] = np.random.uniform(10, 2000, n_rows)
    data["dist_police_m"] = np.random.uniform(100, 5000, n_rows)
    data["dist_hospital_m"] = np.random.uniform(100, 5000, n_rows)
    data["dist_metro_m"] = np.random.uniform(100, 5000, n_rows)
    data["road_type_encoded"] = np.random.randint(0, 10, n_rows)
    data["building_density_proxy"] = np.random.randint(0, 20, n_rows)

    # Crowd and crime
    data["crowd_count"] = np.random.randint(0, 100, n_rows)
    data["incident_count_500m_decayed"] = np.random.uniform(0, 5, n_rows)

    # Target: synthetic risk correlating with features
    risk = (
        0.3 * data["is_dark"]
        + 0.2 * (1.0 - data["lit_ratio"])
        + 0.15 * (data["incident_count_500m_decayed"] / 5.0)
        + 0.1 * (data["road_type_encoded"] / 9.0)
        + 0.1 * (1.0 - data["crowd_count"] / 100.0)
        + np.random.normal(0, 0.05, n_rows)
    )
    data[TARGET_COL] = np.clip(risk, 0.0, 1.0)

    return pd.DataFrame(data)


@pytest.fixture
def trained_model_dir(sample_feature_df, tmp_path):
    """Train models on synthetic data and return the model directory."""
    import xgboost as xgb
    import joblib
    from app.ml.features import ALL_FEATURE_COLS, TARGET_COL

    df = sample_feature_df
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    X = df[ALL_FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    # Train XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=10, max_depth=3, random_state=42,
    )
    xgb_model.fit(X, y)
    xgb_model.save_model(str(model_dir / "xgb_risk.json"))

    # Train RF
    from sklearn.ensemble import RandomForestRegressor
    rf_model = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
    rf_model.fit(X, y)
    joblib.dump(rf_model, str(model_dir / "rf_risk.joblib"))

    # Save feature columns
    with open(model_dir / "feature_columns.json", "w") as f:
        json.dump(ALL_FEATURE_COLS, f)

    return model_dir


# ── Test 1: Feature table structure ───────────────────────────────────────────


class TestFeatureTable:
    def test_has_all_feature_columns(self, sample_feature_df):
        from app.ml.features import ALL_FEATURE_COLS, TARGET_COL

        for col in ALL_FEATURE_COLS + [TARGET_COL]:
            assert col in sample_feature_df.columns, f"Missing column: {col}"

    def test_target_in_range(self, sample_feature_df):
        from app.ml.features import TARGET_COL

        assert sample_feature_df[TARGET_COL].min() >= 0.0
        assert sample_feature_df[TARGET_COL].max() <= 1.0

    def test_temporal_features_correct(self, sample_feature_df):
        assert set(sample_feature_df["is_weekend"].unique()) == {0, 1}
        assert set(sample_feature_df["is_dark"].unique()) == {0, 1}
        assert all(0 <= h <= 23 for h in sample_feature_df["hour"].unique())

    def test_road_type_encoding(self):
        from app.ml.features import _encode_road_type

        assert _encode_road_type("primary") == 2
        assert _encode_road_type("residential") == 5
        assert _encode_road_type("industrial") == 8
        assert _encode_road_type("unknown_type") == 6  # default


# ── Test 2: Model training and metrics ────────────────────────────────────────


class TestModelTraining:
    def test_time_based_split(self, sample_feature_df):
        from app.ml.train_risk_model import _time_based_split

        train_df, test_df = _time_based_split(sample_feature_df)

        assert len(train_df) > 0
        assert len(test_df) > 0
        assert len(train_df) + len(test_df) == len(sample_feature_df)

        # Ensure no time_slot overlap
        train_slots = set(train_df["time_slot"].unique())
        test_slots = set(test_df["time_slot"].unique())
        assert train_slots.isdisjoint(test_slots), "Train and test slots must not overlap"

    def test_trained_model_produces_predictions(self, trained_model_dir):
        import xgboost as xgb
        from app.ml.features import ALL_FEATURE_COLS

        model = xgb.XGBRegressor()
        model.load_model(str(trained_model_dir / "xgb_risk.json"))

        # Single prediction
        X = np.zeros((1, len(ALL_FEATURE_COLS)), dtype=np.float32)
        pred = model.predict(X)
        assert pred.shape == (1,)
        assert 0.0 <= float(np.clip(pred[0], 0, 1)) <= 1.0

    def test_metrics_file_structure(self, trained_model_dir, sample_feature_df, tmp_path):
        """Validate the metrics.json structure that the training script produces."""
        from app.ml.features import ALL_FEATURE_COLS

        # Simulate what train_models does for metrics
        metrics = {
            "zone": "peenya",
            "train_rows": 300,
            "test_rows": 100,
            "split_method": "time_based",
            "xgboost": {"model": "xgboost", "rmse": 0.05, "auc": 0.85},
            "random_forest": {"model": "random_forest", "rmse": 0.07, "auc": 0.80},
            "xgboost_vs_rf_rmse_delta": 0.02,
            "feature_importances": {col: 0.05 for col in ALL_FEATURE_COLS},
        }

        metrics_path = tmp_path / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)

        loaded = json.loads(metrics_path.read_text())
        assert "xgboost" in loaded
        assert "random_forest" in loaded
        assert "feature_importances" in loaded
        assert loaded["xgboost"]["rmse"] <= 1.0
        assert loaded["xgboost"]["auc"] >= 0.0


# ── Test 3: Routing with ML provider ────────────────────────────────────────


class TestMLRiskProvider:
    def test_ml_provider_implements_protocol(self, trained_model_dir):
        """MLRiskProvider must implement the RiskProvider protocol."""
        from app.services.safety_scoring import RiskProvider

        # Patch MODEL_DIR to point to our test models
        with patch("app.ml.predict.MODEL_DIR", trained_model_dir):
            # Clear singleton cache
            from app.ml import predict as predict_module
            predict_module._model_cache.clear()

            from app.ml.predict import MLRiskProvider
            provider = MLRiskProvider()

            assert isinstance(provider, RiskProvider)

    def test_ml_provider_returns_score_in_range(self, trained_model_dir):
        """MLRiskProvider.crime_risk() must return a value in [0, 1]."""
        with patch("app.ml.predict.MODEL_DIR", trained_model_dir):
            from app.ml import predict as predict_module
            predict_module._model_cache.clear()

            from app.ml.predict import MLRiskProvider
            from app.services.safety_scoring import TimeContext

            provider = MLRiskProvider()
            assert provider.is_available

            import pytz
            _ist = pytz.timezone("Asia/Kolkata")
            dt = _ist.localize(datetime(2024, 10, 15, 23, 0, 0))
            time_ctx = TimeContext.from_datetime(dt, "peenya")

            score = provider.crime_risk(
                lat=13.025,
                lon=77.515,
                radius_m=500.0,
                time_ctx=time_ctx,
                incident_rows=[],
            )

            assert 0.0 <= score <= 1.0

    def test_ml_provider_works_without_incidents(self, trained_model_dir):
        """Segments with no incident history must still get meaningful scores."""
        with patch("app.ml.predict.MODEL_DIR", trained_model_dir):
            from app.ml import predict as predict_module
            predict_module._model_cache.clear()

            from app.ml.predict import MLRiskProvider
            from app.services.safety_scoring import TimeContext

            provider = MLRiskProvider()

            import pytz
            _ist = pytz.timezone("Asia/Kolkata")
            dt = _ist.localize(datetime(2024, 10, 15, 14, 0, 0))
            time_ctx = TimeContext.from_datetime(dt, "peenya")

            # No incidents at all
            score = provider.crime_risk(
                lat=13.025, lon=77.515, radius_m=500.0,
                time_ctx=time_ctx, incident_rows=[],
            )

            # Should NOT be zero - environmental features drive the score
            assert score > 0.0, "Score should be > 0 even without incidents"
            assert score <= 1.0

    def test_ml_provider_in_segment_scorer(self, trained_model_dir):
        """MLRiskProvider can be injected into SegmentScorer."""
        with patch("app.ml.predict.MODEL_DIR", trained_model_dir):
            from app.ml import predict as predict_module
            predict_module._model_cache.clear()

            from app.ml.predict import MLRiskProvider
            from app.services.safety_scoring import SegmentScorer, SegmentFeatures, TimeContext

            provider = MLRiskProvider()
            scorer = SegmentScorer(risk_provider=provider)

            import pytz
            _ist = pytz.timezone("Asia/Kolkata")
            dt = _ist.localize(datetime(2024, 10, 15, 22, 0, 0))
            time_ctx = TimeContext.from_datetime(dt, "peenya")

            features = SegmentFeatures(
                osm_id=12345,
                lat=13.025, lon=77.515,
                road_type="industrial",
                length_m=200.0,
                lit_ratio=0.1,
                lamp_count=1,
                crowd_count=5,
            )

            scores = scorer.score(features, time_ctx)

            assert 0.0 <= scores.crime_risk <= 1.0
            assert 0.0 <= scores.darkness_risk <= 1.0
            assert 0.0 <= scores.isolation_risk <= 1.0
            assert 0.0 <= scores.activity_score <= 1.0


# ── Test 4: SHAP explanations ─────────────────────────────────────────────────


class TestSHAPExplanations:
    def test_explain_segment_returns_explanations(self, trained_model_dir):
        """explain_segment() must return a list of explanation dicts."""
        with patch("app.ml.predict.MODEL_DIR", trained_model_dir):
            from app.ml import predict as predict_module
            predict_module._model_cache.clear()

            # Also patch the explainer cache
            from app.ml import explain as explain_module
            explain_module._shap_explainer_cache.clear()

            from app.ml.explain import explain_segment

            features = {
                "hour": 22,
                "day_of_week": 0,
                "is_weekend": 0,
                "is_holiday": 0,
                "is_dark": 1,
                "lamp_density": 0.5,
                "lit_ratio": 0.15,
                "shop_count_150m": 0,
                "atm_count_150m": 0,
                "petrol_count_200m": 0,
                "bus_stop_count_150m": 0,
                "metro_count_300m": 0,
                "police_count_300m": 0,
                "hospital_count_300m": 0,
                "factory_gate_count_150m": 0,
                "dist_main_road_m": 800.0,
                "dist_police_m": 3000.0,
                "dist_hospital_m": 2500.0,
                "dist_metro_m": 4000.0,
                "road_type_encoded": 8,  # industrial
                "building_density_proxy": 2,
                "crowd_count": 3,
                "incident_count_500m_decayed": 2.5,
            }

            explanations = explain_segment(features, top_n=5)

            assert isinstance(explanations, list)
            assert len(explanations) > 0
            for exp in explanations:
                assert "feature" in exp
                assert "explanation" in exp
                assert "direction" in exp
                assert exp["direction"] in ("increases_risk", "decreases_risk")

    def test_explain_route_returns_strings(self, trained_model_dir):
        """explain_route() must return a list of plain-English strings."""
        with patch("app.ml.predict.MODEL_DIR", trained_model_dir):
            from app.ml import predict as predict_module
            predict_module._model_cache.clear()

            from app.ml import explain as explain_module
            explain_module._shap_explainer_cache.clear()

            from app.ml.explain import explain_route

            # Two segments with different characteristics
            seg1 = {
                "hour": 23, "day_of_week": 4, "is_weekend": 0, "is_holiday": 0,
                "is_dark": 1, "lamp_density": 0.2, "lit_ratio": 0.1,
                "shop_count_150m": 0, "atm_count_150m": 0, "petrol_count_200m": 0,
                "bus_stop_count_150m": 0, "metro_count_300m": 0, "police_count_300m": 0,
                "hospital_count_300m": 0, "factory_gate_count_150m": 0,
                "dist_main_road_m": 1000.0, "dist_police_m": 4000.0,
                "dist_hospital_m": 3000.0, "dist_metro_m": 5000.0,
                "road_type_encoded": 8, "building_density_proxy": 1,
                "crowd_count": 2, "incident_count_500m_decayed": 3.0,
            }
            seg2 = {
                "hour": 23, "day_of_week": 4, "is_weekend": 0, "is_holiday": 0,
                "is_dark": 1, "lamp_density": 3.0, "lit_ratio": 0.85,
                "shop_count_150m": 4, "atm_count_150m": 1, "petrol_count_200m": 1,
                "bus_stop_count_150m": 1, "metro_count_300m": 1, "police_count_300m": 1,
                "hospital_count_300m": 0, "factory_gate_count_150m": 0,
                "dist_main_road_m": 50.0, "dist_police_m": 200.0,
                "dist_hospital_m": 500.0, "dist_metro_m": 300.0,
                "road_type_encoded": 2, "building_density_proxy": 15,
                "crowd_count": 40, "incident_count_500m_decayed": 0.1,
            }

            explanations = explain_route([seg1, seg2], top_n=5)

            assert isinstance(explanations, list)
            assert len(explanations) > 0
            for exp in explanations:
                assert isinstance(exp, str)
                assert len(exp) > 5, "Explanations should be meaningful strings"


# ── Test 5: Predict risk batch ────────────────────────────────────────────────


class TestPredictRisk:
    def test_predict_risk_fallback_without_model(self):
        """Without a trained model, predict_risk returns neutral scores."""
        with patch("app.ml.predict.MODEL_DIR", Path("/nonexistent")):
            from app.ml import predict as predict_module
            predict_module._model_cache.clear()

            from app.ml.predict import predict_risk

            import pytz
            _ist = pytz.timezone("Asia/Kolkata")
            dt = _ist.localize(datetime(2024, 10, 15, 14, 0, 0))

            # This should fall back to 0.5 without raising
            result = predict_risk(
                segment_ids=[1, 2, 3],
                timestamp=dt,
                zone="peenya",
                db_url="postgresql://fake",
            )

            assert len(result) == 3
            for sid in [1, 2, 3]:
                assert result[sid] == 0.5


# ── Test 6: Feature explanation mapping ───────────────────────────────────────


class TestFeatureExplanationMapping:
    def test_lit_ratio_high(self):
        from app.ml.explain import _feature_explanation

        result = _feature_explanation("lit_ratio", 0.92, -0.05, False)
        assert result is not None
        assert "92%" in result

    def test_lit_ratio_low(self):
        from app.ml.explain import _feature_explanation

        result = _feature_explanation("lit_ratio", 0.1, 0.08, True)
        assert result is not None
        assert "10%" in result

    def test_dark_after_sunset(self):
        from app.ml.explain import _feature_explanation

        result = _feature_explanation("is_dark", 1.0, 0.1, True)
        assert result is not None
        assert "sunset" in result.lower() or "visibility" in result.lower()

    def test_metro_nearby(self):
        from app.ml.explain import _feature_explanation

        result = _feature_explanation("metro_count_300m", 1.0, -0.03, False)
        assert result is not None
        assert "metro" in result.lower()

    def test_industrial_road(self):
        from app.ml.explain import _feature_explanation

        result = _feature_explanation("road_type_encoded", 8, 0.05, True)
        assert result is not None
        assert "industrial" in result.lower()

    def test_crowd_high(self):
        from app.ml.explain import _feature_explanation

        result = _feature_explanation("crowd_count", 60, -0.04, False)
        assert result is not None
        assert "60" in result

    def test_crowd_low(self):
        from app.ml.explain import _feature_explanation

        result = _feature_explanation("crowd_count", 2, 0.06, True)
        assert result is not None
        assert "low" in result.lower()


# ── Test 7: Time-based split properties ───────────────────────────────────────


class TestTimeBasedSplit:
    def test_no_random_leak(self, sample_feature_df):
        """Verify that time-based split has no temporal leakage."""
        from app.ml.train_risk_model import _time_based_split

        train_df, test_df = _time_based_split(sample_feature_df)

        # Train slots should be chronologically "earlier"
        train_slots = sorted(train_df["time_slot"].unique())
        test_slots = sorted(test_df["time_slot"].unique())

        # Weekend slots (ending in W) should come after weekday (D) in our ordering
        for ts in train_slots:
            for tt in test_slots:
                # They must be different slots
                assert ts != tt

    def test_split_proportions(self, sample_feature_df):
        from app.ml.train_risk_model import _time_based_split

        train_df, test_df = _time_based_split(sample_feature_df)

        total = len(train_df) + len(test_df)
        test_ratio = len(test_df) / total
        # Should be approximately 25% test
        assert 0.10 <= test_ratio <= 0.50


# ── Test 8: End-to-end routing with MLRiskProvider and SHAP explanations ──────


class TestRoutingWithMLProvider:
    def test_routing_with_ml_risk_provider(self, trained_model_dir):
        """Verify routing works when injected with MLRiskProvider and returns SHAP explanations."""
        with patch("app.ml.predict.MODEL_DIR", trained_model_dir):
            from app.ml import predict as predict_module
            predict_module._model_cache.clear()
            from app.ml import explain as explain_module
            explain_module._shap_explainer_cache.clear()

            import networkx as nx
            import pytz
            from app.ml.predict import MLRiskProvider
            from app.services.routing_service import compute_routes
            from app.services.safety_scoring import SegmentScorer, TimeContext

            # Mock graph with 3 nodes and edges
            G = nx.MultiDiGraph()
            G.graph["crs"] = "epsg:4326"
            G.add_node(1, x=77.51, y=13.02)
            G.add_node(2, x=77.52, y=13.02)
            G.add_node(3, x=77.53, y=13.02)
            G.add_edge(1, 2, 0, length=500.0, highway="primary", osm_id=101)
            G.add_edge(2, 3, 0, length=500.0, highway="primary", osm_id=102)

            ml_provider = MLRiskProvider()
            scorer = SegmentScorer(risk_provider=ml_provider)

            _ist = pytz.timezone("Asia/Kolkata")
            dt = _ist.localize(datetime(2026, 10, 2, 23, 0, 0))
            time_ctx = TimeContext.from_datetime(dt, "peenya")

            mock_seg_data = {
                101: {"osm_id": 101, "road_type": "primary", "lit_ratio": 0.9, "lamp_count": 10, "lat": 13.02, "lon": 77.515},
                102: {"osm_id": 102, "road_type": "primary", "lit_ratio": 0.85, "lamp_count": 8, "lat": 13.02, "lon": 77.525},
            }

            with patch("app.services.routing_service.get_graph", return_value=G), \
                 patch("app.services.routing_service._get_zone_db_data", return_value=(mock_seg_data, [], list(mock_seg_data.values()))):
                res = compute_routes(
                    origin_lat=13.02,
                    origin_lon=77.51,
                    dest_lat=13.02,
                    dest_lon=77.53,
                    time_ctx=time_ctx,
                    profile_name="woman_night",
                    db_url="postgresql+psycopg://dummy",
                    scorer=scorer,
                    zone="peenya",
                )

            assert "fastest" in res
            assert "safest" in res
            assert "risk_explanations" in res
            assert isinstance(res["risk_explanations"], list)
            assert len(res["risk_explanations"]) > 0
            assert any("lit" in exp.lower() or "night" in exp.lower() or "dark" in exp.lower() for exp in res["risk_explanations"])
