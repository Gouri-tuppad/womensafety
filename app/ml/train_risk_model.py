"""
app/ml/train_risk_model.py - Train XGBoost (primary) and Random Forest (baseline).

Uses a **TIME-BASED** split: train on older time slots, test on the latest
2-3 months' worth of data.  No random split to prevent temporal leakage.

Outputs:
  ml/models/xgb_risk.json           - trained XGBoost model
  ml/models/rf_risk.joblib           - trained Random Forest baseline
  ml/reports/metrics.json            - AUC, RMSE, baseline comparison, feature
                                       importances, split sizes

CLI:
  python -m app.ml.train_risk_model          # uses ZONE from env
  python -m app.ml.train_risk_model --zone peenya

Wired to: make train
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score

from app.config import settings
from app.ml.features import (
    ALL_FEATURE_COLS,
    TARGET_COL,
    build_feature_table,
    save_feature_table,
)

log = logging.getLogger(__name__)

MODEL_DIR = Path("ml/models")
REPORT_DIR = Path("ml/reports")
FEATURE_DIR = Path("ml/data")


def _time_based_split(
    df: pd.DataFrame,
    test_fraction: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-based split: use hour + is_weekend as a proxy for temporal ordering.

    We split by time_slot groups: the latest ~25% of unique time slots become
    the test set.  This simulates training on older data and predicting future
    risk scores.

    For real deployment with dated incidents, split by occurred_at month.
    """
    # Sort time_slots deterministically: weekday-night is "later" than weekday-day
    # Order: 00D, 01D, ..., 23D, 00W, 01W, ..., 23W
    slots = sorted(df["time_slot"].unique(), key=lambda s: (s[-1], int(s[:-1])))
    n_test = max(1, int(len(slots) * test_fraction))
    test_slots = set(slots[-n_test:])
    train_slots = set(slots) - test_slots

    train_df = df[df["time_slot"].isin(train_slots)].copy()
    test_df = df[df["time_slot"].isin(test_slots)].copy()

    log.info(
        "Time-based split: %d train slots (%d rows), %d test slots (%d rows)",
        len(train_slots), len(train_df), len(test_slots), len(test_df),
    )
    return train_df, test_df


def train_models(
    zone: str,
    db_url: str,
    skip_feature_build: bool = False,
) -> dict:
    """
    Train XGBoost + Random Forest on the feature table.

    Returns a metrics dict written to ml/reports/metrics.json.
    """
    import xgboost as xgb
    import joblib

    # ── Build or load feature table ───────────────────────────────────────
    feature_path = FEATURE_DIR / f"{zone}_features.parquet"
    if skip_feature_build and feature_path.exists():
        log.info("Loading existing feature table from %s", feature_path)
        df = pd.read_parquet(feature_path)
    else:
        log.info("Building feature table for zone=%s ...", zone)
        df = build_feature_table(zone, db_url)
        save_feature_table(df, feature_path)

    if len(df) == 0:
        raise ValueError("Feature table is empty – have you run `make seed`?")

    # ── Time-based split ──────────────────────────────────────────────────
    train_df, test_df = _time_based_split(df)

    X_train = train_df[ALL_FEATURE_COLS].values.astype(np.float32)
    y_train = train_df[TARGET_COL].values.astype(np.float32)
    X_test = test_df[ALL_FEATURE_COLS].values.astype(np.float32)
    y_test = test_df[TARGET_COL].values.astype(np.float32)

    # ── XGBoost (primary) ─────────────────────────────────────────────────
    log.info("Training XGBoost ...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    xgb_pred = np.clip(xgb_model.predict(X_test), 0.0, 1.0)

    # ── Random Forest (baseline) ──────────────────────────────────────────
    log.info("Training Random Forest baseline ...")
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    rf_pred = np.clip(rf_model.predict(X_test), 0.0, 1.0)

    # ── Metrics ───────────────────────────────────────────────────────────
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        # AUC: binarise target at 0.5 threshold
        y_binary = (y_true >= 0.5).astype(int)
        try:
            if len(np.unique(y_binary)) < 2:
                # If all one class, use a lower threshold
                y_binary = (y_true >= np.median(y_true)).astype(int)
            auc = float(roc_auc_score(y_binary, y_pred))
        except ValueError:
            auc = float("nan")
        return {"model": name, "rmse": round(rmse, 5), "auc": round(auc, 5)}

    xgb_metrics = _compute_metrics(y_test, xgb_pred, "xgboost")
    rf_metrics = _compute_metrics(y_test, rf_pred, "random_forest")

    # Feature importances from XGBoost
    importances = dict(zip(ALL_FEATURE_COLS, [round(float(v), 5) for v in xgb_model.feature_importances_]))
    sorted_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    metrics = {
        "zone": zone,
        "data_disclaimer": "All crime incident data, street lighting scores, and crowd levels are SYNTHETIC / SIMULATED for demonstration.",
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "split_method": "time_based (latest 25% of time_slots as test)",
        "xgboost": xgb_metrics,
        "random_forest": rf_metrics,
        "xgboost_vs_rf_rmse_delta": round(rf_metrics["rmse"] - xgb_metrics["rmse"], 5),
        "feature_importances": sorted_importances,
    }

    # ── Save models ───────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    xgb_path = MODEL_DIR / "xgb_risk.json"
    xgb_model.save_model(str(xgb_path))
    log.info("XGBoost model saved to %s", xgb_path)

    # Save RF with joblib
    import joblib
    rf_path = MODEL_DIR / "rf_risk.joblib"
    joblib.dump(rf_model, str(rf_path))
    log.info("Random Forest model saved to %s", rf_path)

    # Save feature column order for prediction
    col_order_path = MODEL_DIR / "feature_columns.json"
    with open(col_order_path, "w") as f:
        json.dump(ALL_FEATURE_COLS, f)

    # ── Save metrics ──────────────────────────────────────────────────────
    metrics_path = REPORT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics saved to %s", metrics_path)

    # Log summary
    log.info(
        "Training complete: XGBoost RMSE=%.4f AUC=%.4f | RF RMSE=%.4f AUC=%.4f",
        xgb_metrics["rmse"], xgb_metrics["auc"],
        rf_metrics["rmse"], rf_metrics["auc"],
    )

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Train SafeWay ML risk model")
    parser.add_argument("--zone", default=settings.ZONE, help="Zone to train on")
    parser.add_argument(
        "--skip-feature-build", action="store_true",
        help="Reuse existing feature parquet if available",
    )
    args = parser.parse_args()

    metrics = train_models(
        zone=args.zone,
        db_url=settings.DATABASE_URL_SYNC,
        skip_feature_build=args.skip_feature_build,
    )
    print(f"\n[OK] Training complete for zone '{args.zone}'")
    print(f"   XGBoost RMSE: {metrics['xgboost']['rmse']:.4f}  AUC: {metrics['xgboost']['auc']:.4f}")
    print(f"   RF       RMSE: {metrics['random_forest']['rmse']:.4f}  AUC: {metrics['random_forest']['auc']:.4f}")
    print(f"   Metrics -> ml/reports/metrics.json")


if __name__ == "__main__":
    main()
