"""
training_pipeline.py

Step 3 of the AQI Predictor project.

What this script does:
  1. Loads historical features from data/features.csv (the Feature Store).
  2. Builds THREE targets: AQI 24h, 48h, and 72h ahead (since we need a
     3-day forecast, not just "next hour").
  3. Trains and evaluates two models per horizon: Ridge Regression (simple,
     fast baseline) and Random Forest (usually stronger, handles
     non-linearity).
  4. Reports RMSE, MAE, R^2 for each model/horizon so you can show the
     comparison in your report.
  5. Saves the best model per horizon to models/ using joblib.
  6. Computes SHAP feature importance for the Random Forest models and
     saves a summary plot per horizon to models/shap_summary_*.png.

How to run:
    python training_pipeline.py
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # no display needed, we just save PNGs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).parent / "data"
FEATURES_CSV = DATA_DIR / "features.csv"
MODELS_DIR = Path(__file__).parent / "models"

FEATURE_COLUMNS = [
    "hour", "day", "month", "day_of_week",
    "pm25", "pm10", "o3", "no2", "so2", "co",
    "temperature", "humidity", "pressure", "wind",
    "aqi_change_rate", "aqi",
]

HORIZONS = {
    "24h": 24,   # rows are hourly, so 24 rows ahead = 24 hours
    "48h": 48,
    "72h": 72,
}


def load_data() -> pd.DataFrame:
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"{FEATURES_CSV} not found. Run generate_synthetic_history.py "
            "and/or feature_pipeline.py first to create training data."
        )
    df = pd.read_csv(FEATURES_CSV, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Adds target columns: AQI N rows (hours) in the future."""
    for name, shift in HORIZONS.items():
        df[f"target_{name}"] = df["aqi"].shift(-shift)
    return df


def evaluate(y_true, y_pred) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def train_one_horizon(df: pd.DataFrame, horizon_name: str) -> dict:
    """Trains Ridge + Random Forest for one horizon, returns results + best model."""
    target_col = f"target_{horizon_name}"
    data = df.dropna(subset=[target_col]).copy()

    X = data[FEATURE_COLUMNS]
    y = data[target_col]

    # time-ordered split (no shuffling) -- this is a time series, so we
    # test on the most recent slice, not a random slice
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    results = {}
    models = {}

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    results["Ridge"] = evaluate(y_test, ridge.predict(X_test))
    models["Ridge"] = ridge

    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    results["RandomForest"] = evaluate(y_test, rf.predict(X_test))
    models["RandomForest"] = rf

    # pick the model with the lower RMSE
    best_name = min(results, key=lambda k: results[k]["RMSE"])
    best_model = models[best_name]

    return {
        "horizon": horizon_name,
        "results": results,
        "best_model_name": best_name,
        "best_model": best_model,
        "X_test": X_test,
        "rf_model": models["RandomForest"],  # kept for SHAP regardless of which won
    }


def save_shap_summary(rf_model, X_test: pd.DataFrame, horizon_name: str) -> Path:
    """Saves a SHAP summary bar plot showing feature importance for this horizon."""
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title(f"Feature importance (SHAP) — {horizon_name} forecast")
    out_path = MODELS_DIR / f"shap_summary_{horizon_name}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    return out_path


def run():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    df = build_targets(df)

    print(f"Loaded {len(df)} rows from {FEATURES_CSV}\n")

    all_results = {}
    for horizon_name in HORIZONS:
        outcome = train_one_horizon(df, horizon_name)
        all_results[horizon_name] = outcome["results"]

        # save the best model
        model_path = MODELS_DIR / f"model_{horizon_name}.joblib"
        joblib.dump(outcome["best_model"], model_path)

        # SHAP summary from the Random Forest (tree explainer requires a tree model)
        shap_path = save_shap_summary(outcome["rf_model"], outcome["X_test"], horizon_name)

        print(f"--- Horizon: {horizon_name} ---")
        for model_name, metrics in outcome["results"].items():
            marker = "  <- selected" if model_name == outcome["best_model_name"] else ""
            print(f"  {model_name:14s} RMSE={metrics['RMSE']:.2f}  MAE={metrics['MAE']:.2f}  R2={metrics['R2']:.3f}{marker}")
        print(f"  Saved model to {model_path}")
        print(f"  Saved SHAP plot to {shap_path}\n")

    # save a metrics report for use in your final report / dashboard
    report_path = MODELS_DIR / "metrics_report.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Full metrics report saved to {report_path}")

    # save the feature column list so app.py knows what order to feed the model
    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(FEATURE_COLUMNS, f)


if __name__ == "__main__":
    run()
