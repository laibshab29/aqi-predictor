"""
app.py

Step 4 of the AQI Predictor project -- the web dashboard.

What this does:
  1. Loads the trained models (24h/48h/72h) and the feature columns list.
  2. Loads the most recent row from data/features.csv (the "current state").
  3. Feeds that row into each model to get a 3-day forecast.
  4. Displays a chart of current + forecasted AQI.
  5. Shows a colored alert banner if any forecasted AQI crosses hazardous
     thresholds (using standard EPA-style AQI bands).
  6. Shows the SHAP feature-importance plots generated during training.

How to run:
    streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
FEATURES_CSV = DATA_DIR / "features.csv"

# Standard AQI category bands (US EPA scale)
AQI_BANDS = [
    (0, 50, "Good", "#00e400"),
    (51, 100, "Moderate", "#ffff00"),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff7e00"),
    (151, 200, "Unhealthy", "#ff0000"),
    (201, 300, "Very Unhealthy", "#8f3f97"),
    (301, 500, "Hazardous", "#7e0023"),
]


def classify_aqi(value: float):
    for low, high, label, color in AQI_BANDS:
        if low <= value <= high:
            return label, color
    return "Hazardous", "#7e0023"


@st.cache_resource
def load_models():
    models = {}
    for horizon in ["24h", "48h", "72h"]:
        path = MODELS_DIR / f"model_{horizon}.joblib"
        if path.exists():
            models[horizon] = joblib.load(path)
    with open(MODELS_DIR / "feature_columns.json") as f:
        feature_columns = json.load(f)
    return models, feature_columns


def load_latest_row() -> pd.Series:
    df = pd.read_csv(FEATURES_CSV, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")
    return df.iloc[-1]


def main():
    st.set_page_config(page_title="Karachi AQI Forecast", page_icon="🌫️", layout="centered")
    st.title("🌫️ Karachi AQI Forecast — Next 3 Days")

    if not FEATURES_CSV.exists() or not MODELS_DIR.exists():
        st.error(
            "No data or models found yet. Run generate_synthetic_history.py, "
            "then training_pipeline.py, before launching this app."
        )
        return

    models, feature_columns = load_models()
    latest = load_latest_row()

    st.caption(f"Latest reading: {latest['timestamp']}")

    current_aqi = latest["aqi"]
    current_label, current_color = classify_aqi(current_aqi)
    st.metric("Current AQI", f"{current_aqi:.0f}", current_label)

    # Build the model input row (must match training feature order exactly)
    X_latest = pd.DataFrame([latest[feature_columns]])

    forecast = {"Now": current_aqi}
    for horizon, model in models.items():
        pred = model.predict(X_latest)[0]
        forecast[horizon] = pred

    # ---- Chart ----
    chart_df = pd.DataFrame({
        "Horizon": list(forecast.keys()),
        "AQI": list(forecast.values()),
    })
    st.bar_chart(chart_df.set_index("Horizon"))

    # ---- Alerts ----
    hazardous_horizons = [h for h, v in forecast.items() if h != "Now" and v >= 151]
    if hazardous_horizons:
        st.error(
            f"⚠️ ALERT: Unhealthy or worse AQI predicted at: {', '.join(hazardous_horizons)}. "
            "Consider limiting outdoor activity."
        )
    else:
        st.success("No hazardous AQI levels predicted in the next 3 days.")

    # ---- Forecast table ----
    st.subheader("Forecast detail")
    table_rows = []
    for h, v in forecast.items():
        label, _ = classify_aqi(v)
        table_rows.append({"Horizon": h, "Predicted AQI": round(v, 1), "Category": label})
    st.table(pd.DataFrame(table_rows))

    # ---- SHAP explanations ----
    st.subheader("Why the model predicts this — feature importance (SHAP)")
    tabs = st.tabs(["24h", "48h", "72h"])
    for tab, horizon in zip(tabs, ["24h", "48h", "72h"]):
        with tab:
            img_path = MODELS_DIR / f"shap_summary_{horizon}.png"
            if img_path.exists():
                st.image(str(img_path))
            else:
                st.info("Run training_pipeline.py to generate this plot.")


if __name__ == "__main__":
    main()
