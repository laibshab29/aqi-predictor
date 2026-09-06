"""
feature_pipeline.py

Step 1 of the AQI Predictor project.

WHY OPEN-METEO INSTEAD OF AQICN:
  backfill_openmeteo.py already builds the training set from Open-Meteo.
  Using AQICN here for live data would mean training and serving on two
  different data sources/methodologies -- a classic "train/serve skew"
  bug that quietly hurts accuracy. Using Open-Meteo for both means:
    - No signup, no API key, nothing to configure as a GitHub secret.
    - The live row and the historical rows are computed identically.

What this script does:
  1. Calls Open-Meteo's Air Quality API + Weather Archive-compatible
     forecast to get the latest available hour for Karachi.
  2. Extracts the same fields backfill_openmeteo.py uses.
  3. Computes time-based features + aqi_change_rate vs. the previous
     saved row.
  4. Appends the resulting feature row to data/features.csv.

How to run:
    python feature_pipeline.py

No API key needed. Just internet access.
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

# ---- Configuration -------------------------------------------------------

LATITUDE = 24.8607   # Karachi
LONGITUDE = 67.0011

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"  # standard forecast endpoint also covers "now"

DATA_DIR = Path(__file__).parent / "data"
FEATURES_CSV = DATA_DIR / "features.csv"

COLUMNS = [
    "timestamp", "hour", "day", "month", "day_of_week",
    "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
    "temperature", "humidity", "pressure", "wind",
    "aqi_change_rate",
]


# ---- Step 1: fetch raw data ----------------------------------------------

def fetch_raw_data() -> dict:
    """Calls Open-Meteo's air quality + weather endpoints for the current hour."""
    aq_params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi",
        "past_hours": 1,
        "forecast_hours": 1,
        "timezone": "auto",
    }
    aq_resp = requests.get(AIR_QUALITY_URL, params=aq_params, timeout=15)
    aq_resp.raise_for_status()
    aq_data = aq_resp.json()["hourly"]

    wx_params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
        "past_hours": 1,
        "forecast_hours": 1,
        "timezone": "auto",
    }
    wx_resp = requests.get(WEATHER_URL, params=wx_params, timeout=15)
    wx_resp.raise_for_status()
    wx_data = wx_resp.json()["hourly"]

    # take the most recent hour available from each response
    return {
        "time": aq_data["time"][-1],
        "pm2_5": aq_data["pm2_5"][-1],
        "pm10": aq_data["pm10"][-1],
        "carbon_monoxide": aq_data["carbon_monoxide"][-1],
        "nitrogen_dioxide": aq_data["nitrogen_dioxide"][-1],
        "sulphur_dioxide": aq_data["sulphur_dioxide"][-1],
        "ozone": aq_data["ozone"][-1],
        "us_aqi": aq_data["us_aqi"][-1],
        "temperature_2m": wx_data["temperature_2m"][-1],
        "relative_humidity_2m": wx_data["relative_humidity_2m"][-1],
        "surface_pressure": wx_data["surface_pressure"][-1],
        "wind_speed_10m": wx_data["wind_speed_10m"][-1],
    }


# ---- Step 2: compute features from raw data -------------------------------

def _get_previous_aqi():
    if not FEATURES_CSV.exists():
        return None
    with open(FEATURES_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    try:
        return float(rows[-1]["aqi"])
    except (KeyError, ValueError):
        return None


def compute_features(raw: dict) -> dict:
    dt = datetime.fromisoformat(raw["time"])
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    current_aqi = raw["us_aqi"]
    previous_aqi = _get_previous_aqi()
    aqi_change_rate = (
        (current_aqi - previous_aqi) if (current_aqi is not None and previous_aqi is not None) else 0.0
    )

    return {
        "timestamp": dt.isoformat(),
        "hour": dt.hour,
        "day": dt.day,
        "month": dt.month,
        "day_of_week": dt.weekday(),
        "aqi": current_aqi,
        "pm25": raw["pm2_5"],
        "pm10": raw["pm10"],
        "o3": raw["ozone"],
        "no2": raw["nitrogen_dioxide"],
        "so2": raw["sulphur_dioxide"],
        "co": raw["carbon_monoxide"],
        "temperature": raw["temperature_2m"],
        "humidity": raw["relative_humidity_2m"],
        "pressure": raw["surface_pressure"],
        "wind": raw["wind_speed_10m"],
        "aqi_change_rate": aqi_change_rate,
    }


# ---- Step 3: store the feature row (CSV for now) --------------------------

def save_feature_row(row: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = FEATURES_CSV.exists()

    with open(FEATURES_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    # HOPSWORKS TODO: replace this function's body with a Hopsworks
    # feature-group .insert() call when you're ready to move off CSV.
    # Everything upstream stays identical.


def run() -> dict:
    raw = fetch_raw_data()
    row = compute_features(raw)
    save_feature_row(row)
    print(f"Saved feature row: {json.dumps(row, indent=2)}")
    return row


if __name__ == "__main__":
    run()
