"""
backfill_openmeteo.py

REPLACES generate_synthetic_history.py with REAL historical data.

Why this is better than synthetic data:
  Open-Meteo provides two completely free APIs, no signup, no API key,
  that between them give genuine historical hourly air quality AND
  weather data for any coordinates:

    - Air Quality API: https://open-meteo.com/en/docs/air-quality-api
      (pm2.5, pm10, CO, NO2, SO2, O3, US AQI -- CAMS global reanalysis,
       available from August 2022 onwards)

    - Historical Weather Archive API: https://open-meteo.com/en/docs/historical-weather-api
      (temperature, humidity, pressure, wind -- ERA5 reanalysis, available
       for decades)

  This is the exact same underlying data source the Zenodo "Air Quality
  Dataset for Karachi" dataset was built from -- so calling the API
  directly gets you the same real data, and you can pick your own date
  range instead of being limited to whatever snapshot was uploaded.

What this script does:
  1. Calls both APIs for Karachi's coordinates over a date range you choose.
  2. Merges pollutant + weather data into one hourly table.
  3. Computes the same feature schema as feature_pipeline.py (time-based
     features + aqi_change_rate), so training_pipeline.py and app.py work
     unchanged.
  4. Writes the result to data/features.csv, replacing the synthetic file.

Usage:
    python backfill_openmeteo.py --start 2024-01-01 --end 2026-09-01

No API key needed. Just internet access.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from feature_pipeline import COLUMNS, DATA_DIR, FEATURES_CSV

# Karachi coordinates
LATITUDE = 24.8607
LONGITUDE = 67.0011

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_air_quality(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches hourly pollutant data + US AQI for the given date range."""
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "auto",
    }
    response = requests.get(AIR_QUALITY_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()["hourly"]
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    return df


def fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches hourly temperature/humidity/pressure/wind for the given date range."""
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "auto",
    }
    response = requests.get(WEATHER_ARCHIVE_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()["hourly"]
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    return df


def build_feature_rows(aq_df: pd.DataFrame, wx_df: pd.DataFrame) -> pd.DataFrame:
    """Merges the two sources and computes the same schema feature_pipeline.py uses."""
    merged = pd.merge(aq_df, wx_df, on="time", how="inner").sort_values("time").reset_index(drop=True)

    rows = []
    prev_aqi = None
    for _, r in merged.iterrows():
        dt = r["time"]
        aqi = r["us_aqi"]
        aqi_change_rate = (aqi - prev_aqi) if (prev_aqi is not None and pd.notna(aqi) and pd.notna(prev_aqi)) else 0.0

        rows.append({
            "timestamp": dt.isoformat(),
            "hour": dt.hour,
            "day": dt.day,
            "month": dt.month,
            "day_of_week": dt.weekday(),
            "aqi": aqi,
            "pm25": r["pm2_5"],
            "pm10": r["pm10"],
            "o3": r["ozone"],
            "no2": r["nitrogen_dioxide"],
            "so2": r["sulphur_dioxide"],
            "co": r["carbon_monoxide"],
            "temperature": r["temperature_2m"],
            "humidity": r["relative_humidity_2m"],
            "pressure": r["surface_pressure"],
            "wind": r["wind_speed_10m"],
            "aqi_change_rate": aqi_change_rate,
        })
        if pd.notna(aqi):
            prev_aqi = aqi

    df = pd.DataFrame(rows, columns=COLUMNS)
    # drop rows where AQI itself is missing -- can't train/evaluate without a target
    df = df.dropna(subset=["aqi"]).reset_index(drop=True)
    return df


def run(start_date: str, end_date: str):
    print(f"Fetching real historical air quality for Karachi: {start_date} to {end_date}")
    aq_df = fetch_air_quality(start_date, end_date)
    print(f"  -> {len(aq_df)} hourly air-quality rows")

    print("Fetching matching historical weather...")
    wx_df = fetch_weather(start_date, end_date)
    print(f"  -> {len(wx_df)} hourly weather rows")

    features_df = build_feature_rows(aq_df, wx_df)
    print(f"Merged into {len(features_df)} complete feature rows")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(FEATURES_CSV, index=False)
    print(f"Wrote real historical data to {FEATURES_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-01-01", help="Start date YYYY-MM-DD (data available from 2022-08-01)")
    parser.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"), help="End date YYYY-MM-DD")
    args = parser.parse_args()
    run(args.start, args.end)
