"""
generate_synthetic_history.py

WHY THIS FILE EXISTS (be upfront about this in your report):
  Training a model needs weeks/months of history. A free-tier API only gives
  you the *current* reading each time you call it, so real history has to be
  accumulated over time. Since you don't have that time, this script creates
  a realistic SYNTHETIC historical dataset -- same schema as feature_pipeline.py
  produces -- so you can build and test the training pipeline TODAY.

  This is a legitimate bootstrapping technique (common in student/prototype
  projects) but it is NOT real data. Say so explicitly in your report:
  "Historical training data for the initial model was synthetically generated
  to simulate realistic diurnal/seasonal AQI patterns, while the live feature
  pipeline (feature_pipeline.py, run hourly via GitHub Actions) collects real
  data going forward. As real data accumulates, the model should be retrained
  on it."

  The patterns simulated are based on well-documented real-world behavior:
    - AQI/PM2.5 tends to be worse in winter months (less rain, temperature
      inversions trap pollutants)
    - AQI tends to spike during traffic rush hours (7-10am, 5-8pm)
    - Temperature and AQI are loosely correlated with time of day/season
    - Random day-to-day noise, because real air quality is noisy

Usage:
    python generate_synthetic_history.py --days 60
"""

import argparse
import csv
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from feature_pipeline import COLUMNS, DATA_DIR, FEATURES_CSV


def simulate_row(dt: datetime, previous_aqi: float | None) -> dict:
    """Generates one plausible hourly feature row for a given datetime."""
    hour = dt.hour
    month = dt.month

    # Seasonal baseline: worse AQI in winter (Nov-Feb), better in monsoon (Jul-Sep)
    winter_months = {11, 12, 1, 2}
    monsoon_months = {7, 8, 9}
    if month in winter_months:
        base_aqi = 160
    elif month in monsoon_months:
        base_aqi = 70
    else:
        base_aqi = 110

    # Diurnal pattern: rush-hour bumps around 8am and 6pm
    rush_hour_bump = 25 * (math.exp(-((hour - 8) ** 2) / 8) + math.exp(-((hour - 18) ** 2) / 8))
    # Overnight dip
    night_dip = -20 if 1 <= hour <= 4 else 0

    noise = random.gauss(0, 12)
    aqi = max(15, base_aqi + rush_hour_bump + night_dip + noise)

    # Correlated pollutant sub-indices (roughly proportional to AQI, plus noise)
    pm25 = max(5, aqi * 0.9 + random.gauss(0, 8))
    pm10 = max(5, aqi * 1.1 + random.gauss(0, 10))
    o3 = max(1, 40 + random.gauss(0, 10) - 0.1 * rush_hour_bump)
    no2 = max(1, 20 + 0.15 * rush_hour_bump + random.gauss(0, 5))
    so2 = max(0, 8 + random.gauss(0, 3))
    co = max(0, 5 + 0.05 * rush_hour_bump + random.gauss(0, 1.5))

    # Weather: warmer in summer, cooler at night; humidity inversely related to temp
    seasonal_temp = {12: 20, 1: 19, 2: 22, 3: 27, 4: 32, 5: 35, 6: 34,
                     7: 31, 8: 30, 9: 31, 10: 30, 11: 25}[month]
    diurnal_temp_swing = 6 * math.sin((hour - 6) / 24 * 2 * math.pi)
    temperature = seasonal_temp + diurnal_temp_swing + random.gauss(0, 1.5)
    humidity = max(10, min(95, 70 - diurnal_temp_swing * 2 + random.gauss(0, 5)))
    pressure = 1008 + random.gauss(0, 3)
    wind = max(0, 8 + random.gauss(0, 3))

    aqi_change_rate = (aqi - previous_aqi) if previous_aqi is not None else 0.0

    return {
        "timestamp": dt.isoformat(),
        "hour": hour,
        "day": dt.day,
        "month": month,
        "day_of_week": dt.weekday(),
        "aqi": round(aqi, 1),
        "pm25": round(pm25, 1),
        "pm10": round(pm10, 1),
        "o3": round(o3, 1),
        "no2": round(no2, 1),
        "so2": round(so2, 1),
        "co": round(co, 2),
        "temperature": round(temperature, 1),
        "humidity": round(humidity, 1),
        "pressure": round(pressure, 1),
        "wind": round(wind, 1),
        "aqi_change_rate": round(aqi_change_rate, 1),
    }


def generate(days: int, out_path: Path) -> int:
    """Generates `days` worth of hourly synthetic rows and writes them to CSV."""
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)

    rows = []
    prev_aqi = None
    dt = start
    while dt <= end:
        row = simulate_row(dt, prev_aqi)
        rows.append(row)
        prev_aqi = row["aqi"]
        dt += timedelta(hours=1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=60, help="How many days of hourly history to simulate")
    args = parser.parse_args()

    n = generate(args.days, FEATURES_CSV)
    print(f"Wrote {n} synthetic rows ({args.days} days, hourly) to {FEATURES_CSV}")
