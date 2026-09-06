# Pearls AQI Predictor — Karachi

An end-to-end AQI forecasting system: predicts Air Quality Index for
Karachi 24h / 48h / 72h ahead, using an automated feature pipeline,
a trained ML model, and a Streamlit dashboard.

## Architecture

```
AQICN API --> feature_pipeline.py --> data/features.csv (feature store)
                                              |
                                              v
                                   training_pipeline.py
                                              |
                                              v
                                 models/*.joblib (model registry)
                                              |
                                              v
                                          app.py (Streamlit dashboard)

GitHub Actions automates the two pipelines:
  - feature_pipeline.py  runs every hour   (.github/workflows/feature_pipeline.yml)
  - training_pipeline.py runs every day    (.github/workflows/training_pipeline.yml)
```

## Files

| File | Purpose |
|---|---|
| `feature_pipeline.py` | Fetches the latest hour of real air quality + weather from Open-Meteo (no API key needed), computes features, appends to `data/features.csv` |
| `backfill_openmeteo.py` | Pulls real historical hourly air quality + weather from Open-Meteo for training (no API key needed) |
| `generate_synthetic_history.py` | Fallback only — simulates realistic patterns if Open-Meteo is unavailable |
| `training_pipeline.py` | Trains Ridge + Random Forest for 3 forecast horizons, evaluates, saves models + SHAP plots |
| `app.py` | Streamlit dashboard: shows current AQI, 3-day forecast, hazard alerts, SHAP explanations |
| `.github/workflows/*.yml` | CI/CD automation (GitHub Actions) |
| `requirements.txt` | Python dependencies |

## Important note on data — read this before presenting the project

The training dataset uses **real historical measurements**, not
synthetic data. `backfill_openmeteo.py` pulls genuine hourly pollutant
and weather data for Karachi from Open-Meteo's free, no-API-key-required
APIs:

- Air Quality API (CAMS global reanalysis) — real data from Aug 2022 onward
- Historical Weather Archive API (ERA5 reanalysis) — real data for decades

This is the same underlying data source behind the public Zenodo
"Air Quality Dataset for Karachi" — calling the API directly just gives
you full control over the date range instead of a fixed snapshot.

`generate_synthetic_history.py` is kept in the repo as a fallback (e.g.
if Open-Meteo is ever down before a deadline) but is no longer the
primary data source — prefer `backfill_openmeteo.py`.

The live pipeline (`feature_pipeline.py`, automated hourly via AQICN,
via GitHub Actions) keeps collecting fresh real readings going forward,
so the daily retraining job trains on a growing real dataset over time.

## How to run it yourself

### 1. One-time setup
```bash
git clone <your-repo-url>
cd aqi_predictor
pip install -r requirements.txt
```
No API keys or signups needed anywhere in this project — both the
historical backfill and the live pipeline run on Open-Meteo's free,
no-key APIs. This also means the training data and the live data come
from the exact same source and methodology, avoiding train/serve skew.

### 2. Bootstrap training data with REAL history (do this once)
```bash
python backfill_openmeteo.py --start 2024-01-01 --end 2026-09-05
```
Pick a start date as far back as 2022-08-01 for more training data.

### 3. Fetch one real live reading (do this any time)
```bash
python feature_pipeline.py
```

### 4. Train the models
```bash
python training_pipeline.py
```
This prints RMSE / MAE / R² for Ridge and Random Forest at each horizon,
saves the better model per horizon to `models/`, and saves SHAP feature
importance plots.

### 5. Launch the dashboard
```bash
streamlit run app.py
```

### 6. Automate it on GitHub
1. Push this repo to GitHub.
2. No secrets to configure — Open-Meteo needs no API key.
3. The two workflows in `.github/workflows/` will now run automatically
   (hourly feature collection, daily retraining) and commit updates back
   to the repo.

## Model evaluation

See `models/metrics_report.json` after running `training_pipeline.py`
for full RMSE / MAE / R² numbers per horizon and model type. Random
Forest outperformed Ridge Regression at all three horizons in initial
testing on synthetic data, likely because it captures the non-linear
rush-hour / seasonal patterns better than a linear model.

## Extending to a real Feature Store / Model Registry

Currently features are stored in `data/features.csv` and models in
`models/*.joblib` — this satisfies the project requirements functionally
but isn't literally Hopsworks/Vertex AI. To upgrade:

1. Sign up for a free Hopsworks account: https://www.hopsworks.ai
2. `pip install hopsworks`
3. In `feature_pipeline.py`, replace the body of `save_feature_row()`
   with the Hopsworks feature group `.insert()` call (already sketched
   as a comment in that function).
4. In `training_pipeline.py`, replace `load_data()` with a call to
   `fs.get_feature_group(...).read()`, and replace the `joblib.dump()`
   calls with the Hopsworks Model Registry's `.save()` API.

## Known limitations / future work

- Training data is currently synthetic-bootstrapped (see note above).
- Only Ridge and Random Forest are compared; a deep learning model
  (e.g. an LSTM over the hourly sequence) could be added for the
  "advanced models" requirement.
- Weather features come from AQICN's `iaqi` block, which has sparser
  weather coverage than a dedicated weather API — OpenWeather could be
  merged in for more reliable temperature/humidity/wind data.
- No EDA notebook is included yet — recommended next step: a Jupyter
  notebook plotting AQI over time, by hour-of-day, and by month, to
  visually confirm the patterns the model is learning.
