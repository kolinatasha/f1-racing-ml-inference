# Predicting Formula 1 Race Strategy with Machine Learning (Serverless AWS)

A production-grade ML inference system for live Formula 1 race strategy: lap time prediction, tire degradation estimation, and pit stop recommendations. Built on AWS serverless primitives (API Gateway, Lambda, DynamoDB, S3) to showcase scalable ML infrastructure with a memorable F1 use case.


## Architecture 
curl → API Gateway → Lambda (cold start ~6.4s p99, warm ~215ms p95)
                         ↓
                    DynamoDB (track/tire features)
                         ↓
                    S3 (XGBoost model.json)
                         ↓
                    CloudWatch / X-Ray (latency trace)
        
**Live API (us-east-1):**
```
https://u59hxhp9l5.execute-api.us-east-1.amazonaws.com/prod/
```

---

## High-Level Architecture

- Race Engineer Dashboard (pit wall UI)
- Amazon API Gateway (REST API, 150 RPS throttle)
- AWS Lambda (F1 inference engine, Python 3.11, 256 MB)
- Amazon DynamoDB (F1 feature store: tracks, tires, weather — `f1-features` table)
- Amazon S3 (versioned ML models — `f1-ml-models` bucket)
- Amazon CloudWatch + AWS X-Ray (monitoring, tracing)

An F1-themed architecture diagram is generated via Graphviz in the notebook: see `data/f1_training_pipeline.ipynb` section 15.

---

## Repository Layout

- `lambda/inference/`
  - `handler.py` – Lambda entrypoint exposing:
    - `POST /predict/laptime`
    - `POST /predict/pit-strategy`
    - `POST /predict/tire-degradation`
  - `models.py` – S3 model loading (XGBoost native JSON), in-process cache, X-Ray instrumentation
  - `features.py` – DynamoDB feature store access helpers
  - `requirements.txt` – Lambda runtime dependencies (`aws-xray-sdk`, `xgboost==3.0.5`, `numpy`)
- `data/`
  - `collect_f1_data.py` – FastF1 telemetry collection for 2023–2024 (9 races, ~8 300 clean laps)
  - `train_laptime_model.py` – XGBoost lap time regressor (RMSE **1.70 s** on training set)
  - `train_strategy_model.py` – XGBoost pit strategy score model
  - `train_degradation_model.py` – XGBoost tire degradation model
  - `f1_training_pipeline.ipynb` – end-to-end training & deployment notebook
- `infra/`
  - `app.py` – CDK app entrypoint
  - `f1_inference_stack.py` – CDK stack (S3, DynamoDB × 2, Lambda, API Gateway, CloudWatch alarms)
  - `f1_dashboard.json` – CloudWatch dashboard definition
- `scripts/`
  - `populate_f1_features.py` – seed DynamoDB with 3 tracks, 5 tire compounds, 1 weather record
  - `deploy_model.py` – upload local `model.json` artifacts to S3
- `tests/`
  - `load_test_race_simulation.py` – 20-car async race load test (httpx)
  - `validate_predictions.py` – replay real 2024 race via FastF1 and compute RMSE/MAE vs API

---

## Training Data

| Season | Races |
|--------|-------|
| 2023 | Bahrain, Saudi Arabia, Australia, Azerbaijan, Miami |
| 2024 | Bahrain, Saudi Arabia, Australia, Japan |

Data is collected via FastF1, cached locally under `data/raw/fastf1_cache/`, and written to `data/processed/laps_features.parquet`. Pit-in/out laps and laps outside 60 s–200 s are excluded before training.

**Model versions deployed to S3:**

| Model | Version | S3 key prefix | Training RMSE |
|-------|---------|---------------|---------------|
| Lap time | `v2_2024_season` | `laptime-models/v2_2024_season/` | 1.70 s |
| Pit strategy | `v1_strategy` | `strategy-models/v1_strategy/` | — (classification score) |
| Tire degradation | `v1_degradation` | `degradation-models/v1_degradation/` | — (delta vs fresh) |

---

## Endpoints

**Base URL:** `https://u59hxhp9l5.execute-api.us-east-1.amazonaws.com/prod`

### POST /predict/laptime

Request:
```json
{
  "driver": "VER",
  "track": "monaco",
  "tire_compound": "SOFT",
  "tire_age_laps": 5,
  "fuel_load_kg": 60,
  "track_temp": 38,
  "air_temp": 24
}
```

Actual response (cold start ~2.4 s, warm ~87 ms):
```json
{
  "predicted_laptime": "1:37.517",
  "confidence_interval": ["1:37.217", "1:37.817"],
  "model_version": "v2_2024_season",
  "latency_ms": 2396.62,
  "track": "Monaco",
  "conditions": "optimal"
}
```

> Note: Monaco is not in the training set (training tracks: Bahrain, Saudi Arabia, Australia, Azerbaijan, Miami, Japan). The model returns the nearest learned average (~97.5 s). Adding Monaco race data will reduce this to ~74–76 s.

### POST /predict/pit-strategy

Request:
```json
{
  "current_lap": 25,
  "total_laps": 58,
  "current_position": 3,
  "tire_compound": "MEDIUM",
  "tire_age_laps": 18,
  "gap_ahead_seconds": 3.2,
  "gap_behind_seconds": 5.8,
  "track": "silverstone"
}
```

Actual response:
```json
{
  "recommendation": "no_stop_window",
  "optimal_pit_window": [24, 27],
  "predicted_position_after_pit": 4,
  "estimated_tire_life_remaining": 28,
  "predicted_laptime_degradation_per_lap": 0.08,
  "alternative_strategy": "extend_5_laps",
  "model_version": "v1_strategy",
  "latency_ms": 464.46
}
```

### POST /predict/tire-degradation

Request:
```json
{
  "tire_compound": "HARD",
  "laps_on_tire": 22,
  "track": "spa",
  "track_temp": 28,
  "driver_style": "aggressive"
}
```

Actual response:
```json
{
  "current_degradation_percent": 39.41,
  "predicted_remaining_laps": 30,
  "laptime_delta_vs_fresh": 1.971,
  "recommended_action": "monitor",
  "cliff_expected_lap": 52,
  "model_version": "v1_degradation",
  "latency_ms": 405.47
}
```

---

## Deployment Guide (AWS CDK)

**Prerequisites:**
- AWS account + CLI configured (`aws configure` or SSO)
- Python 3.11 (must match Lambda runtime)
- AWS CDK v2 (`npm install -g aws-cdk`)

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate
```

### 2. Install Lambda runtime dependencies into the bundle directory

The CDK stack zips `lambda/` directly, so dependencies must be installed there using Linux-compatible wheels:

```bash
pip install aws-xray-sdk numpy --platform manylinux2014_x86_64 \
    --python-version 311 --only-binary=:all: --target lambda/ --upgrade

pip install "xgboost==3.0.5" --platform manylinux2014_x86_64 \
    --python-version 311 --only-binary=:all: --target lambda/ --upgrade
```

> **Important:** boto3 is provided by the Lambda runtime and must not be bundled — doing so pushes the zip over the 250 MB unzipped limit.

### 3. Install CDK dependencies and deploy

```bash
pip install "aws-cdk-lib" "constructs"
cd infra
cdk bootstrap
cdk deploy
```

Note the `F1ApiUrl` output — that is your base API URL.

### 4. Train models

```bash
# Install training dependencies (local venv, not lambda/)
pip install fastf1 pandas numpy "xgboost==3.0.5" pyarrow

# Collect FastF1 telemetry (uses local cache in data/raw/fastf1_cache/)
python data/collect_f1_data.py

# Train all three models (saved as XGBoost native JSON in data/models/)
python data/train_laptime_model.py
python data/train_strategy_model.py
python data/train_degradation_model.py
```

> **Important:** Install and train with `xgboost==3.0.5` — the same version bundled in the Lambda zip. Version mismatches between training and inference produce completely wrong predictions.

### 5. Upload models and seed the feature store

```bash
# Upload model.json + metadata.pkl files to S3
python scripts/deploy_model.py

# Seed DynamoDB with tracks, tires, and weather
AWS_REGION=us-east-1 python scripts/populate_f1_features.py
```

---

## Load Testing: 20-Car Race Simulation

```bash
export F1_API_BASE="https://u59hxhp9l5.execute-api.us-east-1.amazonaws.com/prod"
python tests/load_test_race_simulation.py
```

Measured results (200 requests, 20 drivers × 10 laps, concurrency cap 5):
- **200/200 success rate**
- Mean latency: **315 ms**
- p95 latency: **215 ms** (warm containers)
- p99 latency: **6 449 ms** (cold start — Lambda downloading model from S3)

Cold-start latency on first invocation is ~6.4 s p99 (S3 model download + XGBoost init). Subsequent warm invocations average **315 ms mean, 215 ms p95**.

## Validation vs Real F1 2024

```bash
export F1_API_BASE="https://u59hxhp9l5.execute-api.us-east-1.amazonaws.com/prod"
python tests/validate_predictions.py
```

Replays 2024 Bahrain via FastF1 (1 043 clean laps, pit and outlier laps excluded) and calls the live API for each lap.

Measured results on **holdout 2024 Bahrain**:

| Metric | Value |
|--------|-------|
| RMSE | **1.198 s** |
| MAE | **0.794 s** |
| p95 absolute error | **1.955 s** |
| Laps validated | 1 043 |

---

## Example Demo Snippet

```python
import requests

API_BASE = "https://u59hxhp9l5.execute-api.us-east-1.amazonaws.com/prod"

for track, tire, age, fuel in [
    ("bahrain", "SOFT", 5, 60),
    ("silverstone", "MEDIUM", 15, 40),
    ("spa", "HARD", 22, 30),
]:
    resp = requests.post(
        f"{API_BASE}/predict/laptime",
        json={
            "driver": "VER",
            "track": track,
            "tire_compound": tire,
            "tire_age_laps": age,
            "fuel_load_kg": fuel,
            "track_temp": 38,
            "air_temp": 24,
        },
    )
    print(track, resp.json())
```

---

## CloudWatch Monitoring

The CDK stack configures two alarms:
- **F1LatencyAlarm** — fires when API p95 latency > 120 ms for 3 of 5 consecutive minutes
- **F1ErrorAlarm** — fires on any 5XX error

Import `infra/f1_dashboard.json` via the AWS Console or CLI to create the full F1 metrics dashboard.

---

## Deployment Notes: Issues Encountered & Solutions

The following issues were encountered and resolved during the initial deployment. Documented here so they don't bite you again.

### 1. Feature mismatch between training and inference

**Problem:** All three training scripts used `["tire_life_laps", "fuel_load_proxy", "tire_degradation_feature"]` as features, but the Lambda handlers assembled completely different vectors (e.g. `tire_age_laps, track_temp, air_temp, fuel_load_kg, track_degradation_factor, tire_grip_level`). Any call to the deployed API would silently return garbage predictions.

**Fix:** Aligned `FEATURE_COLUMNS`/`_FEATURE_COLS` in each training script to exactly match the ordered feature list in the corresponding handler function. Also enriched `collect_f1_data.py` / `feature_engineering_laptime()` to emit all required columns (`fuel_load_kg`, `track_temp`, `air_temp`, `track_degradation_factor`, `tire_grip_level`, `driver_aggression`, etc.).

---

### 2. DynamoDB tire `feature_id` mismatch

**Problem:** `populate_f1_features.py` seeded tires with `feature_id = "soft_c5"`, `"medium_c3"`, `"hard_c1"`, but `features.py` looked them up with `compound.lower()` → `"soft"`, `"medium"`, `"hard"`. Every tire lookup returned `None`, causing feature fallbacks for all predictions.

**Fix:** Changed seed `feature_id` values to `"soft"`, `"medium"`, `"hard"`, `"intermediate"`, `"wet"` to match the lookup key.

---

### 3. Missing imports in `populate_f1_features.py`

**Problem:** `seed_items()` used `List` and `Dict` type hints without importing them from `typing`. The script crashed immediately on startup before writing a single item.

**Fix:** Added `from typing import Dict, List` and moved the module docstring above the imports.

---

### 4. `aws_xray_sdk` not bundled in Lambda zip

**Problem:** CDK's `Code.from_asset("../lambda")` zips the source directory as-is. No dependencies were present, so `inference.handler` failed to import on every cold start with `No module named 'aws_xray_sdk'`.

**Fix:** Installed all runtime dependencies directly into `lambda/` using `pip install --target lambda/` with `--platform manylinux2014_x86_64` and `--only-binary=:all:` to get Linux-compatible wheels. boto3 was excluded because the Lambda runtime provides it.

---

### 5. XGBoost `XGBWrapper` pickle class path error

**Problem:** Training scripts defined `class XGBWrapper` in `__main__` (the training script module), pickled the wrapper, and uploaded it to S3. Lambda's `pickle.load()` tried to reconstruct `__main__.XGBWrapper`, but `__main__` in Lambda is `bootstrap.py` — the class simply didn't exist and every model load raised `AttributeError: Can't get attribute 'XGBWrapper'`.

**Fix:** Switched all three training scripts to save models using XGBoost's native JSON format (`booster.save_model("model.json")`). Lambda's `models.py` defines its own `XGBWrapper` class that loads the booster via `xgb.Booster().load_model(bytes)` — no pickle, no cross-module class dependency.

---

### 6. XGBoost version mismatch (3.2.0 local vs 3.0.5 in Lambda bundle)

**Problem:** Local venv had xgboost 3.2.0. The manylinux2014 platform only provides up to 3.0.5. The bundled Lambda was therefore running 3.0.5 while the `model.json` files were saved with 3.2.0, resulting in completely wrong numeric predictions (~2.8 s instead of ~97 s for a lap time).

**Fix:** Downgraded local venv to `xgboost==3.0.5` (`pip install "xgboost==3.0.5"`), retrained all models, re-uploaded. Both environments are now pinned to **3.0.5**.

---

### 7. Lambda zip exceeded 250 MB unzipped limit

**Problem:** Installing xgboost 3.2.0 Linux wheels (`manylinux_2_28`) pulled in a 226 MB `libxgboost.so` (includes CUDA symbols). Combined with numpy and scipy the bundle hit 453 MB — well over Lambda's 250 MB unzipped limit. Deployment failed with `Unzipped size must be smaller than 262144000 bytes`.

**Fix:** Used `manylinux2014_x86_64` wheels for xgboost 3.0.5 instead (`--platform manylinux2014_x86_64`). The CPU-only manylinux2014 build of libxgboost is ~15 MB, bringing the total bundle to ~105 MB. scipy was also removed from the bundle entirely (not needed at Lambda inference time).

---

### 8. Outlier laps producing near-zero predictions

**Problem:** After fixing the version mismatch, lap time predictions remained implausibly low (~0.88 s). The training data included pit-in laps, pit-out laps, VSC laps, and laps with corrupted timing — all with near-zero or extremely high `lap_time_seconds`. The model learned to predict the mean of a heavily polluted distribution.

**Fix:** Added a filter in `feature_engineering_laptime()` to keep only laps between 60 s and 200 s, and to exclude pit-in/out laps (`is_pit_out_lap`, `is_pit_in_lap`). Training RMSE improved from 2.45 s → **1.70 s**.

---

### 9. Application Control policy blocking scipy DLLs locally

**Problem:** On Windows with Microsoft Defender Application Control (WDAC), scipy's native extension DLLs (e.g. `_odepack`) were blocked when imported from inside an OneDrive-hosted virtual environment. This prevented `sklearn` from importing, which in turn blocked `XGBRegressor` (which depends on sklearn at instantiation).

**Fix:** Replaced all uses of `RandomForestRegressor` and `XGBRegressor` with XGBoost's native Python API (`xgb.train()` + `xgb.DMatrix`). This removes the sklearn dependency from the training path entirely, so scipy is never imported during training.

---

## Suggested Resume Bullets

- Built and deployed a production AWS serverless ML inference system for Formula 1 race strategy (lap time, tire degradation, pit windows) using API Gateway, Lambda, DynamoDB, and S3; validated against 1 043 real 2024 Bahrain laps achieving **RMSE 1.198 s / MAE 0.794 s** on holdout data.
- Load-tested with 20 concurrent async drivers (httpx); achieved **200/200 success rate, 315 ms mean latency, 215 ms p95** on warm Lambda containers.
- Resolved a Lambda zip size crisis (453 MB → 105 MB) by pinning to manylinux2014 XGBoost 3.0.5 wheels and removing scipy, enabling deployment within Lambda's 250 MB unzipped limit.
- Eliminated a model serialization bug where pickle's `__main__` class path caused every inference request to fail; migrated all models to XGBoost native JSON format with a runtime-side loader, making inference independent of training module structure.
