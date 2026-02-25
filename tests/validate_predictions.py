"""Validate F1 inference API predictions against real 2024 race data."""

import os
from typing import Dict

import fastf1
import numpy as np
import pandas as pd
import requests

API_BASE = os.environ.get(
    "F1_API_BASE", "https://your-api-id.execute-api.region.amazonaws.com/prod"
)


def validate_race(year: int = 2024, race: str = "Bahrain") -> Dict[str, float]:
    fastf1.Cache.enable_cache("./data/raw/fastf1_cache")

    session = fastf1.get_session(year, race, "R")
    session.load()

    laps = session.laps.reset_index(drop=True)

    errors = []
    for _, row in laps.iterrows():
        if pd.isna(row["LapTime"]):
            continue

        payload = {
            "driver": row["Driver"],
            "track": race.lower(),
            "tire_compound": str(row["Compound"]),
            "tire_age_laps": int(row["TyreLife"] or 0),
            "fuel_load_kg": 60,
            "track_temp": 38,
            "air_temp": 24,
        }

        r = requests.post(f"{API_BASE}/predict/laptime", json=payload)
        r.raise_for_status()
        pred = r.json()["predicted_laptime"]

        mins, secs = pred.split(":")
        pred_secs = int(mins) * 60 + float(secs)
        actual_secs = row["LapTime"].total_seconds()
        errors.append(abs(pred_secs - actual_secs))

    arr = np.array(errors)
    return {
        "rmse": float(np.sqrt((arr ** 2).mean())),
        "mae": float(np.abs(arr).mean()),
    }


if __name__ == "__main__":
    metrics = validate_race()
    print("Validation metrics:", metrics)
