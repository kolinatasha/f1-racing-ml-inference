"""Validate F1 inference API predictions against real 2024 race data."""

import os
from typing import Dict

import fastf1
import numpy as np
import pandas as pd
import requests

API_BASE = os.environ.get(
    "F1_API_BASE", "https://u59hxhp9l5.execute-api.us-east-1.amazonaws.com/prod"
)


def validate_race(year: int = 2024, race: str = "Bahrain") -> Dict[str, float]:
    fastf1.Cache.enable_cache("./data/raw/fastf1_cache")

    session = fastf1.get_session(year, race, "R")
    session.load()

    laps = session.laps.reset_index(drop=True)

    # Apply the same outlier filter used in training
    laps = laps[~laps["LapTime"].isna()].copy()
    laps["lap_time_seconds"] = laps["LapTime"].dt.total_seconds()
    laps = laps[
        (laps["lap_time_seconds"] >= 60)
        & (laps["lap_time_seconds"] <= 200)
        & (~laps["IsPersonalBest"].isna() | laps["IsPersonalBest"].isna())  # keep all
        & (laps["PitOutTime"].isna())   # exclude pit-out laps
        & (laps["PitInTime"].isna())    # exclude pit-in laps
        & (~laps["Compound"].isna())
    ]

    total_laps = len(laps["LapNumber"].unique()) if "LapNumber" in laps.columns else 58

    errors = []
    skipped = 0
    for _, row in laps.iterrows():
        lap_num = int(row.get("LapNumber", 30))
        # Estimate fuel load: starts ~100 kg, burns ~1.6 kg/lap
        fuel_est = max(5.0, 100.0 - lap_num * 1.6)

        payload = {
            "driver": row["Driver"],
            "track": race.lower(),
            "tire_compound": str(row["Compound"]),
            "tire_age_laps": int(row["TyreLife"] or 0),
            "fuel_load_kg": round(fuel_est, 1),
            "track_temp": 38,
            "air_temp": 24,
        }

        try:
            r = requests.post(f"{API_BASE}/predict/laptime", json=payload, timeout=10)
            r.raise_for_status()
        except Exception as e:
            skipped += 1
            continue

        pred = r.json()["predicted_laptime"]
        mins, secs = pred.split(":")
        pred_secs = int(mins) * 60 + float(secs)
        actual_secs = row["lap_time_seconds"]
        errors.append(abs(pred_secs - actual_secs))

    arr = np.array(errors)
    print(f"Validated {len(errors)} laps (skipped {skipped} errors)")
    return {
        "rmse": float(np.sqrt((arr ** 2).mean())),
        "mae": float(arr.mean()),
        "p95_error": float(np.percentile(arr, 95)),
        "n_laps": len(errors),
    }


if __name__ == "__main__":
    import sys
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
    race = sys.argv[2] if len(sys.argv) > 2 else "Bahrain"
    metrics = validate_race(year=year, race=race)
    print(f"Validation metrics ({year} {race}):", metrics)
