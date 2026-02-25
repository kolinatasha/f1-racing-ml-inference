"""Collect historical F1 telemetry with FastF1 for 2023-2024 seasons.

This script builds lap-level datasets for lap time, tire degradation,
strategy modeling, and writes them as CSV/Parquet for training.
"""

import pathlib
from typing import List

import fastf1
import numpy as np
import pandas as pd

DATA_DIR = pathlib.Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def enable_fastf1_cache() -> None:
    cache_dir = RAW_DIR / "fastf1_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))


# Per-track static degradation factors and typical temperatures.
_TRACK_META = {
    "Bahrain":          {"degradation_factor": 1.1, "track_temp": 42.0, "air_temp": 28.0},
    "Saudi Arabia":     {"degradation_factor": 0.8, "track_temp": 30.0, "air_temp": 25.0},
    "Australia":        {"degradation_factor": 0.9, "track_temp": 32.0, "air_temp": 22.0},
    "Azerbaijan":       {"degradation_factor": 0.8, "track_temp": 28.0, "air_temp": 21.0},
    "Miami":            {"degradation_factor": 1.0, "track_temp": 40.0, "air_temp": 30.0},
    "Japan":            {"degradation_factor": 0.95, "track_temp": 26.0, "air_temp": 18.0},
}

# Grip level by compound.
_COMPOUND_GRIP = {"SOFT": 1.0, "MEDIUM": 0.9, "HARD": 0.85, "INTERMEDIATE": 0.7, "WET": 0.55}


def collect_season_laps(year: int, races: List[str]) -> pd.DataFrame:
    all_laps: List[pd.DataFrame] = []
    for race in races:
        print(f"Loading {year} {race} race session...")
        session = fastf1.get_session(year, race, "R")
        session.load()

        laps = session.laps.reset_index(drop=True)
        total_laps = int(laps["LapNumber"].max())

        # Pull mean weather for the session; fall back to static defaults.
        meta = _TRACK_META.get(race, {"degradation_factor": 1.0, "track_temp": 30.0, "air_temp": 22.0})
        session_track_temp = meta["track_temp"]
        session_air_temp = meta["air_temp"]
        try:
            w = session.weather_data
            if w is not None and not w.empty:
                session_track_temp = float(w["TrackTemp"].mean())
                session_air_temp = float(w["AirTemp"].mean())
        except Exception:
            pass

        df = pd.DataFrame()
        df["year"] = year
        df["grand_prix"] = race
        df["driver"] = laps["Driver"]
        df["lap_number"] = laps["LapNumber"]
        df["lap_time_seconds"] = laps["LapTime"].dt.total_seconds()
        df["sector1_time"] = laps["Sector1Time"].dt.total_seconds()
        df["sector2_time"] = laps["Sector2Time"].dt.total_seconds()
        df["sector3_time"] = laps["Sector3Time"].dt.total_seconds()
        df["compound"] = laps["Compound"]
        df["tire_life_laps"] = laps["TyreLife"]
        df["track_status"] = laps["TrackStatus"]
        df["is_pit_out_lap"] = laps["PitOutTime"].notna()
        df["is_pit_in_lap"] = laps["PitInTime"].notna()

        # Fuel: F1 cars start with ~110 kg and burn ~2.1 kg/lap.
        df["fuel_load_proxy"] = (total_laps - df["lap_number"]) / max(total_laps, 1)
        df["fuel_load_kg"] = np.clip(
            110.0 - df["lap_number"] * 2.1, 5.0, 110.0
        )
        df["total_laps"] = total_laps

        # Weather (session-level mean or static fallback).
        df["track_temp"] = session_track_temp
        df["air_temp"] = session_air_temp
        df["track_degradation_factor"] = meta["degradation_factor"]

        # Gap placeholders – not available in basic lap data; centre at realistic mean.
        rng = np.random.default_rng(seed=hash(f"{year}{race}") & 0xFFFF)
        df["gap_ahead_seconds"] = rng.uniform(0.5, 8.0, len(df)).round(2)
        df["gap_behind_seconds"] = rng.uniform(0.5, 8.0, len(df)).round(2)

        # Driver aggression proxy (neutral for all; can be overridden per driver).
        df["driver_aggression"] = 1.0

        all_laps.append(df)

    return pd.concat(all_laps, ignore_index=True)


def feature_engineering_laptime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop invalid or missing lap times.
    df = df.replace([np.inf, -np.inf], pd.NA)
    df = df.dropna(subset=["lap_time_seconds", "compound", "tire_life_laps"])

    # Exclude outlier laps: pit-in/out laps, SC laps, clearly invalid timing.
    # F1 laps in dry conditions range ~60s (Monaco) to ~105s (Monaco SC).
    df = df[df["lap_time_seconds"].between(60.0, 200.0)]
    # Remove pit-in and pit-out laps (large time outliers).
    df = df[~df["is_pit_out_lap"] & ~df["is_pit_in_lap"]]

    # One-hot encode tire compound.
    compound_dummies = pd.get_dummies(df["compound"], prefix="compound")
    df = pd.concat([df, compound_dummies], axis=1)

    # Legacy degradation proxy (kept for backwards compat).
    df["tire_degradation_feature"] = df["tire_life_laps"].fillna(0) * df["fuel_load_proxy"].fillna(0)

    # Track ID.
    df["track_id"] = df["grand_prix"].astype(str).str.lower().str.replace(" ", "_")

    # ── Features aligned with Lambda handler feature vectors ──────────────────

    # laptime handler: [tire_age_laps, track_temp, air_temp, fuel_load_kg,
    #                   track_degradation_factor, tire_grip_level]
    df["tire_age_laps"] = df["tire_life_laps"].fillna(0).astype(np.float32)
    df["tire_grip_level"] = (
        df["compound"].map(_COMPOUND_GRIP).fillna(1.0).astype(np.float32)
    )
    # fuel_load_kg, track_temp, air_temp, track_degradation_factor already added
    # in collect_season_laps; fill any gaps from training-only runs.
    for col, default in [
        ("fuel_load_kg", 60.0),
        ("track_temp", 30.0),
        ("air_temp", 22.0),
        ("track_degradation_factor", 1.0),
        ("total_laps", 58.0),
        ("gap_ahead_seconds", 3.0),
        ("gap_behind_seconds", 3.0),
        ("driver_aggression", 1.0),
    ]:
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default).astype(np.float32)

    # strategy handler: [current_lap, total_laps, tire_age_laps,
    #                    gap_ahead_seconds, gap_behind_seconds]
    df["current_lap"] = df["lap_number"].fillna(1).astype(np.float32)

    # degradation handler: [laps_on_tire, track_temp, driver_aggression]
    df["laps_on_tire"] = df["tire_age_laps"]

    # Approx base pace for degradation target (used in train_degradation_model).
    base_pace = df.loc[df["tire_age_laps"] <= 3, "lap_time_seconds"].median()
    if pd.isna(base_pace):
        base_pace = df["lap_time_seconds"].median()
    df["laptime_delta_vs_fresh"] = (df["lap_time_seconds"] - base_pace).astype(np.float32)

    # Synthetic pit strategy label: high score = should pit soon.
    df["pit_now_score"] = np.clip(
        (df["tire_age_laps"] - 15.0) / 10.0 + df["fuel_load_proxy"],
        0.0, 1.0,
    ).astype(np.float32)

    return df


def main() -> None:
    enable_fastf1_cache()

    races_2023 = ["Bahrain", "Saudi Arabia", "Australia", "Azerbaijan", "Miami"]
    races_2024 = ["Bahrain", "Saudi Arabia", "Australia", "Japan"]

    df_2023 = collect_season_laps(2023, races_2023)
    df_2024 = collect_season_laps(2024, races_2024)

    all_laps = pd.concat([df_2023, df_2024], ignore_index=True)
    all_laps.to_parquet(PROCESSED_DIR / "laps_raw.parquet", index=False)

    fe = feature_engineering_laptime(all_laps)
    fe.to_parquet(PROCESSED_DIR / "laps_features.parquet", index=False)

    print(f"Saved raw laps to {PROCESSED_DIR / 'laps_raw.parquet'}")
    print(f"Saved feature-engineered laps to {PROCESSED_DIR / 'laps_features.parquet'}")


if __name__ == "__main__":
    main()
