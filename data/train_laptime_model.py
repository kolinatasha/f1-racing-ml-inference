"""Train lap time prediction model (XGBoost native API) using FastF1 data."""

import pathlib
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb


class XGBWrapper:
    """Thin wrapper so the trained booster exposes model.predict(list_of_lists).
    Only used locally; Lambda uses its own copy in inference/models.py.
    """
    def __init__(self, booster: xgb.Booster) -> None:
        self.booster = booster

    def predict(self, X):
        dmat = xgb.DMatrix(np.array(X, dtype=np.float32))
        return self.booster.predict(dmat)

DATA_DIR = pathlib.Path(__file__).resolve().parent
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models" / "laptime"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# Must match the feature_order list in lambda/inference/handler.py _handle_predict_laptime.
FEATURE_COLUMNS = [
    "tire_age_laps",
    "track_temp",
    "air_temp",
    "fuel_load_kg",
    "track_degradation_factor",
    "tire_grip_level",
]
TARGET_COLUMN = "lap_time_seconds"


def load_dataset() -> pd.DataFrame:
    path = PROCESSED_DIR / "laps_features.parquet"
    return pd.read_parquet(path)


def train_models(df: pd.DataFrame) -> XGBWrapper:
    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = df[TARGET_COLUMN].values.astype(np.float32)

    dtrain = xgb.DMatrix(X, label=y)
    params = {
        "max_depth": 8,
        "eta": 0.05,
        "objective": "reg:squarederror",
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "nthread": -1,
        "seed": 42,
    }
    booster = xgb.train(params, dtrain, num_boost_round=400, verbose_eval=False)
    return XGBWrapper(booster)


def evaluate_model(model: XGBWrapper, df: pd.DataFrame, name: str) -> float:
    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = df[TARGET_COLUMN].values.astype(np.float32)

    preds = model.predict(X)
    rmse = float(np.sqrt(np.mean((y - preds) ** 2)))
    print(f"{name} RMSE: {rmse:.3f} seconds")
    return rmse


def save_model(model: XGBWrapper, version: str, rmse: float, season_tag: str) -> None:
    meta = {
        "version": version,
        "rmse_seconds": rmse,
        "season": season_tag,
    }
    out_dir = MODELS_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save booster in XGBoost native JSON format — avoids pickle class path issues.
    model.booster.save_model(str(out_dir / "model.json"))

    with open(out_dir / "metadata.pkl", "wb") as f:
        pickle.dump(meta, f)

    print(f"Saved model version {version} to {out_dir}")


def main() -> None:
    df = load_dataset()

    model = train_models(df)
    rmse = evaluate_model(model, df, "XGBoost")

    version = "v2_2024_season"
    season_tag = "2023-2024"

    save_model(model, version, rmse, season_tag)


if __name__ == "__main__":
    main()
