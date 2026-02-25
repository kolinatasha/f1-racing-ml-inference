"""Train tire degradation model predicting lap time delta vs fresh tire."""

import pathlib
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb


class XGBWrapper:
    """Thin wrapper so the trained booster exposes model.predict(list_of_lists)."""
    def __init__(self, booster: xgb.Booster) -> None:
        self.booster = booster

    def predict(self, X):
        import numpy as np
        dmat = xgb.DMatrix(np.array(X, dtype=np.float32))
        return self.booster.predict(dmat)


class XGBWrapper:
    """Thin wrapper so pickled model exposes model.predict(list_of_lists)."""
    def __init__(self, booster: xgb.Booster) -> None:
        self.booster = booster

    def predict(self, X):
        dmat = xgb.DMatrix(np.array(X, dtype=np.float32))
        return self.booster.predict(dmat)

DATA_DIR = pathlib.Path(__file__).resolve().parent
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models" / "degradation"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# Must match X construction in lambda/inference/handler.py _handle_predict_tire_degradation.
_FEATURE_COLS = ["laps_on_tire", "track_temp", "driver_aggression"]


def load_dataset() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "laps_features.parquet")
    # laptime_delta_vs_fresh is pre-computed in feature_engineering_laptime.
    # Drop any rows where degradation target or features are missing.
    df = df.dropna(subset=_FEATURE_COLS + ["laptime_delta_vs_fresh"])
    return df


def train_model(df: pd.DataFrame) -> XGBWrapper:
    feature_cols = _FEATURE_COLS
    X = df[feature_cols].values.astype(np.float32)
    y = df["laptime_delta_vs_fresh"].values.astype(np.float32)

    dtrain = xgb.DMatrix(X, label=y)
    params = {
        "max_depth": 6,
        "eta": 0.05,
        "objective": "reg:squarederror",
        "subsample": 0.8,
        "nthread": -1,
        "seed": 42,
    }
    booster = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=False)
    return XGBWrapper(booster)


def main() -> None:
    df = load_dataset()
    model = train_model(df)

    version = "v1_degradation"
    meta = {
        "version": version,
        "description": "Lap time delta vs fresh tire from FastF1 laps",
    }

    out_dir = MODELS_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)

    model.booster.save_model(str(out_dir / "model.json"))

    with open(out_dir / "metadata.pkl", "wb") as f:
        pickle.dump(meta, f)

    print(f"Saved degradation model version {version} to {out_dir}")


if __name__ == "__main__":
    main()
