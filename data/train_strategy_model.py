"""Train pit stop strategy model using engineered lap/race-level data.

This is a simplified placeholder; in a full project, you would
build race-level sequences and label optimal pit laps from history.
"""

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
MODELS_DIR = DATA_DIR / "models" / "strategy"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# Must match the feature_order list in lambda/inference/handler.py _handle_predict_pit_strategy.
_FEATURE_COLS = [
    "current_lap",
    "total_laps",
    "tire_age_laps",
    "gap_ahead_seconds",
    "gap_behind_seconds",
]


def load_dataset() -> pd.DataFrame:
    # pit_now_score label is built in feature_engineering_laptime.
    df = pd.read_parquet(PROCESSED_DIR / "laps_features.parquet")
    return df


def train_model(df: pd.DataFrame) -> XGBWrapper:
    feature_cols = _FEATURE_COLS
    X = df[feature_cols].values.astype(np.float32)
    y = df["pit_now_score"].values.astype(np.float32)

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

    version = "v1_strategy"
    meta = {
        "version": version,
        "description": "Synthetic pit strategy score from lap features",
    }

    out_dir = MODELS_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)

    model.booster.save_model(str(out_dir / "model.json"))

    with open(out_dir / "metadata.pkl", "wb") as f:
        pickle.dump(meta, f)

    print(f"Saved strategy model version {version} to {out_dir}")


if __name__ == "__main__":
    main()
  