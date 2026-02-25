import os
import io
import time
from typing import Any, Dict, Tuple

import boto3
import numpy as np
import xgboost as xgb
from aws_xray_sdk.core import xray_recorder, patch_all

patch_all()

_s3_client = boto3.client("s3")

# Global in-memory cache: { (model_type, version): (model, metadata, loaded_at) }
_MODEL_CACHE: Dict[Tuple[str, str], Tuple[Any, Dict[str, Any], float]] = {}


class XGBWrapper:
    """Deserializes an XGBoost booster from raw JSON bytes and exposes .predict()."""
    def __init__(self, model_bytes: bytes) -> None:
        self.booster = xgb.Booster()
        self.booster.load_model(bytearray(model_bytes))

    def predict(self, X):
        dmat = xgb.DMatrix(np.array(X, dtype=np.float32))
        return self.booster.predict(dmat)


def _get_model_config(model_type: str) -> Tuple[str, str]:
    """Return (bucket, key_prefix) for a given model type.

    Model types: "laptime", "strategy", "degradation".
    """
    bucket = os.environ.get("F1_MODELS_BUCKET", "f1-ml-models")
    prefix_env = {
        "laptime": "F1_LAPTIME_PREFIX",
        "strategy": "F1_STRATEGY_PREFIX",
        "degradation": "F1_DEGRADATION_PREFIX",
    }.get(model_type)
    default_prefix = {
        "laptime": "laptime-models/",
        "strategy": "strategy-models/",
        "degradation": "degradation-models/",
    }[model_type]
    key_prefix = os.environ.get(prefix_env, default_prefix)
    return bucket, key_prefix


def _get_target_version(model_type: str) -> str:
    """Return desired model version for this Lambda alias/environment.

    Controlled via env vars so canary deployments can split traffic.
    """
    env_key = {
        "laptime": "F1_LAPTIME_VERSION",
        "strategy": "F1_STRATEGY_VERSION",
        "degradation": "F1_DEGRADATION_VERSION",
    }.get(model_type)
    return os.environ.get(env_key, "v1")


@xray_recorder.capture("load_f1_model")
def load_f1_model(model_type: str) -> Tuple[Any, Dict[str, Any]]:
    """Load and cache an F1 model from S3.

    Returns (model, metadata_dict).
    """
    target_version = _get_target_version(model_type)
    cache_key = (model_type, target_version)

    if cache_key in _MODEL_CACHE:
        model, metadata, loaded_at = _MODEL_CACHE[cache_key]
        xray_recorder.current_subsegment().put_annotation("cache_hit", True)
        return model, metadata

    bucket, prefix = _get_model_config(model_type)
    # Artifacts: prefix/{version}/model.json (XGBoost native) + metadata.pkl
    model_key = f"{prefix}{target_version}/model.json"
    metadata_key = f"{prefix}{target_version}/metadata.pkl"

    start = time.perf_counter()
    obj = _s3_client.get_object(Bucket=bucket, Key=model_key)
    body = obj["Body"].read()
    model = XGBWrapper(body)

    metadata: Dict[str, Any] = {
        "version": target_version,
        "s3_bucket": bucket,
        "s3_key": model_key,
    }

    try:
        meta_obj = _s3_client.get_object(Bucket=bucket, Key=metadata_key)
        meta_body = meta_obj["Body"].read()
        import pickle, io
        meta_from_s3 = pickle.load(io.BytesIO(meta_body))
        if isinstance(meta_from_s3, dict):
            metadata.update(meta_from_s3)
    except _s3_client.exceptions.NoSuchKey:
        pass

    load_latency_ms = (time.perf_counter() - start) * 1000.0
    metadata["load_latency_ms"] = load_latency_ms

    _MODEL_CACHE[cache_key] = (model, metadata, time.time())

    subsegment = xray_recorder.current_subsegment()
    subsegment.put_annotation("model_type", model_type)
    subsegment.put_annotation("model_version", target_version)
    subsegment.put_metadata("model_metadata", metadata)

    return model, metadata


def get_cached_models_snapshot() -> Dict[str, Dict[str, Any]]:
    """Return a lightweight view of cached models for debugging/metrics."""
    snapshot: Dict[str, Dict[str, Any]] = {}
    for (model_type, version), (_, meta, loaded_at) in _MODEL_CACHE.items():
        snapshot[f"{model_type}:{version}"] = {
            "version": version,
            "loaded_at": loaded_at,
            "load_latency_ms": meta.get("load_latency_ms"),
        }
    return snapshot
