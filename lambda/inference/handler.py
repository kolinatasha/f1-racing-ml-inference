import json
import time
from typing import Any, Dict

from aws_xray_sdk.core import xray_recorder, patch_all

from .models import load_f1_model
from .features import get_track_features, get_tire_features

patch_all()


def _response(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _parse_body(event: Dict[str, Any]) -> Dict[str, Any]:
    if "body" in event and isinstance(event["body"], str):
        return json.loads(event["body"])
    return event.get("body", {}) or {}


@xray_recorder.capture("predict_laptime")
def _handle_predict_laptime(payload: Dict[str, Any]) -> Dict[str, Any]:
    start = time.perf_counter()

    model, meta = load_f1_model("laptime")
    track_id = payload.get("track")
    track = get_track_features(track_id) if track_id else None
    tire = get_tire_features(payload.get("tire_compound", "").lower())

    # Simple, explicit feature vector construction; must match training.
    features = {
        "tire_age_laps": payload.get("tire_age_laps", 0),
        "tire_compound": payload.get("tire_compound", "SOFT"),
        "track_temp": payload.get("track_temp"),
        "air_temp": payload.get("air_temp"),
        "fuel_load_kg": payload.get("fuel_load_kg"),
        "track_id": track_id,
        "track_degradation_factor": (track or {}).get("degradation_factor", 1.0),
        "tire_grip_level": (tire or {}).get("grip_level", 1.0),
    }

    # Convert to ordered list/array as expected by model
    feature_order = [
        "tire_age_laps",
        "track_temp",
        "air_temp",
        "fuel_load_kg",
        "track_degradation_factor",
        "tire_grip_level",
    ]
    X = [[features[f] for f in feature_order]]

    predicted_seconds: float = float(model.predict(X)[0])  # type: ignore[attr-defined]

    # Placeholder CI – in a real system derive from residuals
    ci_lower = predicted_seconds - 0.3
    ci_upper = predicted_seconds + 0.3

    latency_ms = (time.perf_counter() - start) * 1000.0

    def fmt_laptime(seconds: float) -> str:
        minutes = int(seconds // 60)
        rem = seconds - minutes * 60
        return f"{minutes}:{rem:06.3f}"

    body = {
        "predicted_laptime": fmt_laptime(predicted_seconds),
        "confidence_interval": [
            fmt_laptime(ci_lower),
            fmt_laptime(ci_upper),
        ],
        "model_version": meta.get("version"),
        "latency_ms": round(latency_ms, 2),
        "track": (track or {}).get("track_name", track_id),
        "conditions": ("optimal" if tire and track else "estimated"),
    }

    return body


@xray_recorder.capture("calculate_pit_strategy")
def _handle_predict_pit_strategy(payload: Dict[str, Any]) -> Dict[str, Any]:
    start = time.perf_counter()

    model, meta = load_f1_model("strategy")

    current_lap = int(payload.get("current_lap", 1))
    total_laps = int(payload.get("total_laps", 50))
    tire_age = int(payload.get("tire_age_laps", 0))
    gap_ahead = float(payload.get("gap_ahead_seconds", 0.0))
    gap_behind = float(payload.get("gap_behind_seconds", 0.0))

    feature_order = [
        "current_lap",
        "total_laps",
        "tire_age_laps",
        "gap_ahead_seconds",
        "gap_behind_seconds",
    ]
    X = [[current_lap, total_laps, tire_age, gap_ahead, gap_behind]]

    pred = model.predict(X)[0]  # type: ignore[attr-defined]

    # Map raw model output to strategy classes – placeholder logic
    if pred <= 0.33:
        recommendation = "pit_now"
    elif pred <= 0.66:
        recommendation = "extend_5_laps"
    else:
        recommendation = "no_stop_window"

    # Simple heuristic outputs; in a real model these would be predicted.
    optimal_pit_window = [max(current_lap - 1, 1), min(current_lap + 2, total_laps)]
    predicted_position_after_pit = int(payload.get("current_position", 5)) + 1
    estimated_tire_life_remaining = max(0, total_laps - current_lap - 5)
    predicted_degradation_per_lap = 0.08

    latency_ms = (time.perf_counter() - start) * 1000.0

    return {
        "recommendation": recommendation,
        "optimal_pit_window": optimal_pit_window,
        "predicted_position_after_pit": predicted_position_after_pit,
        "estimated_tire_life_remaining": estimated_tire_life_remaining,
        "predicted_laptime_degradation_per_lap": predicted_degradation_per_lap,
        "alternative_strategy": "extend_5_laps" if recommendation != "extend_5_laps" else "pit_now",
        "model_version": meta.get("version"),
        "latency_ms": round(latency_ms, 2),
    }


@xray_recorder.capture("predict_tire_degradation")
def _handle_predict_tire_degradation(payload: Dict[str, Any]) -> Dict[str, Any]:
    start = time.perf_counter()

    model, meta = load_f1_model("degradation")

    laps_on_tire = int(payload.get("laps_on_tire", 0))
    track_temp = float(payload.get("track_temp", 30.0))
    driver_style = payload.get("driver_style", "normal")

    driver_aggression = {
        "conservative": 0.7,
        "normal": 1.0,
        "aggressive": 1.3,
    }.get(driver_style, 1.0)

    X = [[laps_on_tire, track_temp, driver_aggression]]

    laptime_delta = float(model.predict(X)[0])  # type: ignore[attr-defined]

    # Convert laptime delta into approximate degradation percentage
    current_degradation_percent = min(80.0, max(0.0, laptime_delta * 20))
    predicted_remaining_laps = max(0, int((100.0 - current_degradation_percent) / 2))
    cliff_expected_lap = laps_on_tire + predicted_remaining_laps

    latency_ms = (time.perf_counter() - start) * 1000.0

    return {
        "current_degradation_percent": round(current_degradation_percent, 2),
        "predicted_remaining_laps": predicted_remaining_laps,
        "laptime_delta_vs_fresh": round(laptime_delta, 3),
        "recommended_action": "monitor" if current_degradation_percent < 60 else "box_soon",
        "cliff_expected_lap": cliff_expected_lap,
        "model_version": meta.get("version"),
        "latency_ms": round(latency_ms, 2),
    }


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:  # type: ignore[override]
    """Entry point for API Gateway proxy integration.

    Routes:
      POST /predict/laptime
      POST /predict/pit-strategy
      POST /predict/tire-degradation
    """
    path = event.get("path", "")
    http_method = event.get("httpMethod", "GET")

    if http_method != "POST":
        return _response(405, {"message": "Method not allowed"})

    payload = _parse_body(event)

    if path.endswith("/predict/laptime"):
        body = _handle_predict_laptime(payload)
        return _response(200, body)

    if path.endswith("/predict/pit-strategy"):
        body = _handle_predict_pit_strategy(payload)
        return _response(200, body)

    if path.endswith("/predict/tire-degradation"):
        body = _handle_predict_tire_degradation(payload)
        return _response(200, body)

    return _response(404, {"message": "Not found", "path": path})
