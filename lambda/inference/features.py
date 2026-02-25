import os
from typing import Any, Dict, Optional

import boto3
from aws_xray_sdk.core import xray_recorder

_dynamodb = boto3.resource("dynamodb")


def _get_table(name_env: str, default_name: str):
    table_name = os.environ.get(name_env, default_name)
    return _dynamodb.Table(table_name)


@xray_recorder.capture("get_track_features")
def get_track_features(track_id: str) -> Optional[Dict[str, Any]]:
    table = _get_table("F1_FEATURES_TABLE", "f1-features")
    resp = table.get_item(Key={"feature_type": "track", "feature_id": track_id})
    return resp.get("Item")


@xray_recorder.capture("get_tire_features")
def get_tire_features(compound_id: str) -> Optional[Dict[str, Any]]:
    table = _get_table("F1_FEATURES_TABLE", "f1-features")
    resp = table.get_item(Key={"feature_type": "tire", "feature_id": compound_id})
    return resp.get("Item")


@xray_recorder.capture("get_weather_features")
def get_weather_features(weather_id: str) -> Optional[Dict[str, Any]]:
    table = _get_table("F1_FEATURES_TABLE", "f1-features")
    resp = table.get_item(Key={"feature_type": "weather", "feature_id": weather_id})
    return resp.get("Item")
