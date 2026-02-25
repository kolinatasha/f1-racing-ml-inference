"""Seed the f1-features DynamoDB table with track, tire, and weather data."""

import os
from typing import Dict, List

import boto3


def get_table():
    region = os.environ.get("AWS_REGION", "eu-west-1")
    table_name = os.environ.get("F1_FEATURES_TABLE", "f1-features")
    session = boto3.Session(region_name=region)
    dynamodb = session.resource("dynamodb")
    return dynamodb.Table(table_name)
def seed_items(table, items: List[Dict]):
    from decimal import Decimal
    import copy
    def convert_floats(obj):
        if isinstance(obj, float):
            return Decimal(str(obj))
        elif isinstance(obj, dict):
            return {k: convert_floats(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_floats(x) for x in obj]
        else:
            return obj
    with table.batch_writer() as batch:
        for item in items:
            batch.put_item(Item=convert_floats(item))


def main() -> None:
    table = get_table()

    tracks = [
        {
            "feature_type": "track",
            "feature_id": "monaco",
            "track_name": "Monaco",
            "lap_distance_km": 3.337,
            "corners": 19,
            "avg_speed_kmh": 160,
            "degradation_factor": 1.2,
        },
        {
            "feature_type": "track",
            "feature_id": "silverstone",
            "track_name": "Silverstone",
            "lap_distance_km": 5.891,
            "corners": 18,
            "avg_speed_kmh": 230,
            "degradation_factor": 0.9,
        },
        {
            "feature_type": "track",
            "feature_id": "spa",
            "track_name": "Spa-Francorchamps",
            "lap_distance_km": 7.004,
            "corners": 20,
            "avg_speed_kmh": 220,
            "degradation_factor": 1.0,
        },
    ]

    # feature_id must match compound.lower() used in features.py get_tire_features.
    tires = [
        {
            "feature_type": "tire",
            "feature_id": "soft",
            "compound": "SOFT",
            "optimal_temp_range": [90, 110],
            "expected_life_laps": 15,
            "grip_level": 1.0,
        },
        {
            "feature_type": "tire",
            "feature_id": "medium",
            "compound": "MEDIUM",
            "optimal_temp_range": [85, 105],
            "expected_life_laps": 25,
            "grip_level": 0.9,
        },
        {
            "feature_type": "tire",
            "feature_id": "hard",
            "compound": "HARD",
            "optimal_temp_range": [80, 100],
            "expected_life_laps": 40,
            "grip_level": 0.85,
        },
        {
            "feature_type": "tire",
            "feature_id": "intermediate",
            "compound": "INTERMEDIATE",
            "optimal_temp_range": [30, 60],
            "expected_life_laps": 30,
            "grip_level": 0.7,
        },
        {
            "feature_type": "tire",
            "feature_id": "wet",
            "compound": "WET",
            "optimal_temp_range": [15, 45],
            "expected_life_laps": 20,
            "grip_level": 0.55,
        },
    ]

    weathers = [
        {
            "feature_type": "weather",
            "feature_id": "current_bahrain",
            "track_temp": 42,
            "air_temp": 28,
            "humidity": 45,
            "conditions": "dry",
        }
    ]

    seed_items(table, tracks + tires + weathers)
    print("Seeded f1-features with", len(tracks) + len(tires) + len(weathers), "items")


if __name__ == "__main__":
    main()
