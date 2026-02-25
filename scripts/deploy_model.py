"""Upload trained models from ./data/models to S3 with versioning.

This script takes local trained model directories and syncs them to
S3 in the layout expected by the Lambda inference layer:
- s3://f1-ml-models/laptime-models/{version}/model.pkl
- s3://f1-ml-models/strategy-models/{version}/model.pkl
- s3://f1-ml-models/degradation-models/{version}/model.pkl
"""

import os
import pathlib

import boto3


def main() -> None:
    region = os.environ.get("AWS_REGION", "eu-west-1")
    bucket = os.environ.get("F1_MODELS_BUCKET", "f1-ml-models")
    session = boto3.Session(region_name=region)
    s3 = session.client("s3")

    root = pathlib.Path(__file__).resolve().parents[1] / "data" / "models"
    for model_type in ["laptime", "strategy", "degradation"]:
        base = root / model_type
        if not base.exists():
            continue
        for version_dir in base.iterdir():
            if not version_dir.is_dir():
                continue
            version = version_dir.name
            for fname in ["model.json", "metadata.pkl"]:
                src = version_dir / fname
                if not src.exists():
                    continue
                key = f"{model_type}-models/{version}/{fname}"
                print(f"Uploading {src} to s3://{bucket}/{key}")
                s3.upload_file(str(src), bucket, key)


if __name__ == "__main__":
    main()
