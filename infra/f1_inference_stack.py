import os

import aws_cdk as cdk
from aws_cdk import (
    Duration,
    Stack,
    aws_apigateway as apigw,
    aws_cloudwatch as cloudwatch,
    aws_dynamodb as dynamodb,
    aws_iam as iam,
    aws_lambda as _lambda,
    aws_logs as logs,
    aws_s3 as s3,
)
from constructs import Construct


class F1InferenceStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        bucket = s3.Bucket(
            self,
            "F1ModelsBucket",
            bucket_name=os.environ.get("F1_MODELS_BUCKET_NAME", "f1-ml-models"),
            versioned=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
        )

        features = dynamodb.Table(
            self,
            "F1FeaturesTable",
            table_name=os.environ.get("F1_FEATURES_TABLE_NAME", "f1-features"),
            partition_key=dynamodb.Attribute(
                name="feature_type", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="feature_id", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
        )

        history = dynamodb.Table(
            self,
            "F1PredictionHistory",
            table_name=os.environ.get(
                "F1_PREDICTIONS_HISTORY_TABLE", "f1-predictions-history"
            ),
            partition_key=dynamodb.Attribute(
                name="pk", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="sk", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
        )

        lambda_role = iam.Role(
            self,
            "F1InferenceLambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                ),
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AWSXRayDaemonWriteAccess"
                ),
            ],
        )

        bucket.grant_read(lambda_role)
        features.grant_read_write_data(lambda_role)
        history.grant_read_write_data(lambda_role)

        fn = _lambda.Function(
            self,
            "F1InferenceFunction",
            runtime=_lambda.Runtime.PYTHON_3_11,
            code=_lambda.Code.from_asset("../lambda"),
            handler="inference.handler.lambda_handler",
            role=lambda_role,
            timeout=Duration.seconds(5),
            memory_size=256,
            tracing=_lambda.Tracing.ACTIVE,
            log_retention=logs.RetentionDays.ONE_WEEK,
            environment={
                "F1_MODELS_BUCKET": bucket.bucket_name,
                "F1_FEATURES_TABLE": features.table_name,
                "F1_PREDICTIONS_HISTORY_TABLE": history.table_name,
                "F1_LAPTIME_VERSION": "v2_2024_season",
                "F1_DEGRADATION_VERSION": "v1_degradation",
                "F1_STRATEGY_VERSION": "v1_strategy",
            },
        )

        api = apigw.RestApi(
            self,
            "F1InferenceApi",
            rest_api_name="F1 Race Strategy Inference",
            deploy_options=apigw.StageOptions(
                metrics_enabled=True,
                logging_level=apigw.MethodLoggingLevel.INFO,
                data_trace_enabled=False,
                throttling_burst_limit=200,
                throttling_rate_limit=150,
            ),
        )

        predict = api.root.add_resource("predict")
        for path in ["laptime", "pit-strategy", "tire-degradation"]:
            res = predict.add_resource(path)
            res.add_method("POST", apigw.LambdaIntegration(fn))

        latency_metric = cloudwatch.Metric(
            namespace="AWS/ApiGateway",
            metric_name="Latency",
            statistic="p95",
            period=Duration.minutes(1),
            dimensions_map={"ApiName": api.rest_api_name},
        )

        latency_alarm = cloudwatch.Alarm(
            self,
            "F1LatencyAlarm",
            metric=latency_metric,
            threshold=120,
            evaluation_periods=5,
            datapoints_to_alarm=3,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            alarm_description="p95 latency > 120ms for 5 minutes",
        )

        error_metric = cloudwatch.Metric(
            namespace="AWS/ApiGateway",
            metric_name="5XXError",
            period=Duration.minutes(1),
            statistic="Sum",
            dimensions_map={"ApiName": api.rest_api_name},
        )

        error_alarm = cloudwatch.Alarm(
            self,
            "F1ErrorAlarm",
            metric=error_metric,
            threshold=1,
            evaluation_periods=5,
            datapoints_to_alarm=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            alarm_description="API 5XX errors detected",
        )

        dashboard = cloudwatch.Dashboard(self, "F1RaceStrategyDashboard")
        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Lap Time Prediction Error by Track",
                left=[
                    cloudwatch.Metric(
                        namespace="F1Inference",
                        metric_name="prediction_error",
                        dimensions_map={"track": t},
                        statistic="Average",
                    )
                    for t in ["monaco", "silverstone", "spa"]
                ],
            ),
            cloudwatch.GraphWidget(
                title="Inference Latency p95 by Type",
                left=[
                    cloudwatch.Metric(
                        namespace="F1Inference",
                        metric_name="latency_ms_p95",
                        dimensions_map={"prediction_type": p},
                    )
                    for p in ["laptime", "strategy", "degradation"]
                ],
            ),
        )

        self.api_url_output = cdk.CfnOutput(
            self,
            "F1ApiUrl",
            value=api.url,
            description="Base URL for the F1 inference API",
        )
