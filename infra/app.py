#!/usr/bin/env python3
import os

import aws_cdk as cdk

from f1_inference_stack import F1InferenceStack


app = cdk.App()

F1InferenceStack(
    app,
    "F1InferenceStack",
    env=cdk.Environment(
        account=os.getenv("CDK_DEFAULT_ACCOUNT"),
        region=os.getenv("CDK_DEFAULT_REGION"),
    ),
)

app.synth()
