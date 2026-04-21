"""
Lambda: Notifier
Publishes a compute-complete notification to the SNS topic.
Final step of the compute-pipeline Step Functions state machine.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

AWS_ENDPOINT  = os.getenv("AWS_ENDPOINT_URL", "http://localstack:4566")
AWS_REGION    = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
ACCOUNT_ID    = "000000000000"
SNS_TOPIC_ARN = f"arn:aws:sns:{AWS_REGION}:{ACCOUNT_ID}:compute-complete"


def _boto(service: str):
    return boto3.client(
        service,
        endpoint_url=AWS_ENDPOINT,
        region_name=AWS_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
    )


def lambda_handler(event: dict, context) -> dict:
    """
    Publish a pipeline-complete notification to SNS compute-complete topic.
    Accepts the aggregated output from the ParallelFinalize Step Functions branch.
    """
    logger.info("notifier invoked: %s", json.dumps(event))

    sns  = _boto("sns")
    logs = _boto("logs")
    now  = datetime.now(timezone.utc).isoformat()

    # event may be a list (parallel branch results) or a dict
    payload: dict
    if isinstance(event, list):
        # Merge parallel branch outputs
        payload = {"branches": event, "timestamp": now, "source": "step_functions"}
    else:
        payload = {**event, "timestamp": now, "source": "step_functions"}

    payload.setdefault("status", "complete")

    sns.publish(
        TopicArn=SNS_TOPIC_ARN,
        Message=json.dumps(payload),
        Subject="compute-pipeline-complete",
        MessageAttributes={
            "status": {"DataType": "String", "StringValue": payload["status"]},
        },
    )

    try:
        logs.put_log_events(
            logGroupName="/app/lambda",
            logStreamName="notifier",
            logEvents=[{
                "timestamp": int(time.time() * 1000),
                "message": f"published to {SNS_TOPIC_ARN} status={payload['status']}",
            }],
        )
    except Exception:
        pass

    logger.info("Published compute-complete notification")
    return {"status": "published", "topic": SNS_TOPIC_ARN, "timestamp": now}
