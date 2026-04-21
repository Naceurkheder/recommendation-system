"""
Lambda: ComputeTrigger
Sends a compute job to the SQS FIFO queue to trigger the worker.
Invoked as step 2 of the compute-pipeline Step Functions state machine.
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

AWS_ENDPOINT = os.getenv("AWS_ENDPOINT_URL", "http://localstack:4566")
AWS_REGION   = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
QUEUE_NAME   = "compute-jobs.fifo"
DYNAMO_TABLE = "compute-jobs"


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
    Enqueue a compute job for all users (triggered from Step Functions).
    Uses MessageGroupId='pipeline' and a timestamp-based dedup ID.
    """
    logger.info("compute_trigger invoked: %s", json.dumps(event))

    sqs      = _boto("sqs")
    dynamodb = _boto("dynamodb")

    job_id   = str(uuid.uuid4())
    now_iso  = datetime.now(timezone.utc).isoformat()
    hour_key = datetime.now(timezone.utc).strftime("%Y%m%d%H")

    # Write DynamoDB job record
    try:
        dynamodb.put_item(
            TableName=DYNAMO_TABLE,
            Item={
                "job_id":     {"S": job_id},
                "user_id":    {"S": "all"},
                "status":     {"S": "pending"},
                "created_at": {"S": now_iso},
                "updated_at": {"S": now_iso},
                "expires_at": {"N": str(int(datetime.now(timezone.utc).timestamp()) + 86400)},
            },
        )
    except Exception as exc:
        logger.warning("DynamoDB put failed: %s", exc)

    # Enqueue SQS FIFO message
    queue_url = sqs.get_queue_url(QueueName=QUEUE_NAME)["QueueUrl"]
    sqs.send_message(
        QueueUrl=queue_url,
        MessageBody=json.dumps({
            "job_id":       job_id,
            "user_id":      "all",
            "requested_at": now_iso,
            "source":       "step_functions",
        }),
        MessageGroupId="pipeline",
        MessageDeduplicationId=f"pipeline:{hour_key}",
    )

    logger.info("Enqueued compute job: %s", job_id)
    return {"status": "queued", "job_id": job_id}
