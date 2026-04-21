"""
Lambda: StatusUpdater
Updates a DynamoDB compute-jobs record status.
Runs in the parallel finalize branch of the Step Functions state machine,
and is also used by the PipelineFailed catch state.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

AWS_ENDPOINT = os.getenv("AWS_ENDPOINT_URL", "http://localstack:4566")
AWS_REGION   = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
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
    Update job status in DynamoDB.
    Expected event keys: job_id (str), status (str, optional — defaults to "complete").
    """
    logger.info("status_updater invoked: %s", json.dumps(event))

    job_id = event.get("job_id")
    status = event.get("status", "complete")
    error  = event.get("error", "")

    if not job_id:
        logger.warning("No job_id in event; skipping DynamoDB update")
        return {"status": "skipped", "reason": "no job_id"}

    dynamodb = _boto("dynamodb")
    logs     = _boto("logs")
    now_iso  = datetime.now(timezone.utc).isoformat()

    expr        = "SET #s = :s, updated_at = :u"
    expr_names  = {"#s": "status"}
    expr_values = {":s": {"S": status}, ":u": {"S": now_iso}}

    if error:
        expr += ", error_detail = :e"
        expr_values[":e"] = {"S": str(error)}

    dynamodb.update_item(
        TableName=DYNAMO_TABLE,
        Key={"job_id": {"S": job_id}},
        UpdateExpression=expr,
        ExpressionAttributeNames=expr_names,
        ExpressionAttributeValues=expr_values,
    )

    try:
        logs.put_log_events(
            logGroupName="/app/lambda",
            logStreamName="status-updater",
            logEvents=[{
                "timestamp": int(time.time() * 1000),
                "message": f"job_id={job_id} status={status}",
            }],
        )
    except Exception:
        pass

    logger.info("Updated job %s → %s", job_id, status)
    return {"status": "ok", "job_id": job_id, "new_status": status}
