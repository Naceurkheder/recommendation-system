"""
Compute worker: polls SQS FIFO, exports PostgreSQL interactions, runs the C
similarity engine via ctypes, writes results to Redis + S3, updates DynamoDB,
publishes to SNS.
"""

import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError
import psycopg2
import psycopg2.extras
import redis as redis_module
import numpy as np

from ctypes_bridge import RecEngineBridge

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("worker")

# ── Config ────────────────────────────────────────────────────────────────────
AWS_ENDPOINT  = os.getenv("AWS_ENDPOINT_URL",     "http://localstack:4566")
AWS_REGION    = os.getenv("AWS_DEFAULT_REGION",    "us-east-1")
QUEUE_NAME    = "compute-jobs.fifo"
DYNAMO_TABLE  = "compute-jobs"
S3_BUCKET     = "similarity-matrices"
SNS_TOPIC_ARN = f"arn:aws:sns:{AWS_REGION}:000000000000:compute-complete"
LOG_GROUP     = "/app/worker"
CACHE_KEY_FMT = "recs:{user_id}:latest"
CACHE_TTL     = 86400        # 24 h
TOP_K         = 50
LONG_POLL_S   = 20

# Rating weights for implicit feedback
RATING_MAP: Dict[str, float] = {
    "purchase": 1.0,
    "like":     0.7,
    "view":     0.3,
}
DEFAULT_RATING = 0.1


# ── AWS / DB helpers ──────────────────────────────────────────────────────────

def _boto(service: str) -> Any:
    return boto3.client(
        service,
        endpoint_url=AWS_ENDPOINT,
        region_name=AWS_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
    )


def _fetch_secret(secrets_client: Any, secret_id: str) -> Dict:
    """Retry-loop for Secrets Manager (not provisioned until setup script runs)."""
    for attempt in range(20):
        try:
            raw = secrets_client.get_secret_value(SecretId=secret_id)
            return json.loads(raw["SecretString"])
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code in ("ResourceNotFoundException", "InvalidRequestException"):
                if attempt < 19:
                    logger.info("Secret %s not ready, retry %d/20", secret_id, attempt + 1)
                    time.sleep(3)
                    continue
            raise
    raise RuntimeError(f"Secret {secret_id} unavailable")


def _log_cw(logs_client: Any, message: str) -> None:
    """Best-effort CloudWatch emit."""
    try:
        logs_client.put_log_events(
            logGroupName=LOG_GROUP,
            logStreamName="worker",
            logEvents=[{"timestamp": int(time.time() * 1000), "message": message}],
        )
    except Exception:
        pass


# ── Startup: credentials + clients ───────────────────────────────────────────

def init_clients() -> Tuple[Any, Any, Any, Any, Any, Any, Any]:
    """
    Fetch credentials from Secrets Manager, open DB + Redis connections,
    return (sqs, dynamodb, s3, sns, logs, db_conn, redis_conn).
    """
    secrets = _boto("secretsmanager")
    sqs      = _boto("sqs")
    dynamodb = _boto("dynamodb")
    s3       = _boto("s3")
    sns      = _boto("sns")
    logs     = _boto("logs")

    # Ensure CloudWatch log stream exists
    try:
        logs.create_log_stream(logGroupName=LOG_GROUP, logStreamName="worker")
    except ClientError:
        pass

    # PostgreSQL
    try:
        pg = _fetch_secret(secrets, "db/postgres")
        dsn = (
            f"host={pg['host']} port={pg['port']} dbname={pg['database']} "
            f"user={pg['username']} password={pg['password']}"
        )
    except Exception as exc:
        fallback = os.getenv("DATABASE_URL", "postgresql://recsys_admin:secure_password@db:5432/recsys_db")
        dsn = fallback
        logger.warning("Secrets Manager unavailable (%s), using DATABASE_URL fallback", exc)

    db_conn = psycopg2.connect(dsn)
    db_conn.autocommit = True
    logger.info("PostgreSQL connected")

    # Redis
    try:
        rd = _fetch_secret(secrets, "redis/config")
        redis_host, redis_port = rd["host"], int(rd["port"])
    except Exception as exc:
        redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
        parts = redis_url.replace("redis://", "").split(":")
        redis_host = parts[0]
        redis_port = int(parts[1]) if len(parts) > 1 else 6379
        logger.warning("Redis secret unavailable (%s), using REDIS_URL fallback", exc)

    redis_conn = redis_module.Redis(host=redis_host, port=redis_port, decode_responses=True)
    logger.info("Redis connected at %s:%d", redis_host, redis_port)

    return sqs, dynamodb, s3, sns, logs, db_conn, redis_conn


# ── Core pipeline ─────────────────────────────────────────────────────────────

def export_interactions_to_csv(db_conn: Any, csv_path: str) -> int:
    """
    Query PostgreSQL interactions table, write CSV for the C engine.
    Format: user_id,product_id,rating (no header)
    Returns number of rows written.
    """
    with db_conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                user_id,
                product_id::text,
                CASE interaction_type
                    WHEN 'purchase' THEN 1.0
                    WHEN 'like'     THEN 0.7
                    WHEN 'view'     THEN 0.3
                    ELSE 0.1
                END AS rating
            FROM interactions
            ORDER BY user_id
            """
        )
        rows = cur.fetchall()

    if not rows:
        logger.warning("interactions table is empty — skipping compute")
        return 0

    with open(csv_path, "w") as fh:
        for user_id, product_id, rating in rows:
            fh.write(f"{user_id},{product_id},{rating:.4f}\n")

    logger.info("Exported %d interactions to %s", len(rows), csv_path)
    return len(rows)


def _update_dynamo_status(
    dynamodb: Any,
    job_id: str,
    status: str,
    extra: Optional[Dict] = None,
) -> None:
    """Update DynamoDB job record status."""
    now = datetime.now(timezone.utc).isoformat()
    update_expr = "SET #s = :s, updated_at = :u"
    expr_names  = {"#s": "status"}
    expr_values = {":s": {"S": status}, ":u": {"S": now}}

    if extra:
        for k, v in extra.items():
            update_expr += f", {k} = :{k}"
            expr_values[f":{k}"] = {"S": str(v)}

    try:
        dynamodb.update_item(
            TableName=DYNAMO_TABLE,
            Key={"job_id": {"S": job_id}},
            UpdateExpression=update_expr,
            ExpressionAttributeNames=expr_names,
            ExpressionAttributeValues=expr_values,
        )
    except Exception as exc:
        logger.warning("DynamoDB status update failed: %s", exc)


def process_message(
    body: Dict,
    sqs: Any,
    dynamodb: Any,
    s3: Any,
    sns: Any,
    logs: Any,
    db_conn: Any,
    redis_conn: Any,
    receipt_handle: str,
) -> bool:
    """
    Full compute pipeline for one SQS message.
    Returns True on success (message deleted); False on failure (message left for retry).
    """
    job_id  = body.get("job_id",  "unknown")
    user_id = body.get("user_id", "all")
    logger.info("Processing job_id=%s user_id=%s", job_id, user_id)
    _log_cw(logs, f"job_start job_id={job_id} user_id={user_id}")

    # 1. Mark running
    _update_dynamo_status(dynamodb, job_id, "running")

    try:
        # 2. Export interactions to temp CSV
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp:
            csv_path = tmp.name

        row_count = export_interactions_to_csv(db_conn, csv_path)
        if row_count == 0:
            logger.warning("No data to process for job %s", job_id)
            _update_dynamo_status(dynamodb, job_id, "complete", {"row_count": "0"})
            sqs.delete_message(
                QueueUrl=sqs.get_queue_url(QueueName=QUEUE_NAME)["QueueUrl"],
                ReceiptHandle=receipt_handle,
            )
            return True

        # 3. C ctypes bridge: load matrix, compute similarity, extract top-K
        bridge = RecEngineBridge()
        recommendations, similarity_matrix = bridge.compute_from_csv(csv_path, top_k=TOP_K)

        # 4. Upload full similarity matrix to S3
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        s3_key = f"matrices/{timestamp}.json"
        matrix_payload = {
            "timestamp": timestamp,
            "job_id":    job_id,
            "shape":     list(similarity_matrix.shape),
            "data":      similarity_matrix.tolist(),
        }
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=json.dumps(matrix_payload),
            ContentType="application/json",
        )
        logger.info("Uploaded similarity matrix to s3://%s/%s", S3_BUCKET, s3_key)

        # 5. Write per-user top-K to Redis with 24 h TTL
        pipe = redis_conn.pipeline()
        now_iso = datetime.now(timezone.utc).isoformat()
        for uid, recs in recommendations.items():
            cache_key = CACHE_KEY_FMT.format(user_id=uid)
            payload = json.dumps({"similar_users": recs, "computed_at": now_iso})
            pipe.setex(cache_key, CACHE_TTL, payload)
        pipe.execute()
        logger.info("Wrote Redis cache for %d users", len(recommendations))

        # 6. Update DynamoDB to complete
        _update_dynamo_status(dynamodb, job_id, "complete", {
            "s3_key":    s3_key,
            "row_count": str(row_count),
        })

        # 7. Publish to SNS
        try:
            sns.publish(
                TopicArn=SNS_TOPIC_ARN,
                Message=json.dumps({
                    "job_id":    job_id,
                    "user_id":   str(user_id),
                    "status":    "complete",
                    "s3_key":    s3_key,
                    "timestamp": now_iso,
                }),
                Subject="compute-complete",
            )
        except Exception as exc:
            logger.warning("SNS publish failed (non-fatal): %s", exc)

        # 8. Delete SQS message on success
        queue_url = sqs.get_queue_url(QueueName=QUEUE_NAME)["QueueUrl"]
        sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)

        _log_cw(logs, f"job_complete job_id={job_id} users={len(recommendations)} s3={s3_key}")
        logger.info("Job %s complete. Processed %d users.", job_id, len(recommendations))
        return True

    except Exception as exc:
        logger.error("Job %s failed: %s", job_id, exc, exc_info=True)
        _update_dynamo_status(dynamodb, job_id, "failed")
        _log_cw(logs, f"job_failed job_id={job_id} error={exc}")
        # Do NOT delete message — let SQS retry → DLQ after maxReceiveCount=3
        return False

    finally:
        try:
            os.unlink(csv_path)
        except Exception:
            pass


# ── Main polling loop ─────────────────────────────────────────────────────────

def main() -> None:
    logger.info("Worker starting up...")

    sqs, dynamodb, s3, sns, logs, db_conn, redis_conn = init_clients()

    queue_url = sqs.get_queue_url(QueueName=QUEUE_NAME)["QueueUrl"]
    logger.info("Polling SQS queue: %s", queue_url)

    while True:
        try:
            resp = sqs.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=LONG_POLL_S,
                AttributeNames=["All"],
                MessageAttributeNames=["All"],
            )
            messages = resp.get("Messages", [])

            if not messages:
                continue

            msg = messages[0]
            receipt = msg["ReceiptHandle"]
            try:
                body = json.loads(msg["Body"])
            except json.JSONDecodeError:
                logger.error("Malformed SQS message body: %s", msg["Body"])
                sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt)
                continue

            process_message(
                body, sqs, dynamodb, s3, sns, logs, db_conn, redis_conn, receipt
            )

        except Exception as exc:
            logger.error("Poll loop error: %s", exc, exc_info=True)
            time.sleep(5)


if __name__ == "__main__":
    main()
