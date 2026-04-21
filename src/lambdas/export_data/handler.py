"""
Lambda: ExportData
Queries PostgreSQL interactions and writes a CSV to S3 for downstream processing.
Invoked as step 1 of the compute-pipeline Step Functions state machine.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone

import boto3
import psycopg2

logger = logging.getLogger()
logger.setLevel(logging.INFO)

AWS_ENDPOINT = os.getenv("AWS_ENDPOINT_URL", "http://localstack:4566")
AWS_REGION   = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
S3_BUCKET    = "similarity-matrices"


def _boto(service: str):
    return boto3.client(
        service,
        endpoint_url=AWS_ENDPOINT,
        region_name=AWS_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
    )


def _get_db_dsn() -> str:
    secrets = _boto("secretsmanager")
    try:
        raw = secrets.get_secret_value(SecretId="db/postgres")
        pg = json.loads(raw["SecretString"])
        return (
            f"host={pg['host']} port={pg['port']} dbname={pg['database']} "
            f"user={pg['username']} password={pg['password']}"
        )
    except Exception:
        return os.getenv("DATABASE_URL", "postgresql://recsys_admin:secure_password@db:5432/recsys_db")


def lambda_handler(event: dict, context) -> dict:
    """
    Export interactions table to S3 as CSV.
    Returns S3 key and row count for downstream steps.
    """
    logger.info("export_data invoked: %s", json.dumps(event))

    dsn = _get_db_dsn()
    conn = psycopg2.connect(dsn)
    s3 = _boto("s3")
    logs = _boto("logs")

    try:
        with conn.cursor() as cur:
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
    finally:
        conn.close()

    if not rows:
        return {"status": "no_data", "row_count": 0}

    csv_lines = "\n".join(f"{r[0]},{r[1]},{r[2]:.4f}" for r in rows)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    s3_key = f"exports/{timestamp}.csv"

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=s3_key,
        Body=csv_lines.encode(),
        ContentType="text/csv",
    )

    try:
        logs.put_log_events(
            logGroupName="/app/lambda",
            logStreamName="export-data",
            logEvents=[{
                "timestamp": int(time.time() * 1000),
                "message": f"exported {len(rows)} rows to s3://{S3_BUCKET}/{s3_key}",
            }],
        )
    except Exception:
        pass

    logger.info("Exported %d rows → s3://%s/%s", len(rows), S3_BUCKET, s3_key)
    return {"status": "ok", "s3_key": s3_key, "row_count": len(rows)}
