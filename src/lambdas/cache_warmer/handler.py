"""
Lambda: CacheWarmer
Reads the latest similarity matrix from S3 and pre-populates Redis per-user cache.
Runs in parallel branch of the Step Functions state machine.
"""

import json
import logging
import os
import time

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

AWS_ENDPOINT = os.getenv("AWS_ENDPOINT_URL", "http://localstack:4566")
AWS_REGION   = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
S3_BUCKET    = "similarity-matrices"
CACHE_TTL    = 86400  # 24 h


def _boto(service: str):
    return boto3.client(
        service,
        endpoint_url=AWS_ENDPOINT,
        region_name=AWS_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
    )


def _get_redis():
    """Open a Redis connection using Secrets Manager or env fallback."""
    import redis as redis_module

    secrets = _boto("secretsmanager")
    try:
        raw = secrets.get_secret_value(SecretId="redis/config")
        rd = json.loads(raw["SecretString"])
        host, port = rd["host"], int(rd["port"])
    except Exception:
        redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
        parts = redis_url.replace("redis://", "").split(":")
        host = parts[0]
        port = int(parts[1]) if len(parts) > 1 else 6379

    return redis_module.Redis(host=host, port=port, decode_responses=True)


def lambda_handler(event: dict, context) -> dict:
    """
    List the latest matrix file in S3, parse per-user recommendations,
    and write recs:{user_id}:latest keys to Redis.
    """
    logger.info("cache_warmer invoked: %s", json.dumps(event))

    s3   = _boto("s3")
    logs = _boto("logs")

    # Find the most-recently modified matrix file
    resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix="matrices/")
    objects = resp.get("Contents", [])
    if not objects:
        logger.warning("No matrix files in s3://%s/matrices/", S3_BUCKET)
        return {"status": "no_data"}

    latest = max(objects, key=lambda o: o["LastModified"])
    s3_key = latest["Key"]
    logger.info("Loading matrix from s3://%s/%s", S3_BUCKET, s3_key)

    raw = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)["Body"].read()
    matrix_data = json.loads(raw)

    # The worker stores per-user recs directly in Redis; this lambda re-warms from S3
    # matrix_data has shape info but not per-user recs — we use the raw similarity rows
    sim_data    = matrix_data.get("data", [])
    timestamp   = matrix_data.get("timestamp", "")
    num_users   = len(sim_data)

    if num_users == 0:
        return {"status": "empty_matrix"}

    redis_conn = _get_redis()
    pipe = redis_conn.pipeline()

    for uid in range(num_users):
        row = sim_data[uid]
        # Build sorted similar-user list (exclude self, top 50)
        indexed = [(v, i) for i, v in enumerate(row) if i != uid]
        indexed.sort(reverse=True)
        similar = [{"user_id": i, "score": round(v, 4)} for v, i in indexed[:50]]
        payload = json.dumps({"similar_users": similar, "computed_at": timestamp})
        pipe.setex(f"recs:{uid}:latest", CACHE_TTL, payload)

    pipe.execute()

    try:
        logs.put_log_events(
            logGroupName="/app/lambda",
            logStreamName="cache-warmer",
            logEvents=[{
                "timestamp": int(time.time() * 1000),
                "message": f"warmed cache for {num_users} users from {s3_key}",
            }],
        )
    except Exception:
        pass

    logger.info("Cache warmed for %d users from %s", num_users, s3_key)
    return {"status": "ok", "users_warmed": num_users, "source": s3_key}
