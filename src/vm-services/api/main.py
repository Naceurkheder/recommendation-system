"""
High-Performance Recommendation Engine API
Exposes optimized C implementation via REST endpoints, with AWS-backed async pipeline.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import logging
import json
import uuid
import time
from datetime import datetime, timezone, timedelta

import boto3
from botocore.exceptions import ClientError
import psycopg2
import psycopg2.extras

from rec_engine_wrapper import get_engine

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RecSys API",
    version="2.0.0",
    description="High-Performance Recommendation Engine - OpenMP/MPI/CUDA + AWS"
)

# ── Config ────────────────────────────────────────────────────────────────────
AWS_ENDPOINT = os.getenv("AWS_ENDPOINT_URL", "http://localstack:4566")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
QUEUE_NAME = "compute-jobs.fifo"
DYNAMO_TABLE = "compute-jobs"
LOG_GROUP = "/app/fastapi"
CACHE_KEY_PREFIX = "recs:"
CACHE_TTL = 86400  # 24 h

# ── AWS / DB globals ──────────────────────────────────────────────────────────
_sqs: Any = None
_dynamodb: Any = None
_logs: Any = None
_db_dsn: str = ""
_redis_conn: Any = None

# ── C engine global ───────────────────────────────────────────────────────────
_engine = None


# ── Request / Response models ─────────────────────────────────────────────────
class SimilarUsersRequest(BaseModel):
    user_id: int
    k: int = 10


class SimilarUsersResponse(BaseModel):
    user_id: int
    similar_users: List[int]
    status: str


class ItemRecommendationsRequest(BaseModel):
    user_id: int
    k: int = 10
    num_neighbors: int = 10


class ItemRecommendationsResponse(BaseModel):
    user_id: int
    recommendations: List[int]
    status: str


class SimilarityScoreResponse(BaseModel):
    user_a: int
    user_b: int
    similarity: float
    status: str


class EngineStatusResponse(BaseModel):
    status: str
    num_users: int
    num_items: int
    initialized: bool


class InteractionCreate(BaseModel):
    """Payload for POST /interactions."""
    user_id: int
    product_id: str
    interaction_type: str
    metadata: Optional[Dict[str, Any]] = None


# ── AWS helpers ───────────────────────────────────────────────────────────────

def _boto_client(service: str) -> Any:
    """Return a boto3 client pointing at LocalStack."""
    return boto3.client(
        service,
        endpoint_url=AWS_ENDPOINT,
        region_name=AWS_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
    )


def _fetch_secret(client: Any, secret_id: str, retries: int = 10, delay: float = 3.0) -> Dict:
    """Fetch and parse a Secrets Manager secret, retrying until provisioned."""
    for attempt in range(retries):
        try:
            raw = client.get_secret_value(SecretId=secret_id)
            return json.loads(raw["SecretString"])
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code in ("ResourceNotFoundException", "InvalidRequestException"):
                if attempt < retries - 1:
                    logger.info("Secret %s not ready, retry %d/%d", secret_id, attempt + 1, retries)
                    time.sleep(delay)
                    continue
            raise
    raise RuntimeError(f"Secret {secret_id} unavailable after {retries} attempts")


def _init_aws_and_db() -> None:
    """Fetch credentials from Secrets Manager, init connection pools and AWS clients."""
    global _sqs, _dynamodb, _logs, _db_dsn, _redis_conn

    secrets_client = _boto_client("secretsmanager")
    _sqs = _boto_client("sqs")
    _dynamodb = _boto_client("dynamodb")
    _logs = _boto_client("logs")

    # Fetch DB credentials
    try:
        pg = _fetch_secret(secrets_client, "db/postgres")
        _db_dsn = (
            f"host={pg['host']} port={pg['port']} dbname={pg['database']} "
            f"user={pg['username']} password={pg['password']}"
        )
        logger.info("Fetched db/postgres secret from Secrets Manager")
    except Exception as exc:
        fallback = os.getenv("DATABASE_URL", "postgresql://recsys_admin:secure_password@db:5432/recsys_db")
        _db_dsn = fallback.replace("postgresql://", "").replace("postgres://", "")
        # psycopg2 accepts DSN keyword format; store as URL for later parsing
        _db_dsn = fallback
        logger.warning("Secrets Manager unavailable (%s), falling back to DATABASE_URL", exc)

    # Fetch Redis credentials
    try:
        rd = _fetch_secret(secrets_client, "redis/config")
        redis_host = rd["host"]
        redis_port = int(rd["port"])
    except Exception as exc:
        redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
        parts = redis_url.replace("redis://", "").split(":")
        redis_host = parts[0]
        redis_port = int(parts[1]) if len(parts) > 1 else 6379
        logger.warning("Redis secret unavailable (%s), falling back to REDIS_URL", exc)

    import redis as redis_module
    _redis_conn = redis_module.Redis(host=redis_host, port=redis_port, decode_responses=True)

    # Ensure CloudWatch log stream exists
    try:
        _logs.create_log_stream(
            logGroupName=LOG_GROUP,
            logStreamName="api"
        )
    except ClientError:
        pass  # already exists

    logger.info("AWS clients and connection pools initialized")


def _log_to_cloudwatch(message: str) -> None:
    """Best-effort CloudWatch log emit."""
    if _logs is None:
        return
    try:
        _logs.put_log_events(
            logGroupName=LOG_GROUP,
            logStreamName="api",
            logEvents=[{"timestamp": int(time.time() * 1000), "message": message}],
        )
    except Exception:
        pass


def _db_conn():
    """Open a fresh PostgreSQL connection using stored DSN."""
    if _db_dsn.startswith("postgresql://") or _db_dsn.startswith("postgres://"):
        return psycopg2.connect(_db_dsn)
    return psycopg2.connect(_db_dsn)


# ── C-engine init ─────────────────────────────────────────────────────────────

def init_engine():
    """Initialize the recommendation engine (C shared library)."""
    global _engine
    try:
        lib_path = os.getenv("REC_ENGINE_LIB", "/usr/local/lib/librec_engine.so")
        _engine = get_engine(library_path=lib_path)

        csv_path = os.getenv("REC_ENGINE_DATA", "/vagrant/src/host-cuda/openmp/data/matrix.csv")
        for alt in ["./data/matrix.csv", "../data/matrix.csv"]:
            if os.path.exists(alt):
                csv_path = alt
                break

        if not os.path.exists(csv_path):
            logger.error("CSV file not found: %s", csv_path)
            return False

        success = _engine.init(csv_path)
        if success:
            num_users, num_items = _engine.get_dimensions()
            logger.info("C engine initialized: %d users, %d items", num_users, num_items)
        return success
    except Exception as exc:
        logger.error("C engine init failed: %s", exc)
        return False


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Init AWS clients (Secrets Manager → connection pools) then C engine."""
    _init_aws_and_db()
    if not init_engine():
        logger.warning("C engine not available — sync endpoints will return 503")


@app.on_event("shutdown")
async def shutdown_event():
    global _engine
    if _engine:
        _engine.cleanup()
        logger.info("C engine cleaned up")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Liveness probe — returns 200 as long as the process is up."""
    return {"status": "ok"}


# ── Existing synchronous C-engine endpoints ───────────────────────────────────

@app.get("/")
def read_root():
    return {"message": "Recommendation System API is running!", "version": "2.0.0"}


@app.post("/recommendations/similar-users")
def get_similar_users(request: SimilarUsersRequest) -> SimilarUsersResponse:
    """Get k most similar users (synchronous, C engine)."""
    if not _engine:
        raise HTTPException(status_code=503, detail="C engine not initialized")
    try:
        similar_users = _engine.get_similar_users(request.user_id, request.k)
        return SimilarUsersResponse(user_id=request.user_id, similar_users=similar_users, status="success")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/recommendations/items")
def get_item_recommendations(request: ItemRecommendationsRequest) -> ItemRecommendationsResponse:
    """Get top-k item recommendations (synchronous, C engine)."""
    if not _engine:
        raise HTTPException(status_code=503, detail="C engine not initialized")
    try:
        recs = _engine.get_item_recommendations(request.user_id, request.k, request.num_neighbors)
        return ItemRecommendationsResponse(user_id=request.user_id, recommendations=recs, status="success")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/recommendations/similarity")
def get_similarity(
    user_a: int = Query(..., description="First user ID"),
    user_b: int = Query(..., description="Second user ID"),
) -> SimilarityScoreResponse:
    """Get cosine similarity between two users (synchronous, C engine)."""
    if not _engine:
        raise HTTPException(status_code=503, detail="C engine not initialized")
    try:
        score = _engine.get_similarity(user_a, user_b)
        return SimilarityScoreResponse(user_a=user_a, user_b=user_b, similarity=score, status="success")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/recommendations/top-k")
def get_top_k(
    user_id: int = Query(..., description="User ID"),
    top_n: int = Query(10, ge=1, le=100, description="Number of recommendations"),
):
    """Quick top-n item recommendations (synchronous, C engine)."""
    return get_item_recommendations(ItemRecommendationsRequest(user_id=user_id, k=top_n))


# ── Async AWS-backed endpoints ────────────────────────────────────────────────

@app.get("/recommendations/{user_id}")
def get_recommendations_cached(user_id: int):
    """
    Cache-first recommendation fetch.
    Hit  → 200 with cached similar-user list from Redis.
    Miss → 202, enqueue SQS FIFO job (MessageGroupId=user_id, hourly dedup).
    """
    cache_key = f"{CACHE_KEY_PREFIX}{user_id}:latest"

    # Redis cache check
    if _redis_conn:
        try:
            cached = _redis_conn.get(cache_key)
            if cached:
                data = json.loads(cached)
                _log_to_cloudwatch(f"cache_hit user_id={user_id}")
                return {"user_id": user_id, **data, "cached": True}
        except Exception as exc:
            logger.warning("Redis read failed: %s", exc)

    # Cache miss → dispatch async compute job
    job_id = str(uuid.uuid4())
    current_hour = datetime.now(timezone.utc).strftime("%Y%m%d%H")
    dedup_id = f"{user_id}:{current_hour}"
    now_iso = datetime.now(timezone.utc).isoformat()
    expires_epoch = int((datetime.now(timezone.utc) + timedelta(hours=24)).timestamp())

    # Write job record to DynamoDB
    if _dynamodb:
        try:
            _dynamodb.put_item(
                TableName=DYNAMO_TABLE,
                Item={
                    "job_id":     {"S": job_id},
                    "user_id":    {"S": str(user_id)},
                    "status":     {"S": "pending"},
                    "created_at": {"S": now_iso},
                    "updated_at": {"S": now_iso},
                    "expires_at": {"N": str(expires_epoch)},
                },
            )
        except Exception as exc:
            logger.warning("DynamoDB put failed: %s", exc)

    # Enqueue SQS FIFO job
    if _sqs:
        try:
            queue_url = _sqs.get_queue_url(QueueName=QUEUE_NAME)["QueueUrl"]
            _sqs.send_message(
                QueueUrl=queue_url,
                MessageBody=json.dumps({"job_id": job_id, "user_id": user_id, "requested_at": now_iso}),
                MessageGroupId=str(user_id),
                MessageDeduplicationId=dedup_id,
            )
        except Exception as exc:
            logger.error("SQS enqueue failed: %s", exc)

    _log_to_cloudwatch(f"compute_dispatched user_id={user_id} job_id={job_id}")
    return JSONResponse(
        status_code=202,
        content={"status": "computing", "user_id": user_id, "job_id": job_id},
    )


@app.get("/recommendations/{user_id}/status")
def get_recommendation_status(user_id: int):
    """Query DynamoDB compute-jobs for the latest job status for this user."""
    if not _dynamodb:
        raise HTTPException(status_code=503, detail="DynamoDB not available")
    try:
        resp = _dynamodb.query(
            TableName=DYNAMO_TABLE,
            IndexName="user_id-index",
            KeyConditionExpression="user_id = :uid",
            ExpressionAttributeValues={":uid": {"S": str(user_id)}},
            ScanIndexForward=False,
            Limit=1,
        )
        items = resp.get("Items", [])
        if not items:
            raise HTTPException(status_code=404, detail="No compute job found for user")
        item = items[0]
        return {
            "job_id":     item["job_id"]["S"],
            "user_id":    item["user_id"]["S"],
            "status":     item["status"]["S"],
            "created_at": item.get("created_at", {}).get("S", ""),
            "updated_at": item.get("updated_at", {}).get("S", ""),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/interactions", status_code=201)
def create_interaction(interaction: InteractionCreate):
    """Insert a user-product interaction into PostgreSQL."""
    try:
        conn = _db_conn()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO interactions (user_id, product_id, interaction_type, metadata)
                    VALUES (%s, %s::uuid, %s, %s)
                    RETURNING interaction_id
                    """,
                    (
                        interaction.user_id,
                        interaction.product_id,
                        interaction.interaction_type,
                        json.dumps(interaction.metadata) if interaction.metadata else None,
                    ),
                )
                row = cur.fetchone()
        conn.close()
        _log_to_cloudwatch(f"interaction user_id={interaction.user_id} type={interaction.interaction_type}")
        return {"interaction_id": str(row[0]), "status": "created"}
    except Exception as exc:
        logger.error("Interaction insert failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/status")
def get_status() -> dict:
    """C engine diagnostics."""
    if not _engine:
        return {"status": "not_initialized"}
    try:
        num_users, num_items = _engine.get_dimensions()
        return {
            "status": "initialized",
            "num_users": num_users,
            "num_items": num_items,
            "features": [
                "OpenMP-optimized similarity computation",
                "Cosine similarity metrics",
                "Collaborative filtering recommendations",
                "Top-k efficient selection",
                "Redis-cached async pipeline",
                "SQS FIFO job dispatch",
            ],
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
