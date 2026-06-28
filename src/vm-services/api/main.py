
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RecSys API", version="2.0.0")

AWS_ENDPOINT = os.getenv("AWS_ENDPOINT_URL", "http://localstack:4566")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
QUEUE_NAME = "compute-jobs.fifo"
DYNAMO_TABLE = "compute-jobs"
LOG_GROUP = "/app/fastapi"
CACHE_KEY_PREFIX = "recs:"
CACHE_TTL = 86400

_sqs: Any = None
_dynamodb: Any = None
_logs: Any = None
_db_dsn: str = ""
_redis_conn: Any = None

_engine = None

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

    user_id: int
    product_id: str
    interaction_type: str
    metadata: Optional[Dict[str, Any]] = None

def _boto_client(service: str) -> Any:

    return boto3.client(
        service,
        endpoint_url=AWS_ENDPOINT,
        region_name=AWS_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
    )

def _fetch_secret(client: Any, secret_id: str, default: Optional[Dict] = None) -> Dict:

    try:
        raw = client.get_secret_value(SecretId=secret_id)
        return json.loads(raw["SecretString"])
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ResourceNotFoundException" and default is not None:
            try:
                client.create_secret(Name=secret_id, SecretString=json.dumps(default))
                logger.info("Auto-provisioned Secrets Manager secret: %s", secret_id)
            except Exception:
                pass
            return default
        raise

def _ensure_aws_resources() -> None:

    for group in ["/app/fastapi", "/app/worker", "/app/lambda"]:
        try:
            _logs.create_log_group(logGroupName=group)
            logger.info("Created log group: %s", group)
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "ResourceAlreadyExistsException":
                pass
        except Exception:
            pass

    try:
        _logs.create_log_stream(logGroupName=LOG_GROUP, logStreamName="api")
    except Exception:
        pass

    try:
        _dynamodb.describe_table(TableName=DYNAMO_TABLE)
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ResourceNotFoundException":
            try:
                _dynamodb.create_table(
                    TableName=DYNAMO_TABLE,
                    AttributeDefinitions=[
                        {"AttributeName": "job_id",  "AttributeType": "S"},
                        {"AttributeName": "user_id", "AttributeType": "S"},
                        {"AttributeName": "status",  "AttributeType": "S"},
                    ],
                    KeySchema=[{"AttributeName": "job_id", "KeyType": "HASH"}],
                    GlobalSecondaryIndexes=[
                        {
                            "IndexName": "user_id-index",
                            "KeySchema": [{"AttributeName": "user_id", "KeyType": "HASH"}],
                            "Projection": {"ProjectionType": "ALL"},
                        },
                        {
                            "IndexName": "status-index",
                            "KeySchema": [{"AttributeName": "status", "KeyType": "HASH"}],
                            "Projection": {"ProjectionType": "ALL"},
                        },
                    ],
                    BillingMode="PAY_PER_REQUEST",
                )
                logger.info("Created DynamoDB table: %s", DYNAMO_TABLE)
            except Exception as exc2:
                logger.warning("DynamoDB table creation failed: %s", exc2)
    except Exception:
        pass

    try:
        _sqs.get_queue_url(QueueName=QUEUE_NAME)
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "AWS.SimpleQueueService.NonExistentQueue":
            try:
                _sqs.create_queue(
                    QueueName="compute-jobs-dlq.fifo",
                    Attributes={"FifoQueue": "true", "ContentBasedDeduplication": "true"},
                )
                _sqs.create_queue(
                    QueueName=QUEUE_NAME,
                    Attributes={"FifoQueue": "true", "ContentBasedDeduplication": "true"},
                )
                logger.info("Created SQS FIFO queues")
            except Exception as exc2:
                logger.warning("SQS queue creation failed: %s", exc2)
    except Exception:
        pass

    try:
        _boto_client("s3").head_bucket(Bucket="similarity-matrices")
    except ClientError:
        try:
            _boto_client("s3").create_bucket(Bucket="similarity-matrices")
            logger.info("Created S3 bucket: similarity-matrices")
        except Exception as exc2:
            logger.warning("S3 bucket creation failed: %s", exc2)
    except Exception:
        pass

    try:
        sns = _boto_client("sns")
        resp = sns.create_topic(Name="compute-complete")
        logger.info("SNS topic ready: %s", resp.get("TopicArn", ""))
    except Exception as exc:
        logger.warning("SNS topic creation failed: %s", exc)

def _init_aws_and_db() -> None:

    global _sqs, _dynamodb, _logs, _db_dsn, _redis_conn

    _sqs      = _boto_client("sqs")
    _dynamodb = _boto_client("dynamodb")
    _logs     = _boto_client("logs")
    secrets   = _boto_client("secretsmanager")

    _pg_default = {
        "host": "db", "port": 5432, "database": "recsys_db",
        "username": "recsys_admin", "password": "secure_password",
    }
    _rd_default = {"host": "redis", "port": 6379, "password": ""}

    try:
        pg = _fetch_secret(secrets, "db/postgres", _pg_default)
        _db_dsn = (
            f"host={pg['host']} port={pg['port']} dbname={pg['database']} "
            f"user={pg['username']} password={pg['password']}"
        )
        logger.info("DB credentials loaded from Secrets Manager")
    except Exception as exc:
        _db_dsn = os.getenv("DATABASE_URL", "postgresql://recsys_admin:secure_password@db:5432/recsys_db")
        logger.warning("Secrets Manager unreachable (%s) — using DATABASE_URL env var", exc)

    try:
        rd = _fetch_secret(secrets, "redis/config", _rd_default)
        redis_host, redis_port = rd["host"], int(rd["port"])
    except Exception as exc:
        redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
        parts = redis_url.replace("redis://", "").split(":")
        redis_host = parts[0]
        redis_port = int(parts[1]) if len(parts) > 1 else 6379
        logger.warning("Redis secret unreachable (%s) — using REDIS_URL env var", exc)

    import redis as redis_module
    _redis_conn = redis_module.Redis(host=redis_host, port=redis_port, decode_responses=True)

    _ensure_aws_resources()

    logger.info("AWS clients and DB pool initialized")

def _log_to_cloudwatch(message: str) -> None:

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

    if _db_dsn.startswith("postgresql://") or _db_dsn.startswith("postgres://"):
        return psycopg2.connect(_db_dsn)
    return psycopg2.connect(_db_dsn)

def _generate_csv_from_db(out_path: str) -> bool:

    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        conn = _db_conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT user_id, product_id::text,
                CASE interaction_type
                    WHEN 'purchase' THEN 1.0
                    WHEN 'like'     THEN 0.7
                    WHEN 'view'     THEN 0.3
                    ELSE 0.1
                END
            FROM interactions ORDER BY user_id
            """
        )
        rows = cur.fetchall()
        conn.close()
        if not rows:
            logger.warning("C engine: interactions table is empty — cannot build matrix")
            return False
        with open(out_path, "w") as f:
            for uid, pid, rating in rows:
                f.write(f"{uid},{pid},{float(rating):.4f}\n")
        logger.info("C engine: generated %d-row matrix at %s", len(rows), out_path)
        return True
    except Exception as exc:
        logger.error("C engine: CSV generation from DB failed: %s", exc)
        return False

def init_engine() -> bool:

    global _engine
    try:
        lib_path = os.getenv("REC_ENGINE_LIB", "/usr/local/lib/librec_engine.so")
        wrapper = get_engine(library_path=lib_path)

        csv_path = os.getenv("REC_ENGINE_DATA", "/app/data/matrix.csv")
        for candidate in [csv_path, "./data/matrix.csv", "/app/data/matrix.csv"]:
            if os.path.exists(candidate):
                csv_path = candidate
                break

        if not os.path.exists(csv_path):
            logger.info("C engine: CSV missing at %s — generating from database", csv_path)
            if not _generate_csv_from_db(csv_path):
                logger.warning("C engine: CSV unavailable — will retry later")
                return False

        logger.info("C engine: loading matrix from %s", csv_path)
        success = wrapper.init(csv_path)
        if success:
            _engine = wrapper
            num_users, num_items = _engine.get_dimensions()
            logger.info("C engine initialized: %d users × %d items", num_users, num_items)
        else:
            logger.error("C engine: load_matrix() returned NULL for %s", csv_path)
        return success
    except Exception as exc:
        logger.error("C engine init failed: %s", exc)
        return False

async def _engine_retry_loop() -> None:

    import asyncio
    deadline = asyncio.get_event_loop().time() + 600
    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(30)
        if _engine is not None:
            return
        logger.info("C engine: retrying initialization…")
        if init_engine():
            logger.info("C engine: initialized via background retry")
            return
    logger.warning("C engine: gave up after 10 min — endpoints will return 503")

@app.on_event("startup")
async def startup_event():

    import asyncio
    _init_aws_and_db()
    if not init_engine():
        logger.warning("C engine not available at startup — retrying in background")
        asyncio.create_task(_engine_retry_loop())

@app.on_event("shutdown")
async def shutdown_event():
    global _engine
    if _engine:
        _engine.cleanup()
        logger.info("C engine cleaned up")

VALID_ENGINES = {"openmp", "mpi", "cuda"}
ACTIVE_ENGINE_KEY = "active_engine"

class EngineSelectRequest(BaseModel):
    engine: str

@app.get("/api/engine", tags=["System"])
def get_active_engine():

    engine = "openmp"
    if _redis_conn:
        try:
            val = _redis_conn.get(ACTIVE_ENGINE_KEY)
            if val and val in VALID_ENGINES:
                engine = val
        except Exception:
            pass
    return {"active": engine, "valid_engines": list(VALID_ENGINES)}

@app.put("/api/engine", tags=["System"])
def set_active_engine(body: EngineSelectRequest):

    if body.engine not in VALID_ENGINES:
        raise HTTPException(status_code=422, detail=f"engine must be one of {VALID_ENGINES}")
    if not _redis_conn:
        raise HTTPException(status_code=503, detail="Redis not available")
    try:
        _redis_conn.set(ACTIVE_ENGINE_KEY, body.engine)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    logger.info("Active engine set to: %s", body.engine)
    return {"active": body.engine, "status": "updated"}

@app.get("/health", tags=["System"])
def health_check():

    return {"status": "ok"}

@app.get("/", tags=["System"])
def read_root():
    return {"message": "Recommendation System API is running!", "version": "2.0.0", "docs": "/docs"}

@app.post("/recommendations/similar-users", tags=["Engine"])
def get_similar_users(request: SimilarUsersRequest) -> SimilarUsersResponse:

    if not _engine:
        raise HTTPException(status_code=503, detail="C engine not initialized")
    try:
        similar_users = _engine.get_similar_users(request.user_id, request.k)
        return SimilarUsersResponse(user_id=request.user_id, similar_users=similar_users, status="success")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.post("/recommendations/items", tags=["Engine"])
def get_item_recommendations(request: ItemRecommendationsRequest) -> ItemRecommendationsResponse:

    if not _engine:
        raise HTTPException(status_code=503, detail="C engine not initialized")
    try:
        recs = _engine.get_item_recommendations(request.user_id, request.k, request.num_neighbors)
        return ItemRecommendationsResponse(user_id=request.user_id, recommendations=recs, status="success")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.get("/recommendations/similarity", tags=["Engine"])
def get_similarity(
    user_a: int = Query(..., description="First user ID", example=1),
    user_b: int = Query(..., description="Second user ID", example=2),
) -> SimilarityScoreResponse:

    if not _engine:
        raise HTTPException(status_code=503, detail="C engine not initialized")
    try:
        score = _engine.get_similarity(user_a, user_b)
        return SimilarityScoreResponse(user_a=user_a, user_b=user_b, similarity=score, status="success")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.get("/recommendations/top-k", tags=["Engine"])
def get_top_k(
    user_id: int = Query(..., description="User ID", example=1),
    top_n: int = Query(10, ge=1, le=100, description="Number of recommendations"),
):

    return get_item_recommendations(ItemRecommendationsRequest(user_id=user_id, k=top_n))

@app.get("/recommendations/{user_id}", tags=["Recommendations"])
def get_recommendations_cached(user_id: int, force: bool = Query(False, description="Bypass Redis cache and always dispatch a fresh SQS job")):

    cache_key = f"{CACHE_KEY_PREFIX}{user_id}:latest"

    if _redis_conn and not force:
        try:
            cached = _redis_conn.get(cache_key)
            if cached:
                data = json.loads(cached)
                _log_to_cloudwatch(f"cache_hit user_id={user_id}")
                return {"user_id": user_id, **data, "cached": True}
        except Exception as exc:
            logger.warning("Redis read failed: %s", exc)
    elif force and _redis_conn:

        try:
            _redis_conn.delete(cache_key)
        except Exception as exc:
            logger.warning("Redis delete failed: %s", exc)

    job_id = str(uuid.uuid4())
    if force:
        dedup_id = f"{user_id}:{job_id}"
    else:
        dedup_window = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")[:-1]
        dedup_id = f"{user_id}:{dedup_window}"
    now_iso = datetime.now(timezone.utc).isoformat()
    expires_epoch = int((datetime.now(timezone.utc) + timedelta(hours=24)).timestamp())

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

@app.get("/recommendations/{user_id}/status", tags=["Recommendations"])
def get_recommendation_status(user_id: int):

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

@app.post("/interactions", status_code=201, tags=["Data"])
def create_interaction(interaction: InteractionCreate):

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

@app.get("/status", tags=["System"])
def get_status() -> dict:

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
