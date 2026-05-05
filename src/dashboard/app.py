
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import boto3
import httpx
import psycopg2
import psycopg2.extras
import redis as redis_lib
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger("dashboard")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

AWS_ENDPOINT = os.getenv("AWS_ENDPOINT_URL", "http://localstack:4566")
AWS_REGION   = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
REDIS_URL    = os.getenv("REDIS_URL", "redis://redis:6379")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://recsys_admin:secure_password@db:5432/recsys_db")
API_URL      = os.getenv("API_URL", "http://api:8000")

_docker = None
try:
    import docker as _docker_mod
    _docker = _docker_mod.from_env()
    logger.info("Docker SDK connected")
except Exception as _e:
    logger.warning("Docker SDK unavailable: %s", _e)

app = FastAPI(title="RecSys Platform", docs_url=None, redoc_url=None)

from collections import deque
_NOTIF_BUFFER: deque = deque(maxlen=200)
_NOTIF_LOCK = threading.Lock()

_METRICS: Dict[str, Any] = {
    "total_processed": 0,
    "peak_main_depth": 0,
    "peak_inflight": 0,
    "last_processed_at": None,
    "throughput_timestamps": deque(maxlen=500),
    "started_at": datetime.now(timezone.utc).isoformat(),
}
_METRICS_LOCK = threading.Lock()

def _notification_poller() -> None:

    sqs = boto3.client(
        "sqs", endpoint_url=AWS_ENDPOINT, region_name=AWS_REGION,
        aws_access_key_id="test", aws_secret_access_key="test",
    )
    while True:
        try:
            qurl = sqs.get_queue_url(QueueName="compute-notifications")["QueueUrl"]
            for _ in range(3):
                msgs = sqs.receive_message(
                    QueueUrl=qurl, MaxNumberOfMessages=10,
                    VisibilityTimeout=30, WaitTimeSeconds=0,
                ).get("Messages", [])
                if not msgs:
                    break
                for m in msgs:
                    body = m.get("Body", "")
                    try:
                        parsed = json.loads(body)
                        if isinstance(parsed, dict) and "Message" in parsed and "TopicArn" in parsed:
                            parsed = json.loads(parsed["Message"])
                    except Exception:
                        parsed = {"raw": body}
                    now = datetime.now(timezone.utc)
                    parsed["_seen_at"] = now.isoformat()
                    with _NOTIF_LOCK:
                        _NOTIF_BUFFER.appendleft(parsed)
                    with _METRICS_LOCK:
                        _METRICS["total_processed"] += 1
                        _METRICS["last_processed_at"] = parsed["_seen_at"]
                        _METRICS["throughput_timestamps"].append(now.timestamp())
                    try:
                        sqs.delete_message(QueueUrl=qurl, ReceiptHandle=m["ReceiptHandle"])
                    except Exception:
                        pass
        except Exception as exc:
            logger.debug("notification poller iteration failed: %s", exc)
        time.sleep(2)

threading.Thread(target=_notification_poller, daemon=True, name="notif-poller").start()

def _depth_sampler() -> None:

    sqs = boto3.client(
        "sqs", endpoint_url=AWS_ENDPOINT, region_name=AWS_REGION,
        aws_access_key_id="test", aws_secret_access_key="test",
    )
    while True:
        try:
            qurl = sqs.get_queue_url(QueueName="compute-jobs.fifo")["QueueUrl"]
            attr = sqs.get_queue_attributes(
                QueueUrl=qurl,
                AttributeNames=["ApproximateNumberOfMessages", "ApproximateNumberOfMessagesNotVisible"],
            )["Attributes"]
            depth = int(attr.get("ApproximateNumberOfMessages", 0))
            inflight = int(attr.get("ApproximateNumberOfMessagesNotVisible", 0))
            with _METRICS_LOCK:
                if depth > _METRICS["peak_main_depth"]:
                    _METRICS["peak_main_depth"] = depth
                if inflight > _METRICS["peak_inflight"]:
                    _METRICS["peak_inflight"] = inflight
        except Exception as exc:
            logger.debug("depth sampler iteration failed: %s", exc)
        time.sleep(1)

threading.Thread(target=_depth_sampler, daemon=True, name="depth-sampler").start()
logger.info("Notification poller + depth sampler threads started")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def _boto(service: str):
    return boto3.client(
        service,
        region_name=AWS_REGION,
        endpoint_url=AWS_ENDPOINT,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
    )

def _redis():
    host = "redis"
    if "redis://" in REDIS_URL:
        host = REDIS_URL.split("://")[1].split(":")[0]
    return redis_lib.Redis(host=host, port=6379, decode_responses=True, socket_timeout=2)

def _pg():
    return psycopg2.connect(DATABASE_URL, connect_timeout=3)

def _ok(data):
    return JSONResponse(content=data)

class InteractPayload(BaseModel):
    user_id: int
    product_id: str
    interaction_type: str

class EnginePayload(BaseModel):
    engine: str

class HpcPayload(BaseModel):
    dataset_size_gb: float = 10
    problem_type: str = "machine_learning"
    node_count: int = 1
    precision: str = "fp32"
    iterations: int = 100
    memory_bound: bool = False
    latency_sensitive: bool = False
    selected_engines: List[str] = ["openmp", "cuda", "mpi"]

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(_HTML)

@app.get("/api/status")
async def api_status():
    result: Dict[str, Any] = {}

    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{API_URL}/health")
        result["api"] = "ok" if r.status_code == 200 else f"http_{r.status_code}"
    except Exception as e:
        result["api"] = f"error: {e}"

    try:
        conn = _pg()
        conn.close()
        result["db"] = "ok"
    except Exception as e:
        result["db"] = f"error: {e}"

    try:
        r = _redis()
        r.ping()
        result["redis"] = "ok"
    except Exception as e:
        result["redis"] = f"error: {e}"

    try:
        s3 = _boto("s3")
        s3.list_buckets()
        result["localstack"] = "ok"
    except Exception as e:
        result["localstack"] = f"error: {e}"

    containers = {}
    if _docker:
        try:
            for name in ["docker-api-1", "docker-worker-1", "docker-localstack-1",
                         "docker-db-1", "docker-redis-1", "docker-dashboard-1"]:
                try:
                    c = _docker.containers.get(name)
                    containers[name] = c.status
                except Exception:
                    containers[name] = "not_found"
        except Exception as e:
            containers["error"] = str(e)
    result["containers"] = containers
    return _ok(result)

@app.get("/api/metrics")
async def api_metrics():
    metrics: Dict[str, Any] = {}

    try:
        conn = _pg()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM users")
        metrics["users"] = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM products")
        metrics["products"] = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM interactions")
        metrics["interactions"] = cur.fetchone()[0]
        conn.close()
    except Exception as e:
        metrics["db_error"] = str(e)

    try:
        r = _redis()
        info = r.info("memory")
        metrics["redis_memory_mb"] = round(info.get("used_memory", 0) / 1024 / 1024, 2)
        metrics["redis_keys"] = r.dbsize()
        metrics["active_engine"] = r.get("active_engine") or "unknown"
    except Exception as e:
        metrics["redis_error"] = str(e)

    try:
        sqs = _boto("sqs")
        q_url = sqs.get_queue_url(QueueName="compute-jobs.fifo")["QueueUrl"]
        attrs = sqs.get_queue_attributes(
            QueueUrl=q_url,
            AttributeNames=["ApproximateNumberOfMessages", "ApproximateNumberOfMessagesNotVisible"]
        )["Attributes"]
        metrics["sqs_depth"] = int(attrs.get("ApproximateNumberOfMessages", 0))
        metrics["sqs_inflight"] = int(attrs.get("ApproximateNumberOfMessagesNotVisible", 0))
    except Exception:
        metrics["sqs_depth"] = -1

    try:
        s3 = _boto("s3")
        objs = s3.list_objects_v2(Bucket="similarity-matrices")
        metrics["s3_objects"] = objs.get("KeyCount", 0)
    except Exception:
        metrics["s3_objects"] = -1

    try:
        ddb = _boto("dynamodb")
        r2 = ddb.scan(TableName="compute-jobs", Select="COUNT")
        metrics["dynamo_jobs"] = r2.get("Count", 0)
    except Exception:
        metrics["dynamo_jobs"] = -1

    return _ok(metrics)

@app.get("/api/products")
async def api_products(category: str = "", search: str = "", limit: int = 100, offset: int = 0):
    try:
        conn = _pg()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        base = "SELECT product_id::text, name, category, metadata, created_at::text FROM products WHERE 1=1"
        params: list = []
        if category:
            base += " AND category = %s"
            params.append(category)
        if search:
            base += " AND (name ILIKE %s OR category ILIKE %s)"
            params += [f"%{search}%", f"%{search}%"]
        base += " ORDER BY name LIMIT %s OFFSET %s"
        params += [limit, offset]
        cur.execute(base, params)
        rows = cur.fetchall()

        cur.execute("SELECT DISTINCT category FROM products ORDER BY category")
        cats = [r["category"] for r in cur.fetchall()]

        cur.execute("SELECT COUNT(*) as n FROM products WHERE 1=1" +
                    (" AND category = %s" if category else ""),
                    ([category] if category else []))
        total = cur.fetchone()["n"]
        conn.close()
        return _ok({"products": [dict(r) for r in rows], "categories": cats, "total": total})
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/products/{product_id}")
async def api_product_detail(product_id: str):
    try:
        conn = _pg()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT product_id::text, name, category, metadata, created_at::text FROM products WHERE product_id = %s",
            (product_id,)
        )
        row = cur.fetchone()
        if not row:
            conn.close()
            raise HTTPException(404, "Product not found")
        product = dict(row)

        cur.execute("""
            SELECT p.product_id::text, p.name, p.category, COUNT(DISTINCT i2.user_id) AS shared
            FROM interactions i1
            JOIN interactions i2 ON i1.user_id = i2.user_id AND i2.product_id != %s
            JOIN products p ON p.product_id = i2.product_id
            WHERE i1.product_id = %s
            GROUP BY p.product_id, p.name, p.category
            ORDER BY shared DESC LIMIT 6
        """, (product_id, product_id))
        product["also_viewed"] = [dict(r) for r in cur.fetchall()]
        conn.close()
        return _ok(product)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/users")
async def api_users(limit: int = 100, offset: int = 0, search: str = ""):
    try:
        conn = _pg()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        base = """
            SELECT u.user_id, u.name, u.email,
                   COUNT(i.interaction_id) AS interaction_count
            FROM users u
            LEFT JOIN interactions i ON i.user_id = u.user_id
            WHERE 1=1
        """
        params: list = []
        if search:
            base += " AND (u.name ILIKE %s OR u.email ILIKE %s)"
            params += [f"%{search}%", f"%{search}%"]
        base += " GROUP BY u.user_id, u.name, u.email ORDER BY u.user_id LIMIT %s OFFSET %s"
        params += [limit, offset]
        cur.execute(base, params)
        rows = [dict(r) for r in cur.fetchall()]

        cur.execute("SELECT COUNT(*) as n FROM users" +
                    (" WHERE name ILIKE %s OR email ILIKE %s" if search else ""),
                    ([f"%{search}%", f"%{search}%"] if search else []))
        total = cur.fetchone()["n"]
        conn.close()
        return _ok({"users": rows, "total": total})
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/users/{user_id}/interactions")
async def api_user_interactions(user_id: int):
    try:
        conn = _pg()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT i.interaction_id, i.interaction_type,
                   i."timestamp"::text AS created_at,
                   p.product_id::text, p.name, p.category
            FROM interactions i
            JOIN products p ON p.product_id = i.product_id
            WHERE i.user_id = %s
            ORDER BY i."timestamp" DESC
            LIMIT 50
        """, (user_id,))
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return _ok({"interactions": rows})
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/users/{user_id}/recommendations")
async def api_user_recommendations(user_id: int):
    try:
        conn = _pg()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute("""
            SELECT p.product_id::text, p.name, p.category, p.metadata,
                   COUNT(DISTINCT i2.user_id) AS score
            FROM interactions i1
            JOIN interactions i2 ON i1.product_id = i2.product_id AND i2.user_id != %s
            JOIN interactions i3 ON i3.user_id = i2.user_id
            JOIN products p ON p.product_id = i3.product_id
            WHERE i1.user_id = %s
              AND i3.product_id NOT IN (
                  SELECT product_id FROM interactions WHERE user_id = %s
              )
            GROUP BY p.product_id, p.name, p.category, p.metadata
            ORDER BY score DESC
            LIMIT 12
        """, (user_id, user_id, user_id))
        rows = cur.fetchall()

        if not rows:

            cur.execute("""
                SELECT p.product_id::text, p.name, p.category, p.metadata,
                       COUNT(i.interaction_id) AS score
                FROM products p
                LEFT JOIN interactions i ON i.product_id = p.product_id
                WHERE p.product_id NOT IN (
                    SELECT product_id FROM interactions WHERE user_id = %s
                )
                GROUP BY p.product_id, p.name, p.category, p.metadata
                ORDER BY score DESC
                LIMIT 12
            """, (user_id,))
            rows = cur.fetchall()

        conn.close()
        return _ok({"recommendations": [dict(r) for r in rows], "user_id": user_id})
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/interact")
async def api_interact(payload: InteractPayload):
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.post(f"{API_URL}/interactions", json={
                "user_id": payload.user_id,
                "product_id": payload.product_id,
                "interaction_type": payload.interaction_type,
            })
        if r.status_code not in (200, 201):
            raise HTTPException(r.status_code, r.text)
        return _ok(r.json())
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/trigger/{user_id}")
async def api_trigger(user_id: int, force: bool = True):

    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{API_URL}/recommendations/{user_id}", params={"force": str(force).lower()})
        try:
            payload = r.json()
        except Exception:
            payload = {"raw": r.text}
        return _ok({"status": "dispatched", "http_status": r.status_code, "response": payload})
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/jobs")
async def api_jobs():
    try:
        ddb = _boto("dynamodb")
        resp = ddb.scan(
            TableName="compute-jobs",
            Limit=50,
        )
        items = []
        for item in resp.get("Items", []):
            items.append({k: list(v.values())[0] for k, v in item.items()})
        items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return _ok({"jobs": items[:50]})
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/queues")
async def api_queues():
    try:
        sqs = _boto("sqs")
        queues: Dict[str, Any] = {}
        for qname in ["compute-jobs.fifo", "compute-jobs-dlq.fifo", "compute-notifications"]:
            try:
                url = sqs.get_queue_url(QueueName=qname)["QueueUrl"]
                attrs = sqs.get_queue_attributes(QueueUrl=url, AttributeNames=["All"])["Attributes"]
                queues[qname] = {
                    "url": url,
                    "depth": int(attrs.get("ApproximateNumberOfMessages", 0)),
                    "inflight": int(attrs.get("ApproximateNumberOfMessagesNotVisible", 0)),
                    "delayed": int(attrs.get("ApproximateNumberOfMessagesDelayed", 0)),
                }
            except Exception as e:
                queues[qname] = {"error": str(e)}

        status_counts = {"pending": 0, "computing": 0, "complete": 0, "failed": 0}
        recent_throughput = {"per_minute": 0, "per_5min": 0, "per_hour": 0}
        try:
            ddb = _boto("dynamodb")
            scan = ddb.scan(TableName="compute-jobs", ProjectionExpression="#s, created_at",
                            ExpressionAttributeNames={"#s": "status"})
            now_ts = datetime.now(timezone.utc).timestamp()
            for item in scan.get("Items", []):
                s = item.get("status", {}).get("S", "")
                status_counts[s] = status_counts.get(s, 0) + 1
                created = item.get("created_at", {}).get("S", "")
                try:
                    ts = datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
                    age = now_ts - ts
                    if age <= 60:    recent_throughput["per_minute"] += 1
                    if age <= 300:   recent_throughput["per_5min"] += 1
                    if age <= 3600:  recent_throughput["per_hour"] += 1
                except Exception:
                    pass
        except Exception as exc:
            logger.debug("DynamoDB stats failed: %s", exc)

        with _METRICS_LOCK:
            ts_now = time.time()
            tps = list(_METRICS["throughput_timestamps"])
            processed_per_min  = sum(1 for t in tps if ts_now - t <= 60)
            processed_per_5min = sum(1 for t in tps if ts_now - t <= 300)
            metrics = {
                "total_processed":     _METRICS["total_processed"],
                "peak_main_depth":     _METRICS["peak_main_depth"],
                "peak_inflight":       _METRICS["peak_inflight"],
                "last_processed_at":   _METRICS["last_processed_at"],
                "started_at":          _METRICS["started_at"],
                "processed_per_min":   processed_per_min,
                "processed_per_5min":  processed_per_5min,
            }

        return _ok({
            "queues": queues,
            "metrics": metrics,
            "jobs_by_status": status_counts,
            "dispatched_recent": recent_throughput,
        })
    except Exception as e:
        raise HTTPException(500, str(e))

@app.delete("/api/queues/purge")
async def api_purge_queues():
    try:
        sqs = _boto("sqs")
        purged = []
        for qname in ["compute-jobs.fifo", "compute-jobs-dlq.fifo"]:
            try:
                url = sqs.get_queue_url(QueueName=qname)["QueueUrl"]
                sqs.purge_queue(QueueUrl=url)
                purged.append(qname)
            except Exception as e:
                purged.append(f"{qname}: ERROR {e}")
        return _ok({"purged": purged})
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/s3")
async def api_s3():
    try:
        s3 = _boto("s3")
        buckets = []
        bl = s3.list_buckets().get("Buckets", [])
        for b in bl:
            bname = b["Name"]
            objs = s3.list_objects_v2(Bucket=bname)
            contents = objs.get("Contents", [])
            buckets.append({
                "name": bname,
                "object_count": len(contents),
                "objects": [
                    {"key": o["Key"], "size": o["Size"],
                     "last_modified": o["LastModified"].isoformat()}
                    for o in contents[:20]
                ],
            })
        return _ok({"buckets": buckets})
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/live-engine")
async def api_live_engine():
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{API_URL}/api/engine")
        return _ok(r.json())
    except Exception as e:
        try:
            rv = _redis()
            eng = rv.get("active_engine") or "openmp"
            return _ok({"engine": eng})
        except Exception:
            raise HTTPException(500, str(e))

@app.put("/api/live-engine")
async def api_set_engine(payload: EnginePayload):
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.put(f"{API_URL}/api/engine", json={"engine": payload.engine})
        return _ok(r.json())
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/hpc/engines")
async def api_hpc_engines():
    try:
        from recommendation import ENGINES
        return _ok({"engines": [
            {"id": spec.id, "name": spec.name, "description": spec.description,
             "category": spec.category, "vendor": spec.vendor}
            for spec in ENGINES.values()
        ]})
    except Exception as e:
        raise HTTPException(500, str(e))

def _engine_key(name: str) -> str:

    _map = {"cuda": "CUDA", "openmp": "OpenMP", "mpi": "MPI",
            "opencl": "OpenCL", "tbb": "TBB", "simd": "SIMD"}
    return _map.get(name.lower(), name)

def _result_to_dict(result) -> dict:

    import dataclasses
    def _conv(obj):
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {k: _conv(v) for k, v in dataclasses.asdict(obj).items()}
        if isinstance(obj, list):
            return [_conv(i) for i in obj]
        return obj

    d = _conv(result)

    engines_list = d.get("engines", [])

    for e in engines_list:
        e.setdefault("throughput_gflops", e.get("effective_gflops", 0))
        e.setdefault("latency_ms", round(1000 / max(e.get("effective_gflops", 1), 0.001), 4))
        e.setdefault("efficiency_percent", round(e.get("throughput_score", 0), 2))
        e.setdefault("memory_bandwidth_gbps", e.get("effective_mem_bw", 0))
        e.setdefault("composite_score", round(e.get("overall_score", 0), 2))
    return {
        "results": engines_list,
        "recommended_engine": d.get("top_engine_id", ""),
        "summary": d.get("workload_summary", ""),
        "insights": d.get("insights", []),
        "raw": d,
    }

@app.post("/api/hpc/analyze")
async def api_hpc_analyze(payload: HpcPayload):
    try:
        from recommendation import analyze, ENGINES
        from models import AnalyzeRequest

        normalized = [_engine_key(e) for e in payload.selected_engines]
        unknown = [e for e in normalized if e not in ENGINES]
        if unknown:
            raise HTTPException(400, f"Unknown engines: {unknown}")

        req = AnalyzeRequest(
            dataset_size_gb=payload.dataset_size_gb,
            problem_type=payload.problem_type,
            node_count=payload.node_count,
            precision=payload.precision,
            iterations=payload.iterations,
            memory_bound=payload.memory_bound,
            latency_sensitive=payload.latency_sensitive,
            selected_engines=normalized,
        )
        result = analyze(req)
        return _ok(_result_to_dict(result))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@app.websocket("/ws/logs/{service}")
async def ws_logs(websocket: WebSocket, service: str):
    await websocket.accept()
    container_map = {
        "worker": "docker-worker-1",
        "api": "docker-api-1",
        "localstack": "docker-localstack-1",
        "db": "docker-db-1",
        "redis": "docker-redis-1",
        "dashboard": "docker-dashboard-1",
    }
    cname = container_map.get(service, f"docker-{service}-1")
    if not _docker:
        await websocket.send_text("[Docker SDK unavailable — cannot stream logs]\n")
        await websocket.close()
        return

    queue: asyncio.Queue = asyncio.Queue(maxsize=500)
    loop = asyncio.get_event_loop()
    stop_flag = threading.Event()

    def _reader():
        try:
            container = _docker.containers.get(cname)
            for chunk in container.logs(stream=True, follow=True, tail=200, timestamps=True):
                if stop_flag.is_set():
                    break
                line = chunk.decode("utf-8", errors="replace")
                asyncio.run_coroutine_threadsafe(queue.put(line), loop)
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(
                queue.put(f"[stream error for {cname}: {exc}]\n"), loop
            )
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    try:
        while True:
            try:
                line = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:

                try:
                    await websocket.send_text("")
                except Exception:
                    break
                continue
            if line is None:
                break
            try:
                await websocket.send_text(line)
            except (WebSocketDisconnect, Exception):
                break
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        stop_flag.set()
        try:
            await websocket.close()
        except Exception:
            pass

@app.get("/api/dynamo/tables")
async def api_dynamo_tables():
    try:
        ddb = _boto("dynamodb")
        tables_resp = ddb.list_tables()
        tables = []
        for tname in tables_resp.get("TableNames", []):
            desc = ddb.describe_table(TableName=tname)["Table"]
            tables.append({
                "name": tname,
                "item_count": desc.get("ItemCount", 0),
                "status": desc.get("TableStatus", "UNKNOWN"),
                "size_bytes": desc.get("TableSizeBytes", 0),
            })
        return _ok({"tables": tables})
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/sns/topics")
async def api_sns_topics():
    try:
        sns = _boto("sns")
        sqs = _boto("sqs")
        topics_resp = sns.list_topics().get("Topics", [])
        topics = []
        for t in topics_resp:
            arn = t["TopicArn"]
            subs = sns.list_subscriptions_by_topic(TopicArn=arn).get("Subscriptions", [])
            topics.append({
                "arn": arn,
                "name": arn.rsplit(":", 1)[-1],
                "subscriptions": [
                    {"protocol": s["Protocol"], "endpoint": s["Endpoint"], "arn": s["SubscriptionArn"]}
                    for s in subs
                ],
            })

        depth = 0
        try:
            qurl = sqs.get_queue_url(QueueName="compute-notifications")["QueueUrl"]
            attr = sqs.get_queue_attributes(QueueUrl=qurl, AttributeNames=["ApproximateNumberOfMessages"])
            depth = int(attr["Attributes"].get("ApproximateNumberOfMessages", 0))
        except Exception as exc:
            logger.debug("queue depth read failed: %s", exc)

        with _NOTIF_LOCK:
            notifications = list(_NOTIF_BUFFER)[:50]

        return _ok({
            "topics": topics,
            "notifications": notifications,
            "queue_depth": depth,
            "history_size": len(_NOTIF_BUFFER),
        })
    except Exception as e:
        raise HTTPException(500, str(e))

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RecSys Platform</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;background:#f8fafc;color:#0f172a;display:flex;flex-direction:column;height:100vh;overflow:hidden}
a{color:inherit;text-decoration:none}
button{cursor:pointer;font-family:inherit}
input,select{font-family:inherit}

/* Top bar */
#topbar{height:48px;background:#1e293b;display:flex;align-items:center;padding:0 20px;gap:16px;flex-shrink:0;border-bottom:1px solid #334155}
#topbar .logo{color:#fff;font-weight:700;font-size:15px;letter-spacing:-.3px}
#topbar .logo span{color:#60a5fa}
#topbar .spacer{flex:1}
#live-badge{display:flex;align-items:center;gap:6px;color:#94a3b8;font-size:13px}
#live-dot{width:7px;height:7px;border-radius:50%;background:#16a34a}

/* Layout */
#layout{display:flex;flex:1;overflow:hidden}

/* Sidebar */
#sidebar{width:220px;background:#1e293b;display:flex;flex-direction:column;overflow-y:auto;flex-shrink:0}
#sidebar nav{padding:12px 0}
.nav-section{padding:8px 16px 4px;font-size:11px;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:.6px}
.nav-item{display:flex;align-items:center;gap:10px;padding:8px 16px;color:#94a3b8;font-size:13.5px;cursor:pointer;transition:background .15s,color .15s;border:none;background:none;width:100%;text-align:left}
.nav-item:hover{background:#334155;color:#e2e8f0}
.nav-item.active{background:#2563eb;color:#fff}
.nav-item .icon{font-size:15px;width:18px;text-align:center}

/* Content */
#content{flex:1;overflow-y:auto;padding:24px}

/* Cards */
.card{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:20px;margin-bottom:16px}
.card-title{font-size:14px;font-weight:600;color:#0f172a;margin-bottom:16px}

/* Grid */
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}
.grid-4{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:16px}

/* Stat card */
.stat-card{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:16px}
.stat-label{font-size:12px;color:#64748b;margin-bottom:4px}
.stat-value{font-size:24px;font-weight:700;color:#0f172a}
.stat-sub{font-size:12px;color:#94a3b8;margin-top:2px}

/* Status pill */
.pill{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:500}
.pill.ok{background:#dcfce7;color:#15803d}
.pill.err{background:#fee2e2;color:#b91c1c}
.pill.warn{background:#fef9c3;color:#a16207}
.pill.info{background:#dbeafe;color:#1d4ed8}
.pill-dot{width:6px;height:6px;border-radius:50%;background:currentColor}

/* Table */
.tbl{width:100%;border-collapse:collapse;font-size:13px}
.tbl th{padding:8px 12px;text-align:left;font-weight:600;color:#475569;border-bottom:2px solid #e2e8f0;background:#f8fafc;font-size:12px;text-transform:uppercase;letter-spacing:.4px}
.tbl td{padding:9px 12px;border-bottom:1px solid #f1f5f9;color:#1e293b;vertical-align:middle}
.tbl tr:hover td{background:#f8fafc}
.tbl tr:last-child td{border-bottom:none}

/* Buttons */
.btn{display:inline-flex;align-items:center;gap:6px;padding:7px 14px;border-radius:6px;font-size:13px;font-weight:500;border:none;transition:opacity .15s}
.btn-primary{background:#2563eb;color:#fff}
.btn-primary:hover{opacity:.9}
.btn-success{background:#16a34a;color:#fff}
.btn-success:hover{opacity:.9}
.btn-danger{background:#dc2626;color:#fff}
.btn-danger:hover{opacity:.9}
.btn-outline{background:#fff;color:#475569;border:1px solid #e2e8f0}
.btn-outline:hover{background:#f8fafc}
.btn-sm{padding:4px 10px;font-size:12px}
.btn-xs{padding:2px 8px;font-size:11px;border-radius:4px}

/* Search/input */
.input{padding:8px 12px;border:1px solid #e2e8f0;border-radius:6px;font-size:13px;color:#0f172a;background:#fff;outline:none}
.input:focus{border-color:#2563eb;box-shadow:0 0 0 2px rgba(37,99,235,.1)}
select.input{appearance:none;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%2364748b' d='M6 8L1 3h10z'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 10px center;padding-right:28px}

/* Tabs */
.tabs{display:flex;gap:2px;margin-bottom:16px;border-bottom:2px solid #e2e8f0}
.tab{padding:8px 16px;font-size:13px;font-weight:500;color:#64748b;cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-2px;transition:color .15s,border-color .15s;background:none;border-top:none;border-left:none;border-right:none}
.tab.active{color:#2563eb;border-bottom-color:#2563eb}
.tab:hover{color:#0f172a}

/* Health grid */
.health-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:12px}
.health-item{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:14px}
.health-name{font-size:12px;font-weight:600;color:#475569;margin-bottom:6px}
.health-status{font-size:13px;font-weight:500}
.health-status.ok{color:#16a34a}
.health-status.err{color:#dc2626}
.health-status.loading{color:#94a3b8}

/* Product grid */
.product-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:16px}
.product-card{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:16px;cursor:pointer;transition:border-color .15s,box-shadow .15s}
.product-card:hover{border-color:#2563eb;box-shadow:0 2px 8px rgba(37,99,235,.12)}
.product-card .cat{font-size:11px;font-weight:600;color:#2563eb;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}
.product-card .name{font-size:14px;font-weight:600;color:#0f172a;line-height:1.35;margin-bottom:8px}
.product-card .meta{font-size:12px;color:#64748b}

/* Rec card */
.rec-card{display:flex;gap:12px;padding:12px;border:1px solid #e2e8f0;border-radius:6px;background:#fff;margin-bottom:8px;cursor:pointer;transition:border-color .15s}
.rec-card:hover{border-color:#2563eb}
.rec-card .rc-cat{font-size:11px;color:#2563eb;font-weight:600;text-transform:uppercase}
.rec-card .rc-name{font-size:13px;font-weight:500;color:#0f172a}
.rec-card .rc-score{font-size:12px;color:#64748b}

/* Modal */
.modal-overlay{position:fixed;inset:0;background:rgba(15,23,42,.4);display:flex;align-items:center;justify-content:center;z-index:1000;padding:24px}
.modal{background:#fff;border-radius:10px;max-width:520px;width:100%;max-height:80vh;overflow-y:auto;box-shadow:0 8px 32px rgba(0,0,0,.12)}
.modal-header{display:flex;align-items:center;justify-content:space-between;padding:20px 24px 0}
.modal-title{font-size:16px;font-weight:700;color:#0f172a}
.modal-close{background:none;border:none;font-size:20px;color:#94a3b8;cursor:pointer;line-height:1}
.modal-body{padding:20px 24px 24px}

/* Log viewer */
#log-pane{font-family:'Cascadia Code','Fira Code','Courier New',monospace;font-size:12px;background:#0f172a;color:#e2e8f0;padding:16px;border-radius:6px;height:420px;overflow-y:auto;white-space:pre-wrap;word-break:break-all;line-height:1.6}

/* Loader */
.loader{display:flex;align-items:center;gap:8px;color:#64748b;font-size:13px;padding:24px 0}
.spinner{width:16px;height:16px;border:2px solid #e2e8f0;border-top-color:#2563eb;border-radius:50%;animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

/* Pipeline diagram */
.pipeline{display:flex;align-items:center;gap:0;overflow-x:auto;padding:12px 0}
.pipe-step{display:flex;align-items:center;gap:0;flex-shrink:0}
.pipe-box{background:#fff;border:1px solid #e2e8f0;border-radius:6px;padding:10px 14px;font-size:12px;font-weight:500;color:#0f172a;min-width:90px;text-align:center}
.pipe-box.ok{border-color:#16a34a;background:#f0fdf4;color:#15803d}
.pipe-box.err{border-color:#dc2626;background:#fef2f2;color:#b91c1c}
.pipe-arrow{color:#94a3b8;padding:0 6px;font-size:16px}

/* Responsive */
@media(max-width:768px){
  #sidebar{display:none}
  .grid-2,.grid-3,.grid-4{grid-template-columns:1fr}
}

.section{display:none}
.section.active{display:block}
.flex{display:flex}
.items-center{align-items:center}
.gap-8{gap:8px}
.gap-12{gap:12px}
.mb-16{margin-bottom:16px}
.mt-16{margin-top:16px}
.text-sm{font-size:13px}
.text-xs{font-size:12px}
.text-muted{color:#64748b}
.font-bold{font-weight:700}
.font-medium{font-weight:500}
.truncate{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:200px}
</style>
</head>
<body>

<!-- Top bar -->
<div id="topbar">
  <div class="logo">■ <span>RecSys</span> Platform</div>
  <div class="spacer"></div>
  <div id="live-badge"><div id="live-dot"></div><span id="live-time">--:--:--</span></div>
</div>

<!-- Layout -->
<div id="layout">

<!-- Sidebar -->
<div id="sidebar">
  <nav>
    <div class="nav-section">Monitor</div>
    <button class="nav-item active" onclick="showSection('overview')"><span class="icon">◈</span>Overview</button>
    <button class="nav-item" onclick="showSection('jobs')"><span class="icon">⚙</span>Compute Jobs</button>
    <button class="nav-item" onclick="showSection('aws')"><span class="icon">☁</span>AWS Services</button>
    <button class="nav-item" onclick="showSection('logs')"><span class="icon">≡</span>Live Logs</button>

    <div class="nav-section">Store</div>
    <button class="nav-item" onclick="showSection('storefront')"><span class="icon">⊞</span>Storefront</button>
    <button class="nav-item" onclick="showSection('products')"><span class="icon">◻</span>Products</button>
    <button class="nav-item" onclick="showSection('users')"><span class="icon">◎</span>Users</button>

    <div class="nav-section">System</div>
    <button class="nav-item" onclick="showSection('engine')"><span class="icon">△</span>Engine Control</button>
    <button class="nav-item" onclick="showSection('hpc')"><span class="icon">⬡</span>HPC Analyzer</button>
  </nav>
</div>

<!-- Content -->
<div id="content">
<div id="sec-overview" class="section active">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px">
    <div>
      <div style="font-size:20px;font-weight:700">Overview</div>
      <div style="font-size:13px;color:#64748b;margin-top:2px">System health and key metrics</div>
    </div>
    <button class="btn btn-outline btn-sm" onclick="loadOverview()">Refresh</button>
  </div>

  <div class="card" style="margin-bottom:16px">
    <div class="card-title">Service Health</div>
    <div id="health-grid" class="health-grid"><div class="loader"><div class="spinner"></div>Loading…</div></div>
  </div>

  <div class="card" style="margin-bottom:16px">
    <div class="card-title">Pipeline</div>
    <div id="pipeline-view" class="pipeline"><div class="loader"><div class="spinner"></div>Loading…</div></div>
  </div>

  <div id="metrics-grid" class="grid-4" style="margin-bottom:0">
    <div class="stat-card"><div class="stat-label">Users</div><div class="stat-value" id="m-users">—</div></div>
    <div class="stat-card"><div class="stat-label">Products</div><div class="stat-value" id="m-products">—</div></div>
    <div class="stat-card"><div class="stat-label">Interactions</div><div class="stat-value" id="m-interactions">—</div></div>
    <div class="stat-card"><div class="stat-label">Compute Jobs</div><div class="stat-value" id="m-jobs">—</div></div>
    <div class="stat-card"><div class="stat-label">SQS Depth</div><div class="stat-value" id="m-sqs">—</div><div class="stat-sub">main queue</div></div>
    <div class="stat-card"><div class="stat-label">S3 Objects</div><div class="stat-value" id="m-s3">—</div></div>
    <div class="stat-card"><div class="stat-label">Redis Keys</div><div class="stat-value" id="m-rkeys">—</div></div>
    <div class="stat-card"><div class="stat-label">Active Engine</div><div class="stat-value" id="m-engine" style="font-size:16px;text-transform:uppercase">—</div></div>
  </div>
</div>
<div id="sec-jobs" class="section">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px">
    <div>
      <div style="font-size:20px;font-weight:700">Compute Jobs</div>
      <div style="font-size:13px;color:#64748b;margin-top:2px">DynamoDB job table — auto-refreshes every 5s</div>
    </div>
    <div style="display:flex;gap:8px">
      <button class="btn btn-danger btn-sm" onclick="purgeQueues()">Purge Queues</button>
      <button class="btn btn-outline btn-sm" onclick="loadJobs()">Refresh</button>
    </div>
  </div>
  <div class="card">
    <div id="jobs-table"><div class="loader"><div class="spinner"></div>Loading…</div></div>
  </div>
</div>
<div id="sec-aws" class="section">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px">
    <div>
      <div style="font-size:20px;font-weight:700">AWS Services</div>
      <div style="font-size:13px;color:#64748b;margin-top:2px">LocalStack — SQS, S3, DynamoDB, SNS</div>
    </div>
    <button class="btn btn-outline btn-sm" onclick="loadAws()">Refresh</button>
  </div>

  <div class="grid-2">
    <div class="card">
      <div class="card-title">SQS Queues</div>
      <div id="sqs-view"><div class="loader"><div class="spinner"></div>Loading…</div></div>
    </div>
    <div class="card">
      <div class="card-title">DynamoDB Tables</div>
      <div id="dynamo-view"><div class="loader"><div class="spinner"></div>Loading…</div></div>
    </div>
  </div>

  <div class="card">
    <div class="card-title">S3 Buckets</div>
    <div id="s3-view"><div class="loader"><div class="spinner"></div>Loading…</div></div>
  </div>

  <div class="card">
    <div class="card-title">SNS Topics</div>
    <div id="sns-view"><div class="loader"><div class="spinner"></div>Loading…</div></div>
  </div>
</div>
<div id="sec-logs" class="section">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px">
    <div>
      <div style="font-size:20px;font-weight:700">Live Logs</div>
      <div style="font-size:13px;color:#64748b;margin-top:2px">Real-time Docker log streams</div>
    </div>
    <div style="display:flex;gap:8px">
      <button class="btn btn-outline btn-sm" id="log-pause-btn" onclick="toggleLogPause()">⏸ Pause</button>
      <button class="btn btn-outline btn-sm" onclick="clearLogs()">✕ Clear</button>
    </div>
  </div>
  <div style="margin-bottom:16px">
    <div class="tabs" id="log-tabs">
      <button class="tab active" onclick="switchLog('worker')">Worker</button>
      <button class="tab" onclick="switchLog('api')">API</button>
      <button class="tab" onclick="switchLog('localstack')">LocalStack</button>
      <button class="tab" onclick="switchLog('db')">DB</button>
      <button class="tab" onclick="switchLog('redis')">Redis</button>
      <button class="tab" onclick="switchLog('dashboard')">Dashboard</button>
    </div>
    <div id="log-pane">[Select a service and connect to view logs]</div>
    <div style="margin-top:8px;display:flex;gap:8px">
      <button class="btn btn-primary btn-sm" onclick="connectLog()">▶ Connect</button>
      <button class="btn btn-outline btn-sm" onclick="disconnectLog()">■ Disconnect</button>
      <span id="log-status" style="font-size:12px;color:#64748b;line-height:32px"></span>
    </div>
  </div>
</div>
<div id="sec-storefront" class="section">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px">
    <div>
      <div style="font-size:20px;font-weight:700">Storefront</div>
      <div style="font-size:13px;color:#64748b;margin-top:2px">Browse as a user, record interactions, get recommendations</div>
    </div>
  </div>

  <div style="display:flex;gap:16px;margin-bottom:20px;align-items:center;flex-wrap:wrap">
    <select class="input" id="sf-user" onchange="sfUserChanged()" style="min-width:200px">
      <option value="">— Select a user —</option>
    </select>
    <span id="sf-user-info" style="font-size:13px;color:#64748b"></span>
  </div>

  <div style="display:flex;gap:20px" id="sf-layout">
    <!-- Product side -->
    <div style="flex:1;min-width:0">
      <div style="display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap;align-items:center">
        <input class="input" id="sf-search" placeholder="Search products…" oninput="sfSearch()" style="flex:1;min-width:160px">
        <select class="input" id="sf-category" onchange="sfCategoryChange()">
          <option value="">All categories</option>
        </select>
      </div>
      <div id="sf-products" class="product-grid">
        <div class="loader"><div class="spinner"></div>Loading products…</div>
      </div>
      <div id="sf-pagination" style="margin-top:16px;display:flex;gap:8px;align-items:center">
        <button class="btn btn-outline btn-sm" id="sf-prev" onclick="sfPage(-1)" disabled>← Prev</button>
        <span id="sf-page-info" style="font-size:13px;color:#64748b"></span>
        <button class="btn btn-outline btn-sm" id="sf-next" onclick="sfPage(1)">Next →</button>
      </div>
    </div>

    <!-- Rec panel -->
    <div style="width:280px;flex-shrink:0">
      <div class="card" id="sf-rec-panel">
        <div class="card-title">Your Recommendations</div>
        <div id="sf-recs" style="font-size:13px;color:#94a3b8">Select a user to see recommendations</div>
        <div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap">
          <button class="btn btn-outline btn-sm" onclick="sfLoadRecs()">Refresh</button>
          <button class="btn btn-primary btn-sm" onclick="sfTriggerJob()">⚙ Trigger Job</button>
        </div>
        <div id="sf-trigger-status" style="font-size:12px;color:#64748b;margin-top:8px"></div>
      </div>

      <div class="card" style="margin-top:0">
        <div class="card-title">Recent Activity</div>
        <div id="sf-recent-activity" style="font-size:13px;color:#94a3b8">—</div>
      </div>
    </div>
  </div>
</div>
<div id="sec-products" class="section">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px">
    <div>
      <div style="font-size:20px;font-weight:700">Products</div>
      <div style="font-size:13px;color:#64748b;margin-top:2px" id="prod-count">—</div>
    </div>
  </div>
  <div class="card">
    <div style="display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap">
      <input class="input" id="prod-search" placeholder="Search by name or category…" oninput="prodSearch()" style="flex:1;min-width:200px">
      <select class="input" id="prod-cat-filter" onchange="prodSearch()">
        <option value="">All categories</option>
      </select>
    </div>
    <div id="prod-table"><div class="loader"><div class="spinner"></div>Loading…</div></div>
    <div style="margin-top:12px;display:flex;gap:8px;align-items:center">
      <button class="btn btn-outline btn-sm" id="prod-prev" onclick="prodPage(-1)" disabled>← Prev</button>
      <span id="prod-page-info" style="font-size:13px;color:#64748b"></span>
      <button class="btn btn-outline btn-sm" id="prod-next" onclick="prodPage(1)">Next →</button>
    </div>
  </div>
</div>
<div id="sec-users" class="section">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px">
    <div>
      <div style="font-size:20px;font-weight:700">Users</div>
      <div style="font-size:13px;color:#64748b;margin-top:2px" id="users-count">—</div>
    </div>
  </div>
  <div class="card">
    <div style="display:flex;gap:8px;margin-bottom:16px">
      <input class="input" id="users-search" placeholder="Search by name or email…" oninput="usersSearch()" style="flex:1">
    </div>
    <div id="users-table"><div class="loader"><div class="spinner"></div>Loading…</div></div>
    <div style="margin-top:12px;display:flex;gap:8px;align-items:center">
      <button class="btn btn-outline btn-sm" id="users-prev" onclick="usersPage(-1)" disabled>← Prev</button>
      <span id="users-page-info" style="font-size:13px;color:#64748b"></span>
      <button class="btn btn-outline btn-sm" id="users-next" onclick="usersPage(1)">Next →</button>
    </div>
  </div>
</div>
<div id="sec-engine" class="section">
  <div style="margin-bottom:20px">
    <div style="font-size:20px;font-weight:700">Engine Control</div>
    <div style="font-size:13px;color:#64748b;margin-top:2px">Switch the active recommendation compute engine</div>
  </div>

  <div class="card">
    <div class="card-title">Active Engine</div>
    <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px">
      <div id="active-engine-display" style="font-size:28px;font-weight:700;text-transform:uppercase;color:#2563eb">—</div>
      <button class="btn btn-outline btn-sm" onclick="loadEngine()">Refresh</button>
    </div>
    <div style="display:flex;gap:12px;flex-wrap:wrap">
      <button class="btn btn-primary" onclick="setEngine('openmp')">OpenMP</button>
      <button class="btn btn-primary" onclick="setEngine('cuda')">CUDA</button>
      <button class="btn btn-primary" onclick="setEngine('mpi')">MPI</button>
    </div>
    <div id="engine-msg" style="margin-top:12px;font-size:13px;color:#64748b"></div>
  </div>

  <div class="card">
    <div class="card-title">Engine Descriptions</div>
    <table class="tbl">
      <thead><tr><th>Engine</th><th>Use Case</th><th>Parallelism</th></tr></thead>
      <tbody>
        <tr><td><strong>OpenMP</strong></td><td>Single-node, shared memory</td><td>CPU thread-level</td></tr>
        <tr><td><strong>CUDA</strong></td><td>GPU-accelerated matrix ops</td><td>GPU SIMT</td></tr>
        <tr><td><strong>MPI</strong></td><td>Multi-node distributed</td><td>Process/network</td></tr>
      </tbody>
    </table>
  </div>
</div>
<div id="sec-hpc" class="section">
  <div style="margin-bottom:20px">
    <div style="font-size:20px;font-weight:700">HPC Analyzer</div>
    <div style="font-size:13px;color:#64748b;margin-top:2px">Comparative performance analysis across compute engines</div>
  </div>

  <div class="grid-2">
    <div class="card">
      <div class="card-title">Workload Configuration</div>
      <div style="display:flex;flex-direction:column;gap:12px">
        <div>
          <label style="font-size:12px;font-weight:600;color:#475569;display:block;margin-bottom:4px">Dataset Size (GB)</label>
          <input class="input" type="number" id="hpc-size" value="10" min="0.1" max="1000" step="0.1" style="width:100%">
        </div>
        <div>
          <label style="font-size:12px;font-weight:600;color:#475569;display:block;margin-bottom:4px">Problem Type</label>
          <select class="input" id="hpc-problem" style="width:100%">
            <option value="machine_learning">Machine Learning / Training</option>
            <option value="linear_algebra">Dense Linear Algebra (BLAS)</option>
            <option value="embarrassingly_parallel">Embarrassingly Parallel</option>
            <option value="reduction">Reduction / Aggregation</option>
            <option value="stencil">Stencil / Neighbor Access</option>
            <option value="graph_traversal">Graph Traversal</option>
            <option value="fft">FFT / Spectral Methods</option>
            <option value="monte_carlo">Monte Carlo / Sampling</option>
          </select>
        </div>
        <div>
          <label style="font-size:12px;font-weight:600;color:#475569;display:block;margin-bottom:4px">Node Count</label>
          <input class="input" type="number" id="hpc-nodes" value="1" min="1" max="1000" style="width:100%">
        </div>
        <div>
          <label style="font-size:12px;font-weight:600;color:#475569;display:block;margin-bottom:4px">Precision</label>
          <select class="input" id="hpc-precision" style="width:100%">
            <option value="fp32">FP32</option>
            <option value="fp64">FP64</option>
            <option value="fp16">FP16</option>
            <option value="int8">INT8</option>
          </select>
        </div>
        <div>
          <label style="font-size:12px;font-weight:600;color:#475569;display:block;margin-bottom:4px">Iterations</label>
          <input class="input" type="number" id="hpc-iters" value="100" min="1" style="width:100%">
        </div>
        <div style="display:flex;gap:16px">
          <label style="font-size:13px;color:#475569;display:flex;align-items:center;gap:6px;cursor:pointer">
            <input type="checkbox" id="hpc-membound"> Memory-bound
          </label>
          <label style="font-size:13px;color:#475569;display:flex;align-items:center;gap:6px;cursor:pointer">
            <input type="checkbox" id="hpc-latency"> Latency-sensitive
          </label>
        </div>
        <div>
          <label style="font-size:12px;font-weight:600;color:#475569;display:block;margin-bottom:6px">Engines to Compare</label>
          <div style="display:flex;gap:12px;flex-wrap:wrap">
            <label style="font-size:13px;display:flex;align-items:center;gap:5px;cursor:pointer"><input type="checkbox" class="hpc-eng" value="openmp" checked> OpenMP</label>
            <label style="font-size:13px;display:flex;align-items:center;gap:5px;cursor:pointer"><input type="checkbox" class="hpc-eng" value="cuda" checked> CUDA</label>
            <label style="font-size:13px;display:flex;align-items:center;gap:5px;cursor:pointer"><input type="checkbox" class="hpc-eng" value="mpi" checked> MPI</label>
            <label style="font-size:13px;display:flex;align-items:center;gap:5px;cursor:pointer"><input type="checkbox" class="hpc-eng" value="opencl"> OpenCL</label>
            <label style="font-size:13px;display:flex;align-items:center;gap:5px;cursor:pointer"><input type="checkbox" class="hpc-eng" value="tbb"> TBB</label>
            <label style="font-size:13px;display:flex;align-items:center;gap:5px;cursor:pointer"><input type="checkbox" class="hpc-eng" value="simd"> SIMD</label>
          </div>
        </div>
        <button class="btn btn-primary" onclick="runHpc()" id="hpc-run-btn">▶ Run Analysis</button>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Results</div>
      <div id="hpc-results" style="font-size:13px;color:#94a3b8">Configure and run an analysis to see results here.</div>
    </div>
  </div>
</div>

</div><!-- /content -->
</div><!-- /layout -->

<!-- Product modal -->
<div id="product-modal" class="modal-overlay" style="display:none" onclick="closeModal(event)">
  <div class="modal" onclick="event.stopPropagation()">
    <div class="modal-header">
      <div class="modal-title" id="modal-product-name">Product</div>
      <button class="modal-close" onclick="closeProductModal()">×</button>
    </div>
    <div class="modal-body">
      <div id="modal-cat" style="font-size:12px;color:#2563eb;font-weight:600;text-transform:uppercase;margin-bottom:8px"></div>
      <div id="modal-meta" style="font-size:13px;color:#475569;margin-bottom:20px"></div>
      <div style="display:flex;gap:10px;margin-bottom:20px" id="modal-actions">
        <button class="btn btn-outline" onclick="doInteract('view')">👁 View</button>
        <button class="btn btn-success" onclick="doInteract('like')">♡ Like</button>
        <button class="btn btn-primary" onclick="doInteract('purchase')">⊕ Purchase</button>
      </div>
      <div id="modal-interact-msg" style="font-size:13px;margin-bottom:16px"></div>
      <div id="modal-also" style="font-size:13px;color:#64748b"></div>
    </div>
  </div>
</div>

<!-- User history modal -->
<div id="user-modal" class="modal-overlay" style="display:none" onclick="closeModal(event)">
  <div class="modal" style="max-width:640px" onclick="event.stopPropagation()">
    <div class="modal-header">
      <div class="modal-title" id="um-title">User History</div>
      <button class="modal-close" onclick="document.getElementById('user-modal').style.display='none'">×</button>
    </div>
    <div class="modal-body">
      <div id="um-body"><div class="loader"><div class="spinner"></div>Loading…</div></div>
    </div>
  </div>
</div>

<script>
let currentSection = 'overview';
let _ws = null;
let _logPaused = false;
let _currentLogService = 'worker';
let _sfOffset = 0;
let _sfTotal = 0;
const SF_PAGE = 30;
let _prodOffset = 0;
let _prodTotal = 0;
const PROD_PAGE = 50;
let _usersOffset = 0;
let _usersTotal = 0;
const USERS_PAGE = 50;
let _selectedProduct = null;
let _selectedUserId = null;
let _sfSearchTimer = null;
let _prodSearchTimer = null;
let _usersSearchTimer = null;
let _jobsInterval = null;
function showSection(name) {
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
  const sec = document.getElementById('sec-' + name);
  if (sec) sec.classList.add('active');
  document.querySelectorAll('.nav-item').forEach(b => {
    if (b.getAttribute('onclick') && b.getAttribute('onclick').includes("'" + name + "'"))
      b.classList.add('active');
  });
  currentSection = name;

  if (name === 'overview') loadOverview();
  if (name === 'jobs') { loadJobs(); startJobsInterval(); }
  else stopJobsInterval();
  if (name === 'aws') loadAws();
  if (name === 'storefront') loadStorefront();
  if (name === 'products') loadProducts();
  if (name === 'users') loadUsers();
  if (name === 'engine') loadEngine();
}
function updateClock() {
  const now = new Date();
  document.getElementById('live-time').textContent =
    now.toLocaleTimeString('en-US', {hour12: false});
}
setInterval(updateClock, 1000);
updateClock();
async function loadOverview() {
  await Promise.all([loadHealth(), loadMetrics()]);
}

async function loadHealth() {
  const hg = document.getElementById('health-grid');
  const pv = document.getElementById('pipeline-view');
  try {
    const data = await apiFetch('/api/status');
    const services = ['api', 'db', 'redis', 'localstack'];
    hg.innerHTML = services.map(s => {
      const st = data[s] || 'unknown';
      const ok = st === 'ok';
      return `<div class="health-item">
        <div class="health-name">${s.toUpperCase()}</div>
        <div class="health-status ${ok ? 'ok' : 'err'}">${ok ? '● Online' : '● ' + st}</div>
      </div>`;
    }).join('') + Object.entries(data.containers || {}).map(([name, st]) => {
      const short = name.replace('docker-', '').replace('-1', '');
      return `<div class="health-item">
        <div class="health-name">${short}</div>
        <div class="health-status ${st === 'running' ? 'ok' : 'err'}">${st === 'running' ? '● Running' : '● ' + st}</div>
      </div>`;
    }).join('');

    // Pipeline
    const steps = [
      {label: 'User', key: null, ok: true},
      {label: 'API', key: 'api', ok: data.api === 'ok'},
      {label: 'SQS', key: 'localstack', ok: data.localstack === 'ok'},
      {label: 'Worker', key: 'localstack', ok: data.localstack === 'ok'},
      {label: 'DynamoDB', key: 'localstack', ok: data.localstack === 'ok'},
      {label: 'S3', key: 'localstack', ok: data.localstack === 'ok'},
    ];
    pv.innerHTML = steps.map((s, i) =>
      `<div class="pipe-step">
        <div class="pipe-box ${s.ok ? 'ok' : 'err'}">${s.label}</div>
        ${i < steps.length - 1 ? '<div class="pipe-arrow">→</div>' : ''}
      </div>`
    ).join('');
  } catch (e) {
    hg.innerHTML = `<div style="color:#dc2626;font-size:13px">Error: ${e.message}</div>`;
    pv.innerHTML = '';
  }
}

async function loadMetrics() {
  try {
    const data = await apiFetch('/api/metrics');
    setText('m-users', fmt(data.users));
    setText('m-products', fmt(data.products));
    setText('m-interactions', fmt(data.interactions));
    setText('m-jobs', fmt(data.dynamo_jobs));
    setText('m-sqs', fmt(data.sqs_depth));
    setText('m-s3', fmt(data.s3_objects));
    setText('m-rkeys', fmt(data.redis_keys));
    setText('m-engine', data.active_engine || '—');
  } catch(e) {}
}
async function loadJobs() {
  const el = document.getElementById('jobs-table');
  el.innerHTML = '<div class="loader"><div class="spinner"></div>Loading…</div>';
  try {
    const data = await apiFetch('/api/jobs');
    const jobs = data.jobs || [];
    if (!jobs.length) {
      el.innerHTML = '<div style="color:#94a3b8;font-size:13px;padding:12px 0">No jobs found in DynamoDB.</div>';
      return;
    }
    el.innerHTML = `<table class="tbl">
      <thead><tr>
        <th>Job ID</th><th>User</th><th>Status</th><th>Engine</th><th>Created</th><th>Updated</th>
      </tr></thead>
      <tbody>${jobs.map(j => `<tr>
        <td><span class="truncate" title="${j.job_id || ''}">${(j.job_id || '').slice(0,16)}…</span></td>
        <td>${j.user_id || '—'}</td>
        <td>${statusPill(j.status)}</td>
        <td><span style="text-transform:uppercase;font-size:12px;font-weight:600">${j.engine || '—'}</span></td>
        <td style="font-size:12px;color:#64748b">${fmtDate(j.created_at)}</td>
        <td style="font-size:12px;color:#64748b">${fmtDate(j.updated_at)}</td>
      </tr>`).join('')}</tbody>
    </table>`;
  } catch (e) {
    el.innerHTML = `<div style="color:#dc2626;font-size:13px">Error: ${e.message}</div>`;
  }
}

function startJobsInterval() {
  if (_jobsInterval) return;
  _jobsInterval = setInterval(() => {
    if (currentSection === 'jobs') loadJobs();
  }, 5000);
}

function stopJobsInterval() {
  if (_jobsInterval) { clearInterval(_jobsInterval); _jobsInterval = null; }
}

async function purgeQueues() {
  if (!confirm('Purge both SQS queues? This cannot be undone.')) return;
  try {
    const data = await apiFetch('/api/queues/purge', {method: 'DELETE'});
    alert('Purged: ' + (data.purged || []).join(', '));
    loadJobs();
  } catch (e) {
    alert('Error: ' + e.message);
  }
}
async function loadAws() {
  await Promise.all([loadSqs(), loadDynamo(), loadS3(), loadSns()]);
}

async function loadSqs() {
  const el = document.getElementById('sqs-view');
  try {
    const data = await apiFetch('/api/queues');
    const queues = data.queues || {};
    const m = data.metrics || {};
    const jobs = data.jobs_by_status || {};
    const disp = data.dispatched_recent || {};

    const stat = (label, val, color) => `
      <div style="flex:1;min-width:110px;padding:10px 12px;background:#fff;border:1px solid #e2e8f0;border-radius:6px">
        <div style="font-size:10px;text-transform:uppercase;letter-spacing:.5px;color:#64748b;margin-bottom:2px">${label}</div>
        <div style="font-size:20px;font-weight:600;color:${color || '#0f172a'};line-height:1.1">${val}</div>
      </div>`;

    const statusPill = (label, n, color) => `
      <span style="display:inline-block;background:${color}1a;color:${color};padding:3px 8px;border-radius:10px;font-size:11px;font-weight:600;margin-right:6px">${label}: ${n}</span>`;

    const queuesHtml = Object.entries(queues).map(([q, info]) => {
      if (info.error) return `<div style="color:#dc2626;font-size:13px;margin-bottom:8px">${q}: ${info.error}</div>`;
      const isMain = q === 'compute-jobs.fifo';
      const active = info.depth > 0 || info.inflight > 0;
      return `<div style="margin-bottom:8px;padding:10px 12px;border:1px solid ${active ? '#2563eb' : '#e2e8f0'};border-radius:6px;background:${active ? '#eff6ff' : '#fff'}">
        <div style="font-size:12px;font-weight:600;color:#0f172a;margin-bottom:6px">${q} ${active ? '<span style="color:#2563eb;font-size:10px;margin-left:6px">● ACTIVE</span>' : ''}</div>
        <div style="display:flex;gap:16px;font-size:13px;color:#475569">
          <span>Depth: <strong>${info.depth}</strong></span>
          <span>In-flight: <strong>${info.inflight}</strong></span>
          <span>Delayed: <strong>${info.delayed}</strong></span>
          ${isMain ? `<span style="color:#94a3b8">peak: ${m.peak_main_depth || 0} / ${m.peak_inflight || 0}</span>` : ''}
        </div>
      </div>`;
    }).join('');

    el.innerHTML = `
      <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px">
        ${stat('Total processed', m.total_processed ?? 0, '#16a34a')}
        ${stat('Per minute', m.processed_per_min ?? 0, '#2563eb')}
        ${stat('Per 5 min', m.processed_per_5min ?? 0, '#2563eb')}
        ${stat('Peak depth', m.peak_main_depth ?? 0, '#d97706')}
        ${stat('Peak in-flight', m.peak_inflight ?? 0, '#d97706')}
      </div>

      <div style="margin-bottom:14px;padding:10px 12px;border:1px solid #e2e8f0;border-radius:6px;background:#fff">
        <div style="font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:#64748b;margin-bottom:6px">Jobs by status (DynamoDB)</div>
        ${statusPill('pending', jobs.pending || 0, '#d97706')}
        ${statusPill('computing', jobs.computing || 0, '#2563eb')}
        ${statusPill('complete', jobs.complete || 0, '#16a34a')}
        ${statusPill('failed', jobs.failed || 0, '#dc2626')}
        <div style="margin-top:8px;font-size:11px;color:#64748b">
          Dispatched in last hour: <strong>${disp.per_hour || 0}</strong>
          • last 5 min: <strong>${disp.per_5min || 0}</strong>
          • last min: <strong>${disp.per_minute || 0}</strong>
        </div>
      </div>

      ${queuesHtml || '<div style="color:#94a3b8;font-size:13px">No queues found.</div>'}

      ${m.last_processed_at ? `<div style="font-size:11px;color:#94a3b8;margin-top:8px">Last notification observed: ${m.last_processed_at}</div>` : ''}
    `;
  } catch (e) {
    el.innerHTML = `<div style="color:#dc2626;font-size:13px">Error: ${e.message}</div>`;
  }
}

async function loadDynamo() {
  const el = document.getElementById('dynamo-view');
  try {
    const data = await apiFetch('/api/dynamo/tables');
    const tables = data.tables || [];
    if (!tables.length) { el.innerHTML = '<div style="color:#94a3b8;font-size:13px">No tables found.</div>'; return; }
    el.innerHTML = tables.map(t => `
      <div style="margin-bottom:8px;padding:10px 12px;border:1px solid #e2e8f0;border-radius:6px;display:flex;justify-content:space-between;align-items:center">
        <span style="font-size:13px;font-weight:500">${t.name}</span>
        <span style="font-size:12px;color:#64748b">${fmt(t.item_count)} items · ${t.status}</span>
      </div>`).join('');
  } catch (e) {
    el.innerHTML = `<div style="color:#dc2626;font-size:13px">Error: ${e.message}</div>`;
  }
}

async function loadS3() {
  const el = document.getElementById('s3-view');
  try {
    const data = await apiFetch('/api/s3');
    const buckets = data.buckets || [];
    if (!buckets.length) { el.innerHTML = '<div style="color:#94a3b8;font-size:13px">No buckets found.</div>'; return; }
    el.innerHTML = buckets.map(b => `
      <div style="margin-bottom:12px">
        <div style="font-size:13px;font-weight:600;color:#0f172a;margin-bottom:6px">
          ${b.name} <span style="font-size:12px;color:#64748b;font-weight:400">(${b.object_count} objects)</span>
        </div>
        ${b.objects.length ? `<table class="tbl"><thead><tr><th>Key</th><th>Size</th><th>Modified</th></tr></thead><tbody>
          ${b.objects.map(o => `<tr>
            <td style="font-size:12px">${o.key}</td>
            <td style="font-size:12px">${fmtBytes(o.size)}</td>
            <td style="font-size:12px;color:#64748b">${fmtDate(o.last_modified)}</td>
          </tr>`).join('')}
        </tbody></table>` : '<div style="font-size:12px;color:#94a3b8">Empty bucket</div>'}
      </div>`).join('');
  } catch (e) {
    el.innerHTML = `<div style="color:#dc2626;font-size:13px">Error: ${e.message}</div>`;
  }
}

async function snsTriggerTest() {
  const status = document.getElementById('sns-trigger-status');
  if (status) status.textContent = 'Dispatching job for user 1…';
  try {
    const r = await apiFetch('/api/trigger/1?force=true', {method:'POST'});
    if (status) status.textContent = `✓ Dispatched (HTTP ${r.http_status}) — job_id=${(r.response?.job_id || '').slice(0,8)}… — refreshing in 4s`;
    setTimeout(loadSns, 4000);
  } catch (e) {
    if (status) status.textContent = `✗ Failed: ${e.message}`;
  }
}

async function loadSns() {
  const el = document.getElementById('sns-view');
  try {
    const data = await apiFetch('/api/sns/topics');
    const topics = data.topics || [];
    const notifs = data.notifications || [];
    const depth = data.queue_depth ?? 0;
    if (!topics.length) { el.innerHTML = '<div style="color:#94a3b8;font-size:13px">No topics found.</div>'; return; }

    const topicsHtml = topics.map(t => `
      <div style="border:1px solid #e2e8f0;border-radius:6px;padding:10px;margin-bottom:8px;background:#fff">
        <div style="font-weight:600;font-size:13px;color:#0f172a">${t.name}</div>
        <div style="font-size:11px;font-family:monospace;color:#64748b;margin:2px 0 6px 0">${t.arn}</div>
        ${t.subscriptions.length ? `
          <div style="font-size:11px;color:#475569;margin-bottom:4px">${t.subscriptions.length} subscriber(s):</div>
          ${t.subscriptions.map(s => `
            <div style="font-size:11px;font-family:monospace;color:#0f172a;padding:2px 0">
              <span style="background:#dbeafe;color:#1e40af;padding:1px 6px;border-radius:3px;margin-right:6px">${s.protocol}</span>${s.endpoint}
            </div>`).join('')}
        ` : `<div style="font-size:11px;color:#dc2626">No subscriptions</div>`}
      </div>`).join('');

    const histSize = data.history_size ?? notifs.length;
    const notifsHtml = `
      <div style="margin-top:14px">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
          <div style="font-weight:600;font-size:13px;color:#0f172a">
            Recent notifications
            <span style="color:#64748b;font-weight:400;font-size:12px">
              (showing ${notifs.length} of ${histSize} • queue depth: ${depth})
            </span>
          </div>
          <div style="display:flex;gap:6px">
            <button onclick="snsTriggerTest()" style="background:#2563eb;color:#fff;border:0;padding:5px 10px;border-radius:4px;font-size:12px;cursor:pointer">Fire test job</button>
            <button onclick="loadSns()" style="background:#fff;color:#475569;border:1px solid #e2e8f0;padding:5px 10px;border-radius:4px;font-size:12px;cursor:pointer">Refresh</button>
          </div>
        </div>
        <div id="sns-trigger-status" style="font-size:11px;color:#64748b;margin-bottom:6px;min-height:14px"></div>
        ${notifs.length ? `
          <div style="border:1px solid #e2e8f0;border-radius:6px;background:#fff;max-height:340px;overflow:auto">
            ${notifs.map(n => `
              <div style="padding:8px 10px;border-bottom:1px solid #f1f5f9;font-size:12px;font-family:monospace">
                <div style="color:#16a34a;font-weight:600">user=${n.user_id ?? '?'} job=${(n.job_id || '').slice(0,8)} status=${n.status ?? '?'}</div>
                <div style="color:#64748b;margin-top:2px">s3=${n.s3_key ?? '—'}</div>
                <div style="color:#94a3b8;font-size:10px">published=${n.timestamp ?? ''} • seen=${n._seen_at ?? ''}</div>
              </div>`).join('')}
          </div>
        ` : `<div style="font-size:12px;color:#94a3b8;padding:8px;border:1px dashed #e2e8f0;border-radius:6px">No notifications yet. Hit "Fire test job" or trigger from Storefront / Compute Jobs.</div>`}
      </div>`;

    el.innerHTML = topicsHtml + notifsHtml;
  } catch (e) {
    el.innerHTML = `<div style="color:#dc2626;font-size:13px">Error: ${e.message}</div>`;
  }
}
function switchLog(service) {
  _currentLogService = service;
  document.querySelectorAll('#log-tabs .tab').forEach(t => t.classList.remove('active'));
  event.target.classList.add('active');
  disconnectLog();
  document.getElementById('log-pane').textContent = `[Selected: ${service} — click Connect to stream]`;
  document.getElementById('log-status').textContent = '';
}

function connectLog() {
  disconnectLog();
  const pane = document.getElementById('log-pane');
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = `${proto}//${location.host}/ws/logs/${_currentLogService}`;
  document.getElementById('log-status').textContent = 'Connecting…';
  _ws = new WebSocket(url);
  _ws.onopen = () => {
    document.getElementById('log-status').textContent = `Connected to ${_currentLogService}`;
    pane.textContent = '';
  };
  _ws.onmessage = (ev) => {
    if (_logPaused) return;
    pane.textContent += ev.data;
    pane.scrollTop = pane.scrollHeight;
  };
  _ws.onclose = () => {
    document.getElementById('log-status').textContent = 'Disconnected';
  };
  _ws.onerror = () => {
    document.getElementById('log-status').textContent = 'Connection error';
  };
}

function disconnectLog() {
  if (_ws) { _ws.close(); _ws = null; }
}

function toggleLogPause() {
  _logPaused = !_logPaused;
  document.getElementById('log-pause-btn').textContent = _logPaused ? '▶ Resume' : '⏸ Pause';
}

function clearLogs() {
  document.getElementById('log-pane').textContent = '';
}
async function loadStorefront() {
  await loadSfUsers();
  await loadSfProducts();
}

async function loadSfUsers() {
  const sel = document.getElementById('sf-user');
  try {
    const data = await apiFetch('/api/users?limit=200');
    const users = data.users || [];
    sel.innerHTML = '<option value="">— Select a user —</option>' +
      users.map(u => `<option value="${u.user_id}">${u.name} (${u.user_id})</option>`).join('');
  } catch (e) {
    sel.innerHTML = '<option value="">Error loading users</option>';
  }
}

async function sfUserChanged() {
  const sel = document.getElementById('sf-user');
  _selectedUserId = sel.value ? parseInt(sel.value) : null;
  const info = document.getElementById('sf-user-info');
  if (!_selectedUserId) { info.textContent = ''; document.getElementById('sf-recs').innerHTML = '<div style="color:#94a3b8;font-size:13px">Select a user first</div>'; return; }
  const opt = sel.options[sel.selectedIndex];
  info.textContent = opt.text;
  await sfLoadRecs();
  await sfLoadRecent();
}

async function sfLoadRecs() {
  if (!_selectedUserId) return;
  const el = document.getElementById('sf-recs');
  el.innerHTML = '<div class="loader"><div class="spinner"></div>Loading recommendations…</div>';
  try {
    const data = await apiFetch(`/api/users/${_selectedUserId}/recommendations`);
    const recs = data.recommendations || [];
    if (!recs.length) {
      el.innerHTML = '<div style="color:#94a3b8;font-size:13px">No recommendations yet. Interact with some products first.</div>';
      return;
    }
    el.innerHTML = recs.map(r => `
      <div class="rec-card" onclick="openProduct('${r.product_id}', '${esc(r.name)}', '${esc(r.category)}', '${esc(r.metadata || '')}')">
        <div>
          <div class="rc-cat">${r.category}</div>
          <div class="rc-name">${r.name}</div>
          <div class="rc-score">Score: ${r.score}</div>
        </div>
      </div>`).join('');
  } catch (e) {
    el.innerHTML = `<div style="color:#dc2626;font-size:13px">Error: ${e.message}</div>`;
  }
}

async function sfLoadRecent() {
  if (!_selectedUserId) return;
  const el = document.getElementById('sf-recent-activity');
  try {
    const data = await apiFetch(`/api/users/${_selectedUserId}/interactions`);
    const items = (data.interactions || []).slice(0, 5);
    if (!items.length) { el.innerHTML = '<div style="color:#94a3b8">No activity yet.</div>'; return; }
    el.innerHTML = items.map(i => `
      <div style="display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid #f1f5f9">
        <div>
          <div style="font-size:12px;font-weight:500">${i.name}</div>
          <div style="font-size:11px;color:#64748b">${i.interaction_type}</div>
        </div>
        <div style="font-size:11px;color:#94a3b8">${fmtDate(i.created_at)}</div>
      </div>`).join('');
  } catch (e) {
    el.innerHTML = `<div style="color:#dc2626;font-size:13px">Error: ${e.message}</div>`;
  }
}

async function sfTriggerJob() {
  if (!_selectedUserId) { alert('Select a user first'); return; }
  const el = document.getElementById('sf-trigger-status');
  el.textContent = 'Triggering compute job…';
  try {
    const data = await apiFetch(`/api/trigger/${_selectedUserId}`, {method: 'POST'});
    el.textContent = 'Job dispatched. Check Compute Jobs tab.';
  } catch (e) {
    el.textContent = 'Error: ' + e.message;
  }
}

function sfSearch() {
  clearTimeout(_sfSearchTimer);
  _sfOffset = 0;
  _sfSearchTimer = setTimeout(() => loadSfProducts(), 300);
}

function sfCategoryChange() {
  _sfOffset = 0;
  loadSfProducts();
}

function sfPage(dir) {
  const newOff = _sfOffset + dir * SF_PAGE;
  if (newOff < 0 || newOff >= _sfTotal) return;
  _sfOffset = newOff;
  loadSfProducts();
}

async function loadSfProducts() {
  const el = document.getElementById('sf-products');
  el.innerHTML = '<div class="loader"><div class="spinner"></div>Loading…</div>';
  const search = document.getElementById('sf-search').value;
  const category = document.getElementById('sf-category').value;
  try {
    const data = await apiFetch(`/api/products?limit=${SF_PAGE}&offset=${_sfOffset}&search=${encodeURIComponent(search)}&category=${encodeURIComponent(category)}`);
    _sfTotal = data.total || 0;
    const products = data.products || [];

    // populate category filter
    const catSel = document.getElementById('sf-category');
    if (catSel.options.length <= 1 && data.categories) {
      data.categories.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c; opt.textContent = c;
        catSel.appendChild(opt);
      });
    }

    if (!products.length) {
      el.innerHTML = '<div style="color:#94a3b8;font-size:13px;padding:24px 0;text-align:center">No products found.</div>';
    } else {
      el.innerHTML = products.map(p => `
        <div class="product-card" onclick="openProduct('${p.product_id}', '${esc(p.name)}', '${esc(p.category)}', '${esc(p.metadata || '')}')">
          <div class="cat">${p.category}</div>
          <div class="name">${p.name}</div>
          <div class="meta" style="font-size:11px;color:#94a3b8">${p.product_id.slice(0,8)}…</div>
        </div>`).join('');
    }

    const start = _sfOffset + 1;
    const end = Math.min(_sfOffset + SF_PAGE, _sfTotal);
    document.getElementById('sf-page-info').textContent = `${start}–${end} of ${_sfTotal}`;
    document.getElementById('sf-prev').disabled = _sfOffset === 0;
    document.getElementById('sf-next').disabled = _sfOffset + SF_PAGE >= _sfTotal;
  } catch (e) {
    el.innerHTML = `<div style="color:#dc2626;font-size:13px">Error: ${e.message}</div>`;
  }
}

function openProduct(productId, name, category, meta) {
  _selectedProduct = {product_id: productId, name, category, meta};
  document.getElementById('modal-product-name').textContent = name;
  document.getElementById('modal-cat').textContent = category;
  document.getElementById('modal-meta').textContent = !meta ? '—' : (typeof meta === 'object' ? JSON.stringify(meta) : meta);
  document.getElementById('modal-interact-msg').textContent = '';
  document.getElementById('modal-also').innerHTML = '';
  document.getElementById('product-modal').style.display = 'flex';
  loadAlsoViewed(productId);
}

async function loadAlsoViewed(productId) {
  const el = document.getElementById('modal-also');
  try {
    const data = await apiFetch(`/api/products/${productId}`);
    const also = data.also_viewed || [];
    if (!also.length) return;
    el.innerHTML = '<div style="font-size:12px;font-weight:600;color:#475569;margin-bottom:8px">Customers also viewed</div>' +
      also.map(p => `<div style="font-size:13px;color:#0f172a;padding:3px 0;border-bottom:1px solid #f1f5f9">${p.name} <span style="color:#94a3b8;font-size:11px">${p.category}</span></div>`).join('');
  } catch(e) {}
}

async function doInteract(type) {
  if (!_selectedUserId) { alert('Select a user in the Storefront first'); return; }
  if (!_selectedProduct) return;
  const el = document.getElementById('modal-interact-msg');
  el.style.color = '#64748b';
  el.textContent = 'Recording…';
  try {
    await apiFetch('/api/interact', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        user_id: _selectedUserId,
        product_id: _selectedProduct.product_id,
        interaction_type: type,
      })
    });
    el.style.color = '#16a34a';
    el.textContent = `✓ ${type} recorded — dispatching compute job…`;
    sfLoadRecs();
    sfLoadRecent();
    // Auto-dispatch compute job so DynamoDB always gets a record
    try {
      await apiFetch(`/api/trigger/${_selectedUserId}`, {method: 'POST'});
      el.textContent = `✓ ${type} recorded — compute job queued`;
      document.getElementById('sf-trigger-status').textContent = 'Job dispatched — check Compute Jobs tab';
    } catch (_) {
      el.textContent = `✓ ${type} recorded`;
    }
  } catch (e) {
    el.style.color = '#dc2626';
    el.textContent = 'Error: ' + e.message;
  }
}

function closeProductModal() {
  document.getElementById('product-modal').style.display = 'none';
}

function closeModal(event) {
  if (event.target === event.currentTarget) event.currentTarget.style.display = 'none';
}
function prodSearch() {
  clearTimeout(_prodSearchTimer);
  _prodOffset = 0;
  _prodSearchTimer = setTimeout(() => loadProducts(), 300);
}

function prodPage(dir) {
  const newOff = _prodOffset + dir * PROD_PAGE;
  if (newOff < 0 || newOff >= _prodTotal) return;
  _prodOffset = newOff;
  loadProducts();
}

async function loadProducts() {
  const el = document.getElementById('prod-table');
  el.innerHTML = '<div class="loader"><div class="spinner"></div>Loading…</div>';
  const search = document.getElementById('prod-search').value;
  const category = document.getElementById('prod-cat-filter').value;
  try {
    const data = await apiFetch(`/api/products?limit=${PROD_PAGE}&offset=${_prodOffset}&search=${encodeURIComponent(search)}&category=${encodeURIComponent(category)}`);
    _prodTotal = data.total || 0;
    const products = data.products || [];

    // populate filter
    const catSel = document.getElementById('prod-cat-filter');
    if (catSel.options.length <= 1 && data.categories) {
      data.categories.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c; opt.textContent = c;
        catSel.appendChild(opt);
      });
    }

    document.getElementById('prod-count').textContent = `${_prodTotal} products`;
    if (!products.length) {
      el.innerHTML = '<div style="color:#94a3b8;font-size:13px;padding:12px 0">No products found.</div>';
    } else {
      el.innerHTML = `<table class="tbl">
        <thead><tr><th>ID</th><th>Name</th><th>Category</th><th>Metadata</th></tr></thead>
        <tbody>${products.map(p => `<tr>
          <td style="font-size:11px;font-family:monospace;color:#94a3b8">${p.product_id.slice(0,12)}…</td>
          <td>${p.name}</td>
          <td><span class="pill info">${p.category}</span></td>
          <td style="font-size:12px;color:#64748b;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${p.metadata || '—'}</td>
        </tr>`).join('')}</tbody>
      </table>`;
    }
    const start = _prodOffset + 1;
    const end = Math.min(_prodOffset + PROD_PAGE, _prodTotal);
    document.getElementById('prod-page-info').textContent = `${start}–${end} of ${_prodTotal}`;
    document.getElementById('prod-prev').disabled = _prodOffset === 0;
    document.getElementById('prod-next').disabled = _prodOffset + PROD_PAGE >= _prodTotal;
  } catch (e) {
    el.innerHTML = `<div style="color:#dc2626;font-size:13px">Error: ${e.message}</div>`;
  }
}
function usersSearch() {
  clearTimeout(_usersSearchTimer);
  _usersOffset = 0;
  _usersSearchTimer = setTimeout(() => loadUsers(), 300);
}

function usersPage(dir) {
  const newOff = _usersOffset + dir * USERS_PAGE;
  if (newOff < 0 || newOff >= _usersTotal) return;
  _usersOffset = newOff;
  loadUsers();
}

async function loadUsers() {
  const el = document.getElementById('users-table');
  el.innerHTML = '<div class="loader"><div class="spinner"></div>Loading…</div>';
  const search = document.getElementById('users-search').value;
  try {
    const data = await apiFetch(`/api/users?limit=${USERS_PAGE}&offset=${_usersOffset}&search=${encodeURIComponent(search)}`);
    _usersTotal = data.total || 0;
    const users = data.users || [];
    document.getElementById('users-count').textContent = `${_usersTotal} users`;
    if (!users.length) {
      el.innerHTML = '<div style="color:#94a3b8;font-size:13px;padding:12px 0">No users found.</div>';
    } else {
      el.innerHTML = `<table class="tbl">
        <thead><tr><th>ID</th><th>Name</th><th>Email</th><th>Interactions</th><th>Actions</th></tr></thead>
        <tbody>${users.map(u => `<tr>
          <td>${u.user_id}</td>
          <td>${u.name}</td>
          <td style="font-size:12px;color:#64748b">${u.email}</td>
          <td><span class="pill info">${u.interaction_count}</span></td>
          <td style="display:flex;gap:4px">
            <button class="btn btn-outline btn-xs" onclick="viewUserHistory(${u.user_id},'${esc(u.name)}')">History</button>
            <button class="btn btn-primary btn-xs" onclick="shopAsUser(${u.user_id})">Shop</button>
          </td>
        </tr>`).join('')}</tbody>
      </table>`;
    }
    const start = _usersOffset + 1;
    const end = Math.min(_usersOffset + USERS_PAGE, _usersTotal);
    document.getElementById('users-page-info').textContent = `${start}–${end} of ${_usersTotal}`;
    document.getElementById('users-prev').disabled = _usersOffset === 0;
    document.getElementById('users-next').disabled = _usersOffset + USERS_PAGE >= _usersTotal;
  } catch (e) {
    el.innerHTML = `<div style="color:#dc2626;font-size:13px">Error: ${e.message}</div>`;
  }
}

async function viewUserHistory(userId, name) {
  document.getElementById('um-title').textContent = `${name} — Interaction History`;
  document.getElementById('um-body').innerHTML = '<div class="loader"><div class="spinner"></div>Loading…</div>';
  document.getElementById('user-modal').style.display = 'flex';
  try {
    const data = await apiFetch(`/api/users/${userId}/interactions`);
    const items = data.interactions || [];
    if (!items.length) {
      document.getElementById('um-body').innerHTML = '<div style="color:#94a3b8">No interactions yet.</div>';
      return;
    }
    document.getElementById('um-body').innerHTML = `<table class="tbl">
      <thead><tr><th>Product</th><th>Category</th><th>Type</th><th>Date</th></tr></thead>
      <tbody>${items.map(i => `<tr>
        <td>${i.name}</td>
        <td>${i.category}</td>
        <td>${statusPill(i.interaction_type)}</td>
        <td style="font-size:12px;color:#64748b">${fmtDate(i.created_at)}</td>
      </tr>`).join('')}</tbody>
    </table>`;
  } catch (e) {
    document.getElementById('um-body').innerHTML = `<div style="color:#dc2626">Error: ${e.message}</div>`;
  }
}

function shopAsUser(userId) {
  const sel = document.getElementById('sf-user');
  sel.value = userId;
  _selectedUserId = userId;
  showSection('storefront');
  sfUserChanged();
}
async function loadEngine() {
  const el = document.getElementById('active-engine-display');
  el.textContent = '…';
  try {
    const data = await apiFetch('/api/live-engine');
    el.textContent = data.engine || data.active_engine || '—';
  } catch (e) {
    el.textContent = 'Error';
  }
}

async function setEngine(engine) {
  const el = document.getElementById('engine-msg');
  el.textContent = `Switching to ${engine}…`;
  try {
    const data = await apiFetch('/api/live-engine', {
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({engine})
    });
    el.style.color = '#16a34a';
    el.textContent = `✓ Engine switched to ${engine}`;
    loadEngine();
  } catch (e) {
    el.style.color = '#dc2626';
    el.textContent = 'Error: ' + e.message;
  }
}
async function runHpc() {
  const btn = document.getElementById('hpc-run-btn');
  const el = document.getElementById('hpc-results');
  const engines = [...document.querySelectorAll('.hpc-eng:checked')].map(e => e.value);
  if (!engines.length) { alert('Select at least one engine'); return; }

  btn.disabled = true;
  btn.textContent = '⏳ Running…';
  el.innerHTML = '<div class="loader"><div class="spinner"></div>Analyzing…</div>';

  const payload = {
    dataset_size_gb: parseFloat(document.getElementById('hpc-size').value),
    problem_type: document.getElementById('hpc-problem').value,
    node_count: parseInt(document.getElementById('hpc-nodes').value),
    precision: document.getElementById('hpc-precision').value,
    iterations: parseInt(document.getElementById('hpc-iters').value),
    memory_bound: document.getElementById('hpc-membound').checked,
    latency_sensitive: document.getElementById('hpc-latency').checked,
    selected_engines: engines,
  };

  try {
    const data = await apiFetch('/api/hpc/analyze', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });

    const results = data.results || [];
    const recommended = data.recommended_engine || '';
    const summary = data.summary || '';

    el.innerHTML = `
      <div style="margin-bottom:12px">
        <span class="pill ok">Recommended: ${recommended.toUpperCase()}</span>
      </div>
      <div style="font-size:13px;color:#475569;margin-bottom:16px">${summary}</div>
      <table class="tbl">
        <thead><tr>
          <th>Engine</th>
          <th>Throughput</th>
          <th>Latency</th>
          <th>Efficiency</th>
          <th>Memory</th>
          <th>Score</th>
        </tr></thead>
        <tbody>${results.map(r => `<tr ${r.engine_id === recommended ? 'style="background:#f0fdf4"' : ''}>
          <td><strong>${r.engine_name || r.engine_id}</strong></td>
          <td>${fmtNum(r.throughput_gflops)} GFLOPS</td>
          <td>${fmtNum(r.latency_ms)} ms</td>
          <td>${fmtNum(r.efficiency_percent)}%</td>
          <td>${fmtNum(r.memory_bandwidth_gbps)} GB/s</td>
          <td><strong>${fmtNum(r.composite_score)}</strong></td>
        </tr>`).join('')}</tbody>
      </table>
    `;
  } catch (e) {
    el.innerHTML = `<div style="color:#dc2626;font-size:13px">Error: ${e.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = '▶ Run Analysis';
  }
}
async function apiFetch(url, opts = {}) {
  const resp = await fetch(url, opts);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`HTTP ${resp.status}: ${text}`);
  }
  return resp.json();
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val === undefined || val === null ? '—' : val;
}

function fmt(n) {
  if (n === undefined || n === null || n < 0) return '—';
  return n.toLocaleString();
}

function fmtNum(n) {
  if (n === undefined || n === null) return '—';
  return parseFloat(n).toFixed(2);
}

function fmtBytes(bytes) {
  if (!bytes) return '0 B';
  const k = 1024;
  const sizes = ['B','KB','MB','GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return (bytes / Math.pow(k, i)).toFixed(1) + ' ' + sizes[i];
}

function fmtDate(s) {
  if (!s) return '—';
  try {
    return new Date(s).toLocaleString('en-US', {month:'short',day:'numeric',hour:'2-digit',minute:'2-digit',hour12:false});
  } catch { return s; }
}

function esc(s) {
  if (s === null || s === undefined) return '';
  const str = typeof s === 'string' ? s : JSON.stringify(s);
  return str.replace(/'/g, "\\'").replace(/"/g, '&quot;');
}

function statusPill(s) {
  if (!s) return '—';
  const cls = {
    ok:'ok', completed:'ok', success:'ok',
    error:'err', failed:'err',
    pending:'warn', processing:'info', running:'info',
    view:'info', like:'ok', purchase:'ok',
  };
  const c = cls[s.toLowerCase()] || 'info';
  return `<span class="pill ${c}">${s}</span>`;
}
loadOverview();
</script>
</body>
</html>"""
