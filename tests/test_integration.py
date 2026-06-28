
import json
import os
import time
import uuid
from typing import Optional

import boto3
import pytest
import requests

API_URL      = os.getenv("API_URL",          "http://localhost:8000")
LS_URL       = os.getenv("LOCALSTACK_URL",   "http://localhost:4566")
AWS_REGION   = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
DYNAMO_TABLE = "compute-jobs"
S3_BUCKET    = "similarity-matrices"
LOG_GROUP    = "/app/fastapi"

HEADERS = {"Content-Type": "application/json"}

def _boto(service: str):
    return boto3.client(
        service,
        endpoint_url=LS_URL,
        region_name=AWS_REGION,
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )

def _wait_for_job(job_id: str, timeout: int = 90) -> Optional[str]:

    dynamodb = _boto("dynamodb")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = dynamodb.get_item(
                TableName=DYNAMO_TABLE,
                Key={"job_id": {"S": job_id}},
            )
            item = resp.get("Item", {})
            status = item.get("status", {}).get("S", "unknown")
            if status in ("complete", "failed"):
                return status
        except Exception:
            pass
        time.sleep(3)
    return None

class TestServiceHealth:

    def test_api_health(self):
        resp = requests.get(f"{API_URL}/health", timeout=10)
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_localstack_health(self):
        resp = requests.get(f"{LS_URL}/_localstack/health", timeout=10)
        assert resp.status_code == 200
        health = resp.json()

        services = health.get("services", health)
        for svc in ("s3", "sqs", "dynamodb", "secretsmanager"):
            assert svc in str(services), f"LocalStack service {svc!r} not found"

    def test_s3_bucket_exists(self):
        s3 = _boto("s3")
        buckets = [b["Name"] for b in s3.list_buckets()["Buckets"]]
        assert S3_BUCKET in buckets

    def test_sqs_queue_exists(self):
        sqs = _boto("sqs")
        resp = sqs.get_queue_url(QueueName="compute-jobs.fifo")
        assert "QueueUrl" in resp

    def test_dynamodb_table_exists(self):
        dynamodb = _boto("dynamodb")
        resp = dynamodb.describe_table(TableName=DYNAMO_TABLE)
        assert resp["Table"]["TableStatus"] == "ACTIVE"

    def test_secrets_provisioned(self):
        secrets = _boto("secretsmanager")
        for secret_id in ("db/postgres", "redis/config"):
            resp = secrets.get_secret_value(SecretId=secret_id)
            data = json.loads(resp["SecretString"])
            assert len(data) > 0, f"Secret {secret_id} is empty"

class TestInteractions:

    PRODUCT_A = "550e8400-e29b-41d4-a716-446655440001"
    PRODUCT_B = "550e8400-e29b-41d4-a716-446655440002"
    PRODUCT_C = "550e8400-e29b-41d4-a716-446655440003"

    def _post(self, user_id: int, product_id: str, interaction_type: str) -> requests.Response:
        return requests.post(
            f"{API_URL}/interactions",
            json={"user_id": user_id, "product_id": product_id, "interaction_type": interaction_type},
            headers=HEADERS,
            timeout=10,
        )

    def test_create_interaction_returns_201(self):
        resp = self._post(1, self.PRODUCT_A, "view")
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert "interaction_id" in body
        assert body["status"] == "created"

    def test_multiple_interactions_for_pipeline(self):

        interactions = [
            (1, self.PRODUCT_A, "purchase"),
            (1, self.PRODUCT_B, "view"),
            (2, self.PRODUCT_A, "like"),
            (2, self.PRODUCT_C, "purchase"),
            (3, self.PRODUCT_B, "purchase"),
            (3, self.PRODUCT_C, "like"),
            (4, self.PRODUCT_A, "view"),
            (4, self.PRODUCT_B, "purchase"),
        ]
        for user_id, product_id, itype in interactions:
            resp = self._post(user_id, product_id, itype)
            assert resp.status_code == 201, f"Failed for {user_id}/{product_id}: {resp.text}"

class TestRecommendationPipeline:

    TARGET_USER = 1

    def test_cache_miss_returns_202(self):

        resp = requests.get(f"{API_URL}/recommendations/{self.TARGET_USER}", timeout=10)

        assert resp.status_code in (200, 202), resp.text

    def test_dispatch_and_wait_for_completion(self):

        product_ids = [
            "550e8400-e29b-41d4-a716-446655440011",
            "550e8400-e29b-41d4-a716-446655440012",
        ]
        for pid in product_ids:
            requests.post(
                f"{API_URL}/interactions",
                json={"user_id": self.TARGET_USER, "product_id": pid, "interaction_type": "view"},
                headers=HEADERS,
                timeout=10,
            )

        resp = requests.get(f"{API_URL}/recommendations/{self.TARGET_USER}", timeout=10)
        assert resp.status_code in (200, 202)

        if resp.status_code == 202:
            job_id = resp.json().get("job_id")
            assert job_id, "202 response must include job_id"

            final_status = _wait_for_job(job_id, timeout=90)
            assert final_status == "complete", (
                f"Job {job_id} did not complete within 90s (status={final_status})"
            )

            time.sleep(1)
            recs_resp = requests.get(f"{API_URL}/recommendations/{self.TARGET_USER}", timeout=10)
            assert recs_resp.status_code == 200, recs_resp.text
            body = recs_resp.json()
            assert "similar_users" in body
            assert body.get("cached") is True

    def test_job_status_endpoint(self):

        requests.get(f"{API_URL}/recommendations/{self.TARGET_USER}", timeout=10)
        time.sleep(1)

        resp = requests.get(f"{API_URL}/recommendations/{self.TARGET_USER}/status", timeout=10)
        assert resp.status_code in (200, 404)
        if resp.status_code == 200:
            body = resp.json()
            assert "job_id" in body
            assert "status" in body
            assert body["status"] in ("pending", "running", "complete", "failed")

class TestS3Matrix:

    def test_matrix_file_exists_in_s3(self):

        s3 = _boto("s3")
        resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix="matrices/")
        objects = resp.get("Contents", [])
        assert len(objects) > 0, (
            "No matrix files in s3://similarity-matrices/matrices/ — "
            "ensure the worker has processed at least one job"
        )

        latest = max(objects, key=lambda o: o["LastModified"])
        raw = s3.get_object(Bucket=S3_BUCKET, Key=latest["Key"])["Body"].read()
        data = json.loads(raw)
        assert "shape" in data
        assert "data" in data
        assert len(data["data"]) > 0

class TestCloudWatch:

    def test_log_groups_exist(self):
        logs = _boto("logs")
        for group in ("/app/fastapi", "/app/worker", "/app/lambda"):
            resp = logs.describe_log_groups(logGroupNamePrefix=group)
            found = [g["logGroupName"] for g in resp["logGroups"]]
            assert group in found, f"Log group {group!r} not found in CloudWatch"
