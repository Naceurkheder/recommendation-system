
import os

import psycopg2
import psycopg2.extras
import pytest

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://recsys_admin:secure_password@localhost:5432/recsys_db",
)

_TEST_PRODUCTS = [
    ("550e8400-e29b-41d4-a716-446655440001", "Test Product A", "test"),
    ("550e8400-e29b-41d4-a716-446655440002", "Test Product B", "test"),
    ("550e8400-e29b-41d4-a716-446655440003", "Test Product C", "test"),
    ("550e8400-e29b-41d4-a716-446655440011", "Test Product D", "test"),
    ("550e8400-e29b-41d4-a716-446655440012", "Test Product E", "test"),
]

@pytest.fixture(scope="session", autouse=True)
def seed_test_fixtures() -> None:

    try:
        conn = psycopg2.connect(DB_URL)
    except Exception as exc:
        pytest.skip(f"PostgreSQL not reachable, skipping fixture setup: {exc}")
        return

    try:
        with conn:
            with conn.cursor() as cur:
                psycopg2.extras.execute_batch(
                    cur,
                    """
                    INSERT INTO products (product_id, name, category)
                    VALUES (%s::uuid, %s, %s)
                    ON CONFLICT (product_id) DO NOTHING
                    """,
                    _TEST_PRODUCTS,
                )
    finally:
        conn.close()
