"""
Session-scoped fixtures for integration tests.
Inserts the fixed-UUID products and users used by TestInteractions so the
foreign-key constraints on the interactions table are satisfied.
"""

import os

import psycopg2
import psycopg2.extras
import pytest

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://recsys_admin:secure_password@localhost:5432/recsys_db",
)

# UUIDs must match the hardcoded values in test_integration.py
_TEST_PRODUCTS = [
    ("550e8400-e29b-41d4-a716-446655440001", "Test Product A", "test"),
    ("550e8400-e29b-41d4-a716-446655440002", "Test Product B", "test"),
    ("550e8400-e29b-41d4-a716-446655440003", "Test Product C", "test"),
    ("550e8400-e29b-41d4-a716-446655440011", "Test Product D", "test"),
    ("550e8400-e29b-41d4-a716-446655440012", "Test Product E", "test"),
]

# user_id values used in tests; seed_data.py creates users 1-100 so these
# will exist after seeding, but we ensure them here for a clean DB too.
_TEST_USERS = [
    ("Test User 1", "testuser1_fixture@example.com"),
    ("Test User 2", "testuser2_fixture@example.com"),
    ("Test User 3", "testuser3_fixture@example.com"),
    ("Test User 4", "testuser4_fixture@example.com"),
]


@pytest.fixture(scope="session", autouse=True)
def seed_test_fixtures() -> None:
    """
    Ensure test products (known UUIDs) and at least 4 users exist in the DB.
    Idempotent — uses ON CONFLICT DO NOTHING so repeated runs are safe.
    Skips if the database is unreachable (tests will fail on their own terms).
    """
    try:
        conn = psycopg2.connect(DB_URL)
    except Exception as exc:
        pytest.skip(f"PostgreSQL not reachable, skipping fixture setup: {exc}")
        return

    try:
        with conn:
            with conn.cursor() as cur:
                # Users (GENERATED ALWAYS AS IDENTITY — insert without specifying user_id)
                for name, email in _TEST_USERS:
                    cur.execute(
                        """
                        INSERT INTO users (name, email)
                        VALUES (%s, %s)
                        ON CONFLICT (email) DO NOTHING
                        """,
                        (name, email),
                    )

                # Products with fixed UUIDs
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
