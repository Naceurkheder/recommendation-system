"""
Seed PostgreSQL with sample users, products, and interactions.
Run after docker compose up + setup_localstack.sh, only when tables are empty.

Usage:
    python scripts/seed_data.py
    DATABASE_URL=postgresql://... python scripts/seed_data.py
"""

import os
import sys
import uuid
import random
from typing import List, Tuple

import psycopg2
import psycopg2.extras

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://recsys_admin:secure_password@localhost:5432/recsys_db",
)

INTERACTION_TYPES = ["view", "like", "purchase"]
WEIGHTS = [0.5, 0.3, 0.2]  # view is most common

NUM_USERS    = 20
NUM_PRODUCTS = 15
NUM_INTERACTIONS = 80

SEED_USERS: List[Tuple[str, str]] = [
    (f"User {i}", f"user{i}@example.com") for i in range(1, NUM_USERS + 1)
]


def tables_empty(cur) -> bool:
    cur.execute("SELECT COUNT(*) FROM users")
    return cur.fetchone()[0] == 0


def seed(conn) -> None:
    with conn.cursor() as cur:
        if not tables_empty(cur):
            print("Tables already have data — skipping seed")
            return

        # Users
        user_ids = []
        for name, email in SEED_USERS:
            cur.execute(
                "INSERT INTO users (name, email) VALUES (%s, %s) RETURNING user_id",
                (name, email),
            )
            user_ids.append(cur.fetchone()[0])
        print(f"Inserted {len(user_ids)} users")

        # Products
        product_ids = []
        categories = ["electronics", "books", "clothing", "food", "sports"]
        for i in range(NUM_PRODUCTS):
            cat = categories[i % len(categories)]
            cur.execute(
                "INSERT INTO products (name, category, metadata) VALUES (%s, %s, %s) RETURNING product_id",
                (f"Product {i+1}", cat, psycopg2.extras.Json({"index": i})),
            )
            product_ids.append(str(cur.fetchone()[0]))
        print(f"Inserted {len(product_ids)} products")

        # Interactions
        rng = random.Random(42)
        interaction_data = []
        for _ in range(NUM_INTERACTIONS):
            uid  = rng.choice(user_ids)
            pid  = rng.choice(product_ids)
            itype = rng.choices(INTERACTION_TYPES, WEIGHTS)[0]
            interaction_data.append((uid, pid, itype))

        psycopg2.extras.execute_batch(
            cur,
            "INSERT INTO interactions (user_id, product_id, interaction_type) VALUES (%s, %s::uuid, %s)",
            interaction_data,
        )
        print(f"Inserted {len(interaction_data)} interactions")

    conn.commit()
    print("Seed complete")


if __name__ == "__main__":
    try:
        conn = psycopg2.connect(DATABASE_URL)
        seed(conn)
        conn.close()
    except Exception as exc:
        print(f"Seed failed: {exc}", file=sys.stderr)
        sys.exit(1)
