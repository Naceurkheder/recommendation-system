#!/usr/bin/env bash
# One-command stack bring-up:
#   1. docker compose up (all services)
#   2. Wait for all healthchecks
#   3. Provision LocalStack (12 AWS services)
#   4. Seed PostgreSQL if empty
#   5. Run integration tests
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/infra/docker/docker-compose.yml"

# ── Helpers ───────────────────────────────────────────────────────────────────
info()  { echo "[$(date +%H:%M:%S)] $*"; }
error() { echo "[$(date +%H:%M:%S)] ERROR: $*" >&2; }

wait_healthy() {
    local service="$1"
    local max_wait="${2:-120}"
    info "Waiting for $service to be healthy..."
    local deadline=$((SECONDS + max_wait))
    while [ $SECONDS -lt $deadline ]; do
        STATUS=$(docker compose -f "$COMPOSE_FILE" ps --format json "$service" 2>/dev/null \
            | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('Health',''))" 2>/dev/null \
            || docker compose -f "$COMPOSE_FILE" ps "$service" 2>/dev/null | grep -o 'healthy' || echo "")
        if echo "$STATUS" | grep -qi "healthy"; then
            info "$service is healthy"
            return 0
        fi
        sleep 3
    done
    error "$service did not become healthy within ${max_wait}s"
    docker compose -f "$COMPOSE_FILE" logs --tail=30 "$service" >&2
    return 1
}

# ── Step 1: Docker Compose up ─────────────────────────────────────────────────
info "Starting all services..."
docker compose -f "$COMPOSE_FILE" up -d --build
info "Containers started"

# ── Step 2: Wait for healthchecks ─────────────────────────────────────────────
wait_healthy localstack 120
wait_healthy db 60
wait_healthy redis 30

# Give the API and worker a moment to initialize
info "Waiting for API to start..."
for i in $(seq 1 40); do
    if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
        info "API is up"
        break
    fi
    [ "$i" -eq 40 ] && { error "API did not start in time"; exit 1; }
    sleep 3
done

# ── Step 3: Provision LocalStack ──────────────────────────────────────────────
info "Provisioning LocalStack AWS services..."
bash "$SCRIPT_DIR/setup_localstack.sh"

# ── Step 4: Seed PostgreSQL (if empty) ────────────────────────────────────────
info "Seeding PostgreSQL..."
if python3 - <<'PYEOF' 2>/dev/null; then
import psycopg2
conn = psycopg2.connect("host=localhost port=5432 dbname=recsys_db user=recsys_admin password=secure_password")
cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM users")
count = cur.fetchone()[0]
conn.close()
exit(0 if count > 0 else 1)
PYEOF
    info "Database already has data — skipping seed"
else
    python3 "$SCRIPT_DIR/seed_data.py"
fi

# ── Step 5: Run integration tests ─────────────────────────────────────────────
info "Running integration tests..."
cd "$PROJECT_ROOT"

# Install test dependencies if not present
python3 -c "import pytest, requests" 2>/dev/null || \
    pip install pytest requests boto3 psycopg2-binary --quiet

pytest tests/test_integration.py -v --tb=short 2>&1
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    info "All tests passed ✓"
else
    error "Some tests failed — check output above"
fi

exit $EXIT_CODE
