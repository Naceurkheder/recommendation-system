#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/infra/docker/docker-compose.yml"

log()   { echo "[$(date +%H:%M:%S)] $*"; }
fail()  { echo "[$(date +%H:%M:%S)] ERROR: $*" >&2; exit 1; }

if [ ! -f "$PROJECT_ROOT/dist/librec_engine.so" ]; then
    fail "dist/librec_engine.so not found. Run 'bash launch.sh' from the host first."
fi

wait_healthy() {
    local service="$1" max_wait="${2:-120}"
    log "Waiting for $service"
    local deadline=$((SECONDS + max_wait))
    while [ $SECONDS -lt $deadline ]; do
        STATUS=$(docker compose -f "$COMPOSE_FILE" ps "$service" 2>/dev/null \
            | grep -oE 'healthy|unhealthy' | head -1 || echo "")
        if [ "$STATUS" = "healthy" ]; then
            return 0
        fi
        sleep 3
    done
    docker compose -f "$COMPOSE_FILE" logs --tail=30 "$service" >&2
    fail "$service did not become healthy within ${max_wait}s"
}

mkdir -p "$PROJECT_ROOT/data"

log "Starting services"
docker compose -f "$COMPOSE_FILE" up -d --build

wait_healthy localstack 120
wait_healthy db 60
wait_healthy redis 30

log "Waiting for API"
for i in $(seq 1 40); do
    if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
        break
    fi
    [ "$i" -eq 40 ] && fail "API did not start"
    sleep 3
done

log "Provisioning LocalStack"
bash "$SCRIPT_DIR/setup_localstack.sh"

log "Validating LocalStack"
bash "$SCRIPT_DIR/validate_localstack.sh" "http://localhost:4566" || {
    docker compose -f "$COMPOSE_FILE" logs --tail=50 localstack >&2
    fail "LocalStack validation failed"
}

log "Seeding PostgreSQL"
if python3 - <<'PYEOF' 2>/dev/null; then
import psycopg2
conn = psycopg2.connect("host=localhost port=5432 dbname=recsys_db user=recsys_admin password=secure_password")
cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM users")
count = cur.fetchone()[0]
conn.close()
exit(0 if count > 0 else 1)
PYEOF
    log "Database already seeded"
else
    python3 "$SCRIPT_DIR/seed_data.py"
fi

log "Exporting matrix.csv"
export MATRIX_CSV="$PROJECT_ROOT/data/matrix.csv"
python3 <<'PYEOF'
import psycopg2, os, sys
conn = psycopg2.connect("host=localhost port=5432 dbname=recsys_db user=recsys_admin password=secure_password")
cur = conn.cursor()
cur.execute("""
    SELECT user_id, product_id::text,
        CASE interaction_type
            WHEN 'purchase' THEN 1.0
            WHEN 'like'     THEN 0.7
            WHEN 'view'     THEN 0.3
            ELSE 0.1
        END
    FROM interactions ORDER BY user_id
""")
rows = cur.fetchall()
conn.close()
if not rows:
    sys.exit(0)
with open(os.environ["MATRIX_CSV"], "w") as f:
    for uid, pid, rating in rows:
        f.write(f"{uid},{pid},{float(rating):.4f}\n")
PYEOF

log "Restarting API to load matrix"
docker compose -f "$COMPOSE_FILE" restart api
for i in $(seq 1 40); do
    if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
        break
    fi
    [ "$i" -eq 40 ] && fail "API did not restart"
    sleep 3
done

log "Running integration tests"
cd "$PROJECT_ROOT"
python3 -c "import pytest, requests" 2>/dev/null || \
    pip install pytest requests boto3 psycopg2-binary --quiet
pytest tests/test_integration.py -v --tb=short
EXIT_CODE=$?

VM_IP="192.168.56.10"
log "Stack live"
echo "  Platform    http://$VM_IP:3000"
echo "  API         http://$VM_IP:8000"
echo "  LocalStack  http://$VM_IP:4566/_localstack/health"

exit $EXIT_CODE
