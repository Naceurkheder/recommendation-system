#!/usr/bin/env bash
set -uo pipefail

API="http://localhost:8000"
DB_DSN="host=localhost port=5432 dbname=recsys_db user=recsys_admin password=secure_password"

MAX_USERS=10
MAX_PRODUCTS=6
WAIT_FOR_JOBS=true
COMPLETED=0
FAILED_JOBS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --users)     MAX_USERS="$2";    shift 2 ;;
    --products)  MAX_PRODUCTS="$2"; shift 2 ;;
    --no-wait)   WAIT_FOR_JOBS=false; shift ;;
    *) shift ;;
  esac
done

log()  { printf "[%s] %s\n" "$(date '+%H:%M:%S')" "$*"; }
fail() { printf "[%s] ERROR: %s\n" "$(date '+%H:%M:%S')" "$*" >&2; }

log "Batch run: users=$MAX_USERS products=$MAX_PRODUCTS wait=$WAIT_FOR_JOBS"

log "Checking API"
for i in $(seq 1 10); do
  HTTP=$(curl -so /dev/null -w "%{http_code}" "$API/health" 2>/dev/null || echo "000")
  [ "$HTTP" = "200" ] && break
  sleep 3
done
[ "$HTTP" != "200" ] && { fail "API unreachable"; exit 1; }

log "Loading users"
USERS=$(python3 -c "
import psycopg2, json
conn = psycopg2.connect('$DB_DSN')
cur = conn.cursor()
cur.execute('SELECT user_id FROM users ORDER BY user_id LIMIT $MAX_USERS')
print(json.dumps([r[0] for r in cur.fetchall()]))
conn.close()
" 2>/dev/null)
USER_COUNT=$(echo "$USERS" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))")

log "Loading products"
PRODUCTS=$(python3 -c "
import psycopg2, json
conn = psycopg2.connect('$DB_DSN')
cur = conn.cursor()
cur.execute('SELECT product_id::text, name, category FROM products ORDER BY RANDOM() LIMIT $MAX_PRODUCTS')
rows = cur.fetchall()
conn.close()
print(json.dumps([{'id': r[0], 'name': r[1], 'cat': r[2]} for r in rows]))
" 2>/dev/null)
PROD_COUNT=$(echo "$PRODUCTS" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))")

log "Loaded $USER_COUNT users, $PROD_COUNT products"

TYPES=("view" "like" "purchase" "view" "like")
TOTAL_INTERACTIONS=0
FAILED_INTERACTIONS=0

log "Recording interactions"
for ui in $(echo "$USERS" | python3 -c "import sys,json; [print(u) for u in json.load(sys.stdin)]"); do
  PROD_IDX=0
  while IFS=$'\t' read -r PID PNAME PCAT; do
    [ -z "$PID" ] && continue
    TYPE_IDX=$(( (ui + PROD_IDX) % 5 ))
    ITYPE="${TYPES[$TYPE_IDX]}"

    RESULT=$(curl -s -X POST "$API/interactions" \
      -H "Content-Type: application/json" \
      -d "{\"user_id\": $ui, \"product_id\": \"$PID\", \"interaction_type\": \"$ITYPE\"}" 2>/dev/null)
    STATUS=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','error'))" 2>/dev/null)

    if [ "$STATUS" = "created" ]; then
      TOTAL_INTERACTIONS=$((TOTAL_INTERACTIONS + 1))
    else
      FAILED_INTERACTIONS=$((FAILED_INTERACTIONS + 1))
    fi
    PROD_IDX=$((PROD_IDX + 1))
  done < <(echo "$PRODUCTS" | python3 -c "
import sys, json
for p in json.load(sys.stdin):
    print(p['id'] + '\t' + p['name'][:30] + '\t' + p['cat'])
")
done

log "Interactions: $TOTAL_INTERACTIONS recorded, $FAILED_INTERACTIONS failed"

log "Triggering compute jobs"
declare -a JOB_IDS=()
for uid in $(echo "$USERS" | python3 -c "import sys,json; [print(u) for u in json.load(sys.stdin)]"); do
  RESP=$(curl -s "$API/recommendations/$uid?force=true" 2>/dev/null)
  JOB_STATUS=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','?'))" 2>/dev/null)
  JOB_ID=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('job_id',''))" 2>/dev/null)

  if [ "$JOB_STATUS" = "computing" ] && [ -n "$JOB_ID" ]; then
    log "user=$uid job=${JOB_ID:0:8}"
    JOB_IDS+=("$JOB_ID")
  fi
  sleep 0.3
done

log "Dispatched ${#JOB_IDS[@]} jobs"

if [ "$WAIT_FOR_JOBS" = "true" ] && [ "${#JOB_IDS[@]}" -gt 0 ]; then
  log "Polling DynamoDB for completion (max 120s)"
  DEADLINE=$((SECONDS + 120))

  while [ $SECONDS -lt $DEADLINE ]; do
    COMPLETED=0
    FAILED_JOBS=0
    STILL_RUNNING=0

    for jid in "${JOB_IDS[@]}"; do
      STATUS=$(python3 -c "
import boto3
ddb = boto3.client('dynamodb', endpoint_url='http://localhost:4566', region_name='us-east-1',
                   aws_access_key_id='test', aws_secret_access_key='test')
try:
    r = ddb.get_item(TableName='compute-jobs', Key={'job_id': {'S': '$jid'}})
    print(r.get('Item', {}).get('status', {}).get('S', 'not_found'))
except:
    print('error')
" 2>/dev/null)

      case "$STATUS" in
        complete) COMPLETED=$((COMPLETED + 1)) ;;
        failed)   FAILED_JOBS=$((FAILED_JOBS + 1)) ;;
        *)        STILL_RUNNING=$((STILL_RUNNING + 1)) ;;
      esac
    done

    printf "\r  complete=%d failed=%d pending=%d  " "$COMPLETED" "$FAILED_JOBS" "$STILL_RUNNING"

    if [ $((COMPLETED + FAILED_JOBS)) -eq "${#JOB_IDS[@]}" ]; then
      echo ""
      break
    fi
    sleep 4
  done
  echo ""
  log "Jobs: $COMPLETED complete, $FAILED_JOBS failed of ${#JOB_IDS[@]}"
fi

log "Summary"
echo "  Interactions recorded: $TOTAL_INTERACTIONS"
echo "  Compute jobs fired:    ${#JOB_IDS[@]}"
echo "  Jobs completed:        $COMPLETED"
echo "  Dashboard:             http://192.168.56.10:3000"
