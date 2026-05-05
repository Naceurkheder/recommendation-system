#!/usr/bin/env bash
set -euo pipefail

AWS_REGION="us-east-1"
ACCOUNT_ID="000000000000"

if [ -n "${AWS_ENDPOINT_URL:-}" ]; then
    LOCALSTACK_URL="$AWS_ENDPOINT_URL"
elif ping -c1 -W1 localstack &>/dev/null 2>&1; then
    LOCALSTACK_URL="http://localstack:4566"
else
    LOCALSTACK_URL="http://localhost:4566"
fi

if command -v awslocal &>/dev/null; then
    AWSCLI="awslocal"
else
    export AWS_DEFAULT_REGION="$AWS_REGION"
    export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-test}"
    export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-test}"
    AWSCLI="aws --endpoint-url=$LOCALSTACK_URL --region $AWS_REGION"
fi

log() { printf "[%s] %s\n" "$(date +%H:%M:%S)" "$*"; }

log "Provisioning LocalStack at $LOCALSTACK_URL"

log "Waiting for health"
for i in $(seq 1 60); do
    HEALTH=$(curl -sf "$LOCALSTACK_URL/_localstack/health" 2>/dev/null || echo "{}")
    if echo "$HEALTH" | grep -q '"s3"' && echo "$HEALTH" | grep -q '"sqs"'; then
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "ERROR: LocalStack did not become ready in time" >&2
        exit 1
    fi
    sleep 2
done
sleep 2

log "S3: similarity-matrices"
if ! $AWSCLI s3api head-bucket --bucket similarity-matrices 2>/dev/null; then
    $AWSCLI s3api create-bucket --bucket similarity-matrices
    $AWSCLI s3api put-bucket-versioning \
        --bucket similarity-matrices \
        --versioning-configuration Status=Enabled
    $AWSCLI s3api put-bucket-lifecycle-configuration \
        --bucket similarity-matrices \
        --lifecycle-configuration '{
            "Rules": [{
                "ID": "expire-30d",
                "Status": "Enabled",
                "Expiration": {"Days": 30},
                "Filter": {"Prefix": ""}
            }]
        }'
fi

log "SQS: compute-jobs-dlq.fifo"
if ! $AWSCLI sqs get-queue-url --queue-name compute-jobs-dlq.fifo 2>/dev/null; then
    $AWSCLI sqs create-queue \
        --queue-name compute-jobs-dlq.fifo \
        --attributes FifoQueue=true,ContentBasedDeduplication=true >/dev/null
fi
DLQ_ARN="arn:aws:sqs:${AWS_REGION}:${ACCOUNT_ID}:compute-jobs-dlq.fifo"

log "SQS: compute-jobs.fifo"
if ! $AWSCLI sqs get-queue-url --queue-name compute-jobs.fifo 2>/dev/null; then
    $AWSCLI sqs create-queue \
        --queue-name compute-jobs.fifo \
        --attributes FifoQueue=true,ContentBasedDeduplication=true >/dev/null
    QUEUE_URL=$($AWSCLI sqs get-queue-url --queue-name compute-jobs.fifo --query QueueUrl --output text)
    python3 -c "
import json
redrive = json.dumps({'deadLetterTargetArn': '${DLQ_ARN}', 'maxReceiveCount': '3'})
print(json.dumps({'RedrivePolicy': redrive}))
" > /tmp/sqs_attrs.json
    $AWSCLI sqs set-queue-attributes \
        --queue-url "$QUEUE_URL" \
        --attributes file:///tmp/sqs_attrs.json >/dev/null
    rm -f /tmp/sqs_attrs.json
fi

log "SNS: compute-complete"
SNS_ARN=$($AWSCLI sns list-topics \
    --query "Topics[?ends_with(TopicArn,':compute-complete')].TopicArn" \
    --output text 2>/dev/null || echo "")
if [ -z "$SNS_ARN" ] || [ "$SNS_ARN" = "None" ]; then
    SNS_ARN=$($AWSCLI sns create-topic --name compute-complete --query TopicArn --output text)
fi

if ! $AWSCLI sqs get-queue-url --queue-name compute-notifications >/dev/null 2>&1; then
    $AWSCLI sqs create-queue --queue-name compute-notifications \
        --attributes '{"MessageRetentionPeriod":"86400"}' >/dev/null
fi

NOTIF_ARN="arn:aws:sqs:${AWS_REGION}:${ACCOUNT_ID}:compute-notifications"
EXISTING_SUB=$($AWSCLI sns list-subscriptions-by-topic --topic-arn "$SNS_ARN" \
    --query "Subscriptions[?Endpoint=='${NOTIF_ARN}'].SubscriptionArn" \
    --output text 2>/dev/null || echo "")
if [ -z "$EXISTING_SUB" ] || [ "$EXISTING_SUB" = "None" ]; then
    $AWSCLI sns subscribe \
        --topic-arn "$SNS_ARN" \
        --protocol sqs \
        --notification-endpoint "$NOTIF_ARN" \
        --attributes RawMessageDelivery=true >/dev/null
fi

log "DynamoDB: compute-jobs"
if ! $AWSCLI dynamodb describe-table --table-name compute-jobs 2>/dev/null; then
    $AWSCLI dynamodb create-table \
        --table-name compute-jobs \
        --attribute-definitions \
            AttributeName=job_id,AttributeType=S \
            AttributeName=user_id,AttributeType=S \
            AttributeName=status,AttributeType=S \
        --key-schema AttributeName=job_id,KeyType=HASH \
        --global-secondary-indexes '[
            {
                "IndexName": "user_id-index",
                "KeySchema": [{"AttributeName": "user_id", "KeyType": "HASH"}],
                "Projection": {"ProjectionType": "ALL"}
            },
            {
                "IndexName": "status-index",
                "KeySchema": [{"AttributeName": "status", "KeyType": "HASH"}],
                "Projection": {"ProjectionType": "ALL"}
            }
        ]' \
        --billing-mode PAY_PER_REQUEST >/dev/null
    $AWSCLI dynamodb update-time-to-live \
        --table-name compute-jobs \
        --time-to-live-specification Enabled=true,AttributeName=expires_at >/dev/null
fi

log "Secrets Manager: db/postgres, redis/config"
PG_SECRET='{"host":"db","port":5432,"database":"recsys_db","username":"recsys_admin","password":"secure_password"}'
REDIS_SECRET='{"host":"redis","port":6379,"password":""}'

for secret_name in "db/postgres" "redis/config"; do
    if ! $AWSCLI secretsmanager describe-secret --secret-id "$secret_name" 2>/dev/null; then
        if [ "$secret_name" = "db/postgres" ]; then
            VALUE="$PG_SECRET"
        else
            VALUE="$REDIS_SECRET"
        fi
        $AWSCLI secretsmanager create-secret \
            --name "$secret_name" \
            --secret-string "$VALUE" >/dev/null
    fi
done

log "CloudWatch: log groups"
for log_group in "/app/fastapi" "/app/worker"; do
    EXISTING=$($AWSCLI logs describe-log-groups \
        --log-group-name-prefix "$log_group" \
        --query "logGroups[?logGroupName=='${log_group}'].logGroupName" \
        --output text 2>/dev/null || echo "")
    if [ -z "$EXISTING" ] || [ "$EXISTING" = "None" ]; then
        $AWSCLI logs create-log-group --log-group-name "$log_group" >/dev/null
    fi
done

log "EventBridge: periodic-recompute"
if ! $AWSCLI events describe-rule --name periodic-recompute 2>/dev/null; then
    $AWSCLI events put-rule \
        --name periodic-recompute \
        --schedule-expression "rate(6 hours)" \
        --state ENABLED >/dev/null
    QUEUE_ARN="arn:aws:sqs:${AWS_REGION}:${ACCOUNT_ID}:compute-jobs.fifo"
    $AWSCLI events put-targets \
        --rule periodic-recompute \
        --targets "[{
            \"Id\": \"sqs-target\",
            \"Arn\": \"${QUEUE_ARN}\",
            \"SqsParameters\": {\"MessageGroupId\": \"periodic\"},
            \"Input\": \"{\\\"job_id\\\":\\\"periodic-trigger\\\",\\\"user_id\\\":\\\"all\\\",\\\"trigger\\\":\\\"eventbridge\\\"}\"
        }]" >/dev/null
fi

log "Provisioning complete"
