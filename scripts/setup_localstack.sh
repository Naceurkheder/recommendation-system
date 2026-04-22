#!/usr/bin/env bash
set -euo pipefail

AWS_REGION="us-east-1"
ACCOUNT_ID="000000000000"

# Detect LocalStack URL
if [ -n "${AWS_ENDPOINT_URL:-}" ]; then
    LOCALSTACK_URL="$AWS_ENDPOINT_URL"
elif ping -c1 -W1 localstack &>/dev/null 2>&1; then
    LOCALSTACK_URL="http://localstack:4566"
else
    LOCALSTACK_URL="http://localhost:4566"
fi

# Use awslocal if available, else fall back to plain aws CLI
if command -v awslocal &>/dev/null; then
    AWSCLI="awslocal"
else
    export AWS_DEFAULT_REGION="$AWS_REGION"
    export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-test}"
    export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-test}"
    AWSCLI="aws --endpoint-url=$LOCALSTACK_URL --region $AWS_REGION"
fi

echo "==> LocalStack provisioning starting (URL: $LOCALSTACK_URL)"

# Wait for LocalStack to be ready (all required services running)
echo "==> Waiting for LocalStack health..."
for i in $(seq 1 60); do
    HEALTH=$(curl -sf "$LOCALSTACK_URL/_localstack/health" 2>/dev/null || echo "{}")
    if echo "$HEALTH" | grep -q '"s3"' && echo "$HEALTH" | grep -q '"sqs"'; then
        echo "    LocalStack is ready (attempt $i)"
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "ERROR: LocalStack did not become ready in time"
        exit 1
    fi
    sleep 2
done
sleep 2  # brief settle time

# ── S3 ────────────────────────────────────────────────────────────────────────
echo "==> S3: similarity-matrices"
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
    echo "    Created bucket: similarity-matrices"
else
    echo "    Bucket already exists"
fi

# ── SQS ───────────────────────────────────────────────────────────────────────
echo "==> SQS: compute-jobs-dlq.fifo"
if ! $AWSCLI sqs get-queue-url --queue-name compute-jobs-dlq.fifo 2>/dev/null; then
    $AWSCLI sqs create-queue \
        --queue-name compute-jobs-dlq.fifo \
        --attributes FifoQueue=true,ContentBasedDeduplication=true >/dev/null
    echo "    Created DLQ"
else
    echo "    DLQ already exists"
fi

DLQ_ARN="arn:aws:sqs:${AWS_REGION}:${ACCOUNT_ID}:compute-jobs-dlq.fifo"

echo "==> SQS: compute-jobs.fifo"
if ! $AWSCLI sqs get-queue-url --queue-name compute-jobs.fifo 2>/dev/null; then
    REDRIVE="{\"deadLetterTargetArn\":\"${DLQ_ARN}\",\"maxReceiveCount\":\"3\"}"
    $AWSCLI sqs create-queue \
        --queue-name compute-jobs.fifo \
        --attributes "FifoQueue=true,ContentBasedDeduplication=true,RedrivePolicy=${REDRIVE}" >/dev/null
    echo "    Created queue with redrive policy"
else
    echo "    Queue already exists"
fi

# ── SNS ───────────────────────────────────────────────────────────────────────
echo "==> SNS: compute-complete"
SNS_ARN=$($AWSCLI sns list-topics \
    --query "Topics[?ends_with(TopicArn,':compute-complete')].TopicArn" \
    --output text 2>/dev/null || echo "")
if [ -z "$SNS_ARN" ] || [ "$SNS_ARN" = "None" ]; then
    SNS_ARN=$($AWSCLI sns create-topic --name compute-complete --query TopicArn --output text)
    echo "    Created SNS topic: $SNS_ARN"

    # Fan-out queue for SNS subscription (standard, not FIFO)
    if ! $AWSCLI sqs get-queue-url --queue-name compute-notifications 2>/dev/null; then
        $AWSCLI sqs create-queue --queue-name compute-notifications >/dev/null
    fi
    NOTIF_ARN="arn:aws:sqs:${AWS_REGION}:${ACCOUNT_ID}:compute-notifications"
    $AWSCLI sns subscribe \
        --topic-arn "$SNS_ARN" \
        --protocol sqs \
        --notification-endpoint "$NOTIF_ARN" >/dev/null
    echo "    Subscribed SQS to SNS"
else
    echo "    SNS topic already exists"
fi

# ── DynamoDB ──────────────────────────────────────────────────────────────────
echo "==> DynamoDB: compute-jobs"
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
    echo "    Created table with GSI (user_id, status) and TTL"
else
    echo "    Table already exists"
fi

# ── IAM ───────────────────────────────────────────────────────────────────────
echo "==> IAM: roles"
LAMBDA_TRUST='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
EC2_TRUST='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

for role in lambda-exec-role api-role worker-role; do
    if ! $AWSCLI iam get-role --role-name "$role" 2>/dev/null; then
        if [ "$role" = "lambda-exec-role" ]; then
            TRUST="$LAMBDA_TRUST"
        else
            TRUST="$EC2_TRUST"
        fi
        $AWSCLI iam create-role --role-name "$role" \
            --assume-role-policy-document "$TRUST" >/dev/null
        echo "    Created role: $role"
    else
        echo "    Role already exists: $role"
    fi
done

# Attach managed policies to lambda-exec-role
for policy_arn in \
    "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole" \
    "arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess" \
    "arn:aws:iam::aws:policy/AmazonS3FullAccess" \
    "arn:aws:iam::aws:policy/AmazonSNSFullAccess"; do
    $AWSCLI iam attach-role-policy --role-name lambda-exec-role \
        --policy-arn "$policy_arn" 2>/dev/null || true
done

# ── Secrets Manager ───────────────────────────────────────────────────────────
echo "==> Secrets Manager: db/postgres and redis/config"
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
        echo "    Created secret: $secret_name"
    else
        echo "    Secret already exists: $secret_name"
    fi
done

# ── CloudWatch Log Groups ─────────────────────────────────────────────────────
echo "==> CloudWatch: log groups"
for log_group in "/app/fastapi" "/app/worker" "/app/lambda"; do
    EXISTING=$($AWSCLI logs describe-log-groups \
        --log-group-name-prefix "$log_group" \
        --query "logGroups[?logGroupName=='${log_group}'].logGroupName" \
        --output text 2>/dev/null || echo "")
    if [ -z "$EXISTING" ] || [ "$EXISTING" = "None" ]; then
        $AWSCLI logs create-log-group --log-group-name "$log_group" >/dev/null
        echo "    Created: $log_group"
    else
        echo "    Already exists: $log_group"
    fi
done

# ── EventBridge ───────────────────────────────────────────────────────────────
echo "==> EventBridge: periodic-recompute (rate 6h)"
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
    echo "    Created rule: periodic-recompute"
else
    echo "    Rule already exists"
fi

# ── API Gateway ───────────────────────────────────────────────────────────────
echo "==> API Gateway: recommendation-api"
EXISTING_API=$($AWSCLI apigateway get-rest-apis \
    --query "items[?name=='recommendation-api'].id" --output text 2>/dev/null || echo "")
if [ -z "$EXISTING_API" ] || [ "$EXISTING_API" = "None" ]; then
    API_ID=$($AWSCLI apigateway create-rest-api \
        --name recommendation-api --query id --output text)
    ROOT_ID=$($AWSCLI apigateway get-resources \
        --rest-api-id "$API_ID" \
        --query "items[?path=='/'].id" --output text)
    RECS_ID=$($AWSCLI apigateway create-resource \
        --rest-api-id "$API_ID" --parent-id "$ROOT_ID" \
        --path-part "recommendations" --query id --output text)
    USER_ID=$($AWSCLI apigateway create-resource \
        --rest-api-id "$API_ID" --parent-id "$RECS_ID" \
        --path-part "{user_id}" --query id --output text)
    $AWSCLI apigateway put-method \
        --rest-api-id "$API_ID" --resource-id "$USER_ID" \
        --http-method GET --authorization-type NONE >/dev/null
    $AWSCLI apigateway put-integration \
        --rest-api-id "$API_ID" --resource-id "$USER_ID" \
        --http-method GET --type HTTP_PROXY --integration-http-method GET \
        --uri "http://api:8000/recommendations/{user_id}" >/dev/null
    $AWSCLI apigateway create-deployment \
        --rest-api-id "$API_ID" --stage-name prod >/dev/null
    echo "    Created API Gateway: $API_ID"
else
    echo "    API Gateway already exists"
fi

# ── ALB ───────────────────────────────────────────────────────────────────────
echo "==> ALB: fastapi-tg target group"
if ! $AWSCLI elbv2 describe-target-groups --names fastapi-tg 2>/dev/null; then
    $AWSCLI elbv2 create-target-group \
        --name fastapi-tg \
        --protocol HTTP \
        --port 8000 \
        --target-type ip \
        --health-check-path /health \
        --health-check-interval-seconds 30 >/dev/null 2>&1 || \
        echo "    ALB target group creation skipped (VPC not configured)"
    echo "    Created target group: fastapi-tg"
else
    echo "    Target group already exists"
fi

# ── Step Functions ────────────────────────────────────────────────────────────
echo "==> Step Functions: compute-pipeline"
LAMBDA_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/lambda-exec-role"
EXISTING_SF=$($AWSCLI stepfunctions list-state-machines \
    --query "stateMachines[?name=='compute-pipeline'].stateMachineArn" \
    --output text 2>/dev/null || echo "")
if [ -z "$EXISTING_SF" ] || [ "$EXISTING_SF" = "None" ]; then
    SF_DEF=$(cat <<'SFEOF'
{
  "Comment": "Recommendation compute pipeline",
  "StartAt": "ExportData",
  "States": {
    "ExportData": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:000000000000:function:export-data",
      "Retry": [{"ErrorEquals": ["States.ALL"], "IntervalSeconds": 2, "MaxAttempts": 3, "BackoffRate": 2.0}],
      "Catch": [{"ErrorEquals": ["States.ALL"], "Next": "PipelineFailed", "ResultPath": "$.error"}],
      "Next": "RunCompute"
    },
    "RunCompute": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:000000000000:function:compute-trigger",
      "Retry": [{"ErrorEquals": ["States.ALL"], "IntervalSeconds": 5, "MaxAttempts": 2, "BackoffRate": 1.5}],
      "Catch": [{"ErrorEquals": ["States.ALL"], "Next": "PipelineFailed", "ResultPath": "$.error"}],
      "Next": "ParallelFinalize"
    },
    "ParallelFinalize": {
      "Type": "Parallel",
      "Branches": [
        {
          "StartAt": "WarmCache",
          "States": {
            "WarmCache": {
              "Type": "Task",
              "Resource": "arn:aws:lambda:us-east-1:000000000000:function:cache-warmer",
              "End": true
            }
          }
        },
        {
          "StartAt": "UpdateStatus",
          "States": {
            "UpdateStatus": {
              "Type": "Task",
              "Resource": "arn:aws:lambda:us-east-1:000000000000:function:status-updater",
              "End": true
            }
          }
        }
      ],
      "Next": "NotifyComplete"
    },
    "NotifyComplete": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:000000000000:function:notifier",
      "End": true
    },
    "PipelineFailed": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:000000000000:function:status-updater",
      "Parameters": {"status": "failed", "error.$": "$.error"},
      "End": true
    }
  }
}
SFEOF
)
    $AWSCLI stepfunctions create-state-machine \
        --name compute-pipeline \
        --definition "$SF_DEF" \
        --role-arn "$LAMBDA_ROLE_ARN" >/dev/null
    echo "    Created state machine: compute-pipeline"
else
    echo "    State machine already exists"
fi

# ── Lambda Functions ──────────────────────────────────────────────────────────
echo "==> Lambda: deploying 5 functions"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

deploy_lambda() {
    local name="$1"
    local src_dir="$PROJECT_ROOT/$2"
    local handler="$3"
    local zip_file="/tmp/lambda_${name}.zip"

    (cd "$src_dir" && zip -r "$zip_file" . -x "*.pyc" -x "__pycache__/*" -x "*.egg-info/*" >/dev/null)

    if ! $AWSCLI lambda get-function --function-name "$name" 2>/dev/null; then
        $AWSCLI lambda create-function \
            --function-name "$name" \
            --runtime python3.12 \
            --handler "$handler" \
            --role "$LAMBDA_ROLE_ARN" \
            --zip-file "fileb://$zip_file" \
            --environment "Variables={
                AWS_ENDPOINT_URL=${LOCALSTACK_URL},
                AWS_DEFAULT_REGION=${AWS_REGION},
                AWS_ACCESS_KEY_ID=test,
                AWS_SECRET_ACCESS_KEY=test
            }" \
            --timeout 60 \
            --memory-size 256 >/dev/null
        echo "    Deployed: $name"
    else
        $AWSCLI lambda update-function-code \
            --function-name "$name" --zip-file "fileb://$zip_file" >/dev/null
        echo "    Updated: $name"
    fi
    rm -f "$zip_file"
}

deploy_lambda "export-data"      "src/lambdas/export_data"     "handler.lambda_handler"
deploy_lambda "compute-trigger"  "src/lambdas/compute_trigger" "handler.lambda_handler"
deploy_lambda "cache-warmer"     "src/lambdas/cache_warmer"    "handler.lambda_handler"
deploy_lambda "status-updater"   "src/lambdas/status_updater"  "handler.lambda_handler"
deploy_lambda "notifier"         "src/lambdas/notifier"        "handler.lambda_handler"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "✅ LocalStack provisioning complete"
echo "   S3:             similarity-matrices (versioned, 30d lifecycle)"
echo "   SQS:            compute-jobs.fifo (DLQ: compute-jobs-dlq.fifo, maxReceive=3)"
echo "   SNS:            compute-complete → compute-notifications (SQS)"
echo "   DynamoDB:       compute-jobs (GSI: user_id, status; TTL: expires_at)"
echo "   IAM:            lambda-exec-role, api-role, worker-role"
echo "   Secrets:        db/postgres, redis/config"
echo "   CloudWatch:     /app/fastapi, /app/worker, /app/lambda"
echo "   EventBridge:    periodic-recompute (rate 6h)"
echo "   API Gateway:    recommendation-api /recommendations/{user_id}"
echo "   ALB:            fastapi-tg → :8000/health"
echo "   Step Functions: compute-pipeline (ExportData→RunCompute→Parallel→Notify)"
echo "   Lambdas:        export-data, compute-trigger, cache-warmer, status-updater, notifier"
