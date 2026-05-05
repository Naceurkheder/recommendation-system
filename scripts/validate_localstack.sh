#!/usr/bin/env bash
set -uo pipefail

ENDPOINT="${1:-http://localhost:4566}"
REGION="us-east-1"
PASS=0
FAIL=0

export AWS_PAGER=""
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_DEFAULT_REGION=$REGION

_ok()  { echo "  [PASS] $*"; PASS=$((PASS+1)); }
_fail(){ echo "  [FAIL] $*"; FAIL=$((FAIL+1)); }
_hdr() { echo ""; echo "[$*]"; }

_py() {
    local label="$1" code="$2" out
    if out=$(python3 -c "$code" 2>&1); then
        _ok "$label${out:+  ->  $out}"
        return 0
    fi
    _fail "$label"
    printf '%s\n' "$out" | sed 's/^/         /'
    return 1
}

_mk_client() {
    local svc="$1"
    echo "boto3.client('$svc', endpoint_url='$ENDPOINT', region_name='$REGION', aws_access_key_id='test', aws_secret_access_key='test')"
}

S3_CLIENT=$(_mk_client s3)
SQS_CLIENT=$(_mk_client sqs)
SNS_CLIENT=$(_mk_client sns)
SM_CLIENT=$(_mk_client secretsmanager)

_hdr "Health"

_py "health endpoint reachable" "
import json, urllib.request, sys
try:
    with urllib.request.urlopen('$ENDPOINT/_localstack/health', timeout=10) as r:
        d = json.loads(r.read())
    print(f\"edition={d.get('edition','?')} version={d.get('version','?')} services={len(d.get('services',{}))}\")
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr); sys.exit(1)
"

_hdr "S3"
BUCKET="val-s3-$$"

_py "create-bucket" "
import boto3
$S3_CLIENT.create_bucket(Bucket='$BUCKET')
print('$BUCKET')
"

_py "put-object" "
import boto3
$S3_CLIENT.put_object(Bucket='$BUCKET', Key='probe.txt', Body=b'hello')
"

_py "get-object matches" "
import boto3
body = $S3_CLIENT.get_object(Bucket='$BUCKET', Key='probe.txt')['Body'].read().decode()
assert body == 'hello', repr(body)
print(repr(body))
"

python3 -c "
import boto3
try:
    s3 = $S3_CLIENT
    for o in s3.list_objects_v2(Bucket='$BUCKET').get('Contents', []):
        s3.delete_object(Bucket='$BUCKET', Key=o['Key'])
    s3.delete_bucket(Bucket='$BUCKET')
except: pass
" 2>/dev/null

_py "similarity-matrices bucket exists" "
import boto3
$S3_CLIENT.head_bucket(Bucket='similarity-matrices')
print('similarity-matrices')
"

_hdr "SQS"
QNAME="val-sqs-$$.fifo"

_py "create-queue (FIFO)" "
import boto3
r = $SQS_CLIENT.create_queue(
    QueueName='$QNAME',
    Attributes={'FifoQueue': 'true', 'ContentBasedDeduplication': 'true'})
print(r['QueueUrl'].split('/')[-1])
"

_py "send-message (FIFO)" "
import boto3, json
sqs = $SQS_CLIENT
url = sqs.get_queue_url(QueueName='$QNAME')['QueueUrl']
r = sqs.send_message(QueueUrl=url, MessageBody=json.dumps({'probe': True}),
                     MessageGroupId='validate', MessageDeduplicationId='probe-1')
print(r['MessageId'])
"

_py "receive-message (FIFO)" "
import boto3, json
sqs = $SQS_CLIENT
url = sqs.get_queue_url(QueueName='$QNAME')['QueueUrl']
msgs = sqs.receive_message(QueueUrl=url, MaxNumberOfMessages=1, WaitTimeSeconds=3).get('Messages', [])
assert msgs, 'no messages returned'
body = json.loads(msgs[0]['Body'])
assert body.get('probe') is True, repr(body)
print(json.dumps(body))
"

python3 -c "
import boto3
try:
    sqs = $SQS_CLIENT
    url = sqs.get_queue_url(QueueName='$QNAME')['QueueUrl']
    sqs.delete_queue(QueueUrl=url)
except: pass
" 2>/dev/null

_py "compute-jobs.fifo provisioned" "
import boto3
url = $SQS_CLIENT.get_queue_url(QueueName='compute-jobs.fifo')['QueueUrl']
print(url.split('/')[-1])
"

_hdr "SNS"
TOPIC="val-sns-$$"

_py "create-topic" "
import boto3
arn = $SNS_CLIENT.create_topic(Name='$TOPIC')['TopicArn']
print(arn.split(':')[-1])
"

_py "publish to topic" "
import boto3, json
sns = $SNS_CLIENT
arn = sns.create_topic(Name='$TOPIC')['TopicArn']
mid = sns.publish(TopicArn=arn, Message=json.dumps({'probe': True}))['MessageId']
sns.delete_topic(TopicArn=arn)
print(mid)
"

_py "compute-complete topic provisioned" "
import boto3
topics = $SNS_CLIENT.list_topics().get('Topics', [])
matches = [t['TopicArn'] for t in topics if t['TopicArn'].endswith(':compute-complete')]
assert matches, 'compute-complete topic not found'
print(matches[0].split(':')[-1])
"

_hdr "Secrets Manager"
SECRET="val-sm-$$"

_py "create-secret" "
import boto3, json
$SM_CLIENT.create_secret(Name='$SECRET', SecretString=json.dumps({'probe': True}))
print('$SECRET')
"

_py "get-secret-value" "
import boto3, json
v = json.loads($SM_CLIENT.get_secret_value(SecretId='$SECRET')['SecretString'])
assert v.get('probe') is True, repr(v)
$SM_CLIENT.delete_secret(SecretId='$SECRET', ForceDeleteWithoutRecovery=True)
print(list(v.keys()))
"

for secret in db/postgres redis/config; do
    _py "secret '$secret' provisioned" "
import boto3, json
v = json.loads($SM_CLIENT.get_secret_value(SecretId='$secret')['SecretString'])
assert v, 'empty secret'
print(list(v.keys()))
"
done

echo ""
echo "Results: ${PASS} passed, ${FAIL} failed"
[ "$FAIL" -gt 0 ] && exit 1
exit 0
