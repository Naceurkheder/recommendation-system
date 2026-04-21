# Distributed Recommendation Engine

A high-performance recommendation system combining OpenMP/MPI/CUDA C backends with a fully distributed AWS architecture — all running locally via LocalStack. Built as a hands-on lab for the AWS Cloud Practitioner, Developer Associate, and Solutions Architect certifications.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Client / Tests                              │
└─────────────────┬────────────────────────────────────────────────────┘
                  │ HTTP
                  ▼
┌─────────────────────────┐      ┌──────────────────────────────────────┐
│   API Gateway (LocalStack)│───▶│  FastAPI  :8000                      │
│   /recommendations/{id}  │      │  GET /recommendations/{user_id}      │
└─────────────────────────┘      │  GET /recommendations/{id}/status     │
                                  │  POST /interactions                   │
              ┌───────────────────┤  GET /health                         │
              │                   └──────────┬────────────┬──────────────┘
              │ cache miss                   │            │
              │                              │ DynamoDB   │ SQS FIFO
              ▼                              ▼            ▼
┌─────────────────────┐     ┌────────────────┐  ┌──────────────────────┐
│   Redis              │     │  DynamoDB       │  │  compute-jobs.fifo   │
│   recs:{uid}:latest  │     │  compute-jobs   │  │  (+ DLQ, redrive=3) │
│   TTL 24h            │     │  GSI: user_id   │  └──────────┬───────────┘
└─────────────────────┘     └────────────────┘             │ long-poll
              ▲                                              ▼
              │ setex                              ┌──────────────────────┐
              │                                    │  Compute Worker       │
              │                                    │  PostgreSQL → CSV     │
              │                                    │  ctypes → C .so       │
              │                                    │  (OpenMP similarity)  │
              │                                    │  numpy cosine sim     │
              │                                    └────┬────────┬─────────┘
              └────────────────────────────────────────┘        │
                                                                 │ s3.put_object
                                                                 ▼
                                                       ┌─────────────────────┐
                                                       │  S3                  │
                                                       │  similarity-matrices │
                                                       │  matrices/{ts}.json  │
                                                       └─────────────────────┘
```

### Step Functions Pipeline (periodic / on-demand)

```
ExportData (Lambda) → RunCompute (Lambda) → ┌─ UploadMatrix (Lambda)
                                            ├─ WarmCache (Lambda)
                                            └─ UpdateStatus (Lambda)
                                                      ↓
                                            NotifyComplete (Lambda)
                                                      ↓
                                            SNS: compute-complete
                                                      ↓
                                            SQS: compute-notifications
```

---

## Components

| Component | Location | Purpose |
|-----------|----------|---------|
| FastAPI app | `src/vm-services/api/` | REST API, Redis cache, SQS dispatch |
| Compute worker | `src/worker/` | SQS consumer, C bridge, S3/Redis/DynamoDB writes |
| C engine (OpenMP) | `src/host-cuda/openmp/` | Compiled to `.so` via MPI sources |
| C engine (MPI) | `src/host-cuda/mpi/` | Source used for `.so` build (has `matrix.c`) |
| C engine (CUDA) | `src/host-cuda/cuda/` | GPU variant (requires nvcc) |
| Lambda functions | `src/lambdas/` | Step Functions pipeline steps |
| LocalStack setup | `scripts/setup_localstack.sh` | AWS service provisioning |
| DB schema | `src/vm-services/api/db/schema.sql` | PostgreSQL tables |

---

## AWS Services Used

| Service | Usage | Cert Domain |
|---------|-------|-------------|
| **S3** | Store similarity matrix JSON; CSV exports; versioning + lifecycle | CCP/SAA: Storage |
| **SQS FIFO** | Job queue with per-user MessageGroupId; DLQ after 3 retries | DVA/SAA: Messaging |
| **SNS** | Publish compute-complete events; fan-out to SQS subscriber | DVA/SAA: Messaging |
| **DynamoDB** | Job status tracking; GSI on user_id and status; TTL on expires_at | DVA/SAA: Database |
| **Secrets Manager** | DB + Redis credentials; fetched on startup with retry | DVA/SAA: Security |
| **Lambda** | Stateless functions in Step Functions pipeline | DVA/SAA: Compute |
| **Step Functions** | Orchestrate pipeline with parallel branches, retry, error catch | DVA/SAA: Orchestration |
| **EventBridge** | Trigger periodic recompute (rate 6 hours) | SAA: Events |
| **CloudWatch Logs** | Structured logging from API, worker, Lambda | CCP/DVA: Observability |
| **API Gateway** | REST API proxy to FastAPI; /recommendations/{user_id} | DVA/SAA: API |
| **ALB / ELBv2** | Load balancer target group pointing to FastAPI :8000 | SAA: Load Balancing |
| **IAM** | lambda-exec-role, api-role, worker-role with least-privilege policies | CCP/SAA: Security |

> All services run through **LocalStack** on `http://localhost:4566`. No real AWS account required.

---

## Prerequisites

```
Docker >= 24
Docker Compose >= 2.20   (for condition: service_healthy)
Python 3.12              (for seed script + tests on host)
Git
```

Install Python test dependencies on the host:
```bash
pip install pytest requests boto3 psycopg2-binary
```

---

## Quick Start

```bash
# Clone and enter the project
git clone https://github.com/Naceurkheder/recommendation-system.git
cd recommendation-system

# One command: build, start, provision AWS, seed data, run tests
bash scripts/run_all.sh
```

The script will:
1. `docker compose up -d --build` — start LocalStack, PostgreSQL, Redis, API, Worker
2. Wait for all healthchecks to pass
3. Run `setup_localstack.sh` — provision 12 AWS services
4. Run `seed_data.py` — insert 20 users, 15 products, 80 interactions (if empty)
5. Run `pytest tests/test_integration.py -v`

---

## Manual Operations

### Start / stop
```bash
# Start (from infra/docker/)
docker compose up -d

# Stop and remove volumes
docker compose down -v
```

### Provision AWS services (LocalStack)
```bash
bash scripts/setup_localstack.sh
```

### Seed the database
```bash
python3 scripts/seed_data.py
```

### Run tests only
```bash
pytest tests/test_integration.py -v
```

### Inspect LocalStack resources
```bash
# Requires awslocal (pip install awscli-local) or plain aws CLI
export AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_DEFAULT_REGION=us-east-1

# List S3 objects
aws --endpoint-url=http://localhost:4566 s3 ls s3://similarity-matrices/matrices/

# Query DynamoDB jobs
aws --endpoint-url=http://localhost:4566 dynamodb scan --table-name compute-jobs

# Peek SQS messages
aws --endpoint-url=http://localhost:4566 sqs receive-message \
    --queue-url http://localhost:4566/000000000000/compute-jobs.fifo \
    --wait-time-seconds 1
```

### Call the API
```bash
# Create an interaction
curl -X POST http://localhost:8000/interactions \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "product_id": "550e8400-e29b-41d4-a716-446655440001", "interaction_type": "purchase"}'

# Get recommendations (cache-first)
curl http://localhost:8000/recommendations/1

# Poll job status
curl http://localhost:8000/recommendations/1/status

# Synchronous C-engine endpoint (instant)
curl -X POST http://localhost:8000/recommendations/similar-users \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "k": 5}'
```

---

## C Engine

The recommendation core is implemented in C with three parallelism variants:

| Variant | Location | Parallelism | Status |
|---------|----------|-------------|--------|
| OpenMP | `src/host-cuda/openmp/` | Shared-memory threads | Binary compiled |
| MPI | `src/host-cuda/mpi/` | Distributed processes | Source used for `.so` |
| CUDA | `src/host-cuda/cuda/` | GPU kernels (32×32 tiles) | Requires nvcc |

The Docker images compile `librec_engine.so` from the MPI C sources (which include `matrix.c`) using `gcc -shared -fPIC -fopenmp`. The worker's `ctypes_bridge.py` uses this `.so` for data loading and top-K extraction, with numpy computing the cosine similarity matrix.

**Algorithm**: user-based collaborative filtering via cosine similarity of rating vectors, quickselect top-K extraction.

---

## Repository Structure

```
rec-engine/
├── infra/
│   ├── docker/
│   │   ├── api/Dockerfile          # python:3.12-slim + C .so compilation
│   │   └── docker-compose.yml      # LocalStack + db + redis + api + worker
│   └── vagrant/Vagrantfile         # Ubuntu 20.04 dev VM
├── scripts/
│   ├── setup_localstack.sh         # Provision all 12 AWS services
│   ├── seed_data.py                # Insert sample data
│   └── run_all.sh                  # One-command bring-up
├── src/
│   ├── host-cuda/
│   │   ├── openmp/                 # OpenMP C implementation
│   │   ├── mpi/                    # MPI + OpenMP C implementation
│   │   └── cuda/                   # CUDA C implementation
│   ├── lambdas/
│   │   ├── export_data/            # Lambda: PostgreSQL → S3 CSV
│   │   ├── compute_trigger/        # Lambda: enqueue SQS job
│   │   ├── cache_warmer/           # Lambda: S3 matrix → Redis
│   │   ├── status_updater/         # Lambda: DynamoDB status write
│   │   └── notifier/               # Lambda: SNS publish
│   ├── vm-services/api/            # FastAPI application
│   │   ├── main.py                 # API endpoints
│   │   ├── rec_engine_wrapper.py   # ctypes wrapper (existing)
│   │   └── db/schema.sql           # PostgreSQL schema
│   └── worker/
│       ├── worker.py               # SQS poll loop + pipeline
│       ├── ctypes_bridge.py        # numpy + C .so bridge
│       └── Dockerfile
└── tests/
    └── test_integration.py         # End-to-end pytest suite
```

---

## AWS Certification Mapping

### Cloud Practitioner (CCP)
- **S3** — object storage, versioning, lifecycle policies
- **CloudWatch** — log groups, structured observability
- **IAM** — roles, least-privilege policies
- **Shared Responsibility Model** — LocalStack simulates the AWS managed layer

### Developer Associate (DVA)
- **SQS FIFO** — exactly-once delivery, MessageGroupId, deduplication, DLQ
- **SNS** — pub/sub, fan-out to SQS subscriber
- **DynamoDB** — single-table design, GSI, TTL, conditional writes
- **Lambda** — event-driven handlers, environment variables, packaging
- **Secrets Manager** — credential rotation pattern, retry on ResourceNotFound
- **API Gateway** — REST API, HTTP proxy integration, stage deployment
- **Step Functions** — ASL, parallel branches, retry/catch/error states

### Solutions Architect Associate (SAA)
- **Decoupled architecture** — SQS decouples API from compute worker
- **Event-driven design** — EventBridge → SQS → Worker → SNS → SQS fan-out
- **Cache-aside pattern** — Redis in front of DynamoDB + heavy compute
- **ALB** — target group, health check, HTTP routing
- **S3 lifecycle** — cost optimization via expiration rules
- **Step Functions orchestration** — replace polling with managed state machine
- **Multi-AZ / HA** — each service has healthchecks; worker reconnects on failure
