# Recommendation Engine

User-based collaborative filtering with a C/OpenMP compute core, FastAPI service,
SQS-driven worker, and a single web platform on port 3000. AWS services
(S3, SQS, SNS, DynamoDB, Secrets Manager, EventBridge, CloudWatch) are emulated
with LocalStack.

## Layout

```
rec-engine/
├── infra/
│   ├── docker/                 docker-compose + api/ Dockerfile
│   └── vagrant/                Ubuntu dev VM
├── scripts/
│   ├── run_all.sh              start stack inside VM, seed, run tests
│   ├── setup_localstack.sh     provision AWS resources
│   ├── validate_localstack.sh  smoke-test S3/SQS/SNS/Secrets
│   ├── seed_data.py            insert sample users/products/interactions
│   └── batch_jobs.sh           record interactions and trigger compute jobs
├── src/
│   ├── host-cuda/              C engines (openmp / mpi / cuda)
│   ├── vm-services/api/        FastAPI service
│   ├── worker/                 SQS consumer + ctypes bridge
│   └── dashboard/              FastAPI platform on :3000
├── tests/                      pytest integration tests
├── launch.sh                   build C engines and bring up the VM
└── README.md
```

## Quick start

```bash
bash launch.sh
```

This compiles the C engines on the host, starts the Vagrant VM, then runs
`scripts/run_all.sh` inside it which brings up docker-compose, provisions
LocalStack, seeds Postgres, and runs the integration tests.

Endpoints once the stack is up:

- Platform   `http://192.168.56.10:3000`
- API        `http://192.168.56.10:8000`
- LocalStack `http://192.168.56.10:4566/_localstack/health`
- Postgres   `192.168.56.10:5432`
- Redis      `192.168.56.10:6379`

## Manual operations

```bash
# bring docker-compose up by hand (inside the VM)
cd infra/docker && docker compose up -d --build

# provision LocalStack
bash scripts/setup_localstack.sh

# smoke-test LocalStack
bash scripts/validate_localstack.sh

# seed Postgres
python3 scripts/seed_data.py

# generate load: record interactions and trigger compute jobs
bash scripts/batch_jobs.sh --users 10 --products 6
```

## API

```bash
# record an interaction
curl -X POST http://localhost:8000/interactions \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "product_id": "<uuid>", "interaction_type": "purchase"}'

# trigger a compute job (force=true bypasses cache + dedup)
curl "http://localhost:8000/recommendations/1?force=true"

# poll job status
curl http://localhost:8000/recommendations/1/status
```

## C engine

Three variants live under `src/host-cuda/`:

| Variant | Path                       | Parallelism            |
|---------|----------------------------|------------------------|
| OpenMP  | `src/host-cuda/openmp/`    | shared-memory threads  |
| MPI     | `src/host-cuda/mpi/`       | distributed processes  |
| CUDA    | `src/host-cuda/cuda/`      | GPU kernels            |

`launch.sh` builds the OpenMP and CUDA-fallback `.so` files into `dist/` and
copies the OpenMP build to `dist/librec_engine.so`. The api and worker
containers re-compile the same sources internally so libuuid ABI matches the
runtime image.

The algorithm is user-based collaborative filtering: cosine similarity on
rating vectors with quickselect for top-K extraction.

## AWS services (LocalStack)

| Service          | Resource                                    |
|------------------|---------------------------------------------|
| S3               | `similarity-matrices` (versioned, 30d TTL)  |
| SQS              | `compute-jobs.fifo` + DLQ (`maxReceive=3`)  |
| SNS              | `compute-complete` -> `compute-notifications` (SQS) |
| DynamoDB         | `compute-jobs` (GSI on user_id, status; TTL on expires_at) |
| Secrets Manager  | `db/postgres`, `redis/config`               |
| CloudWatch Logs  | `/app/fastapi`, `/app/worker`               |
| EventBridge      | `periodic-recompute` (rate 6h -> SQS)       |
