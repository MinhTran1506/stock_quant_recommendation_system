# Getting Started

This guide walks you through running the full platform locally from scratch.

---

## Prerequisites

| Requirement | Minimum Version | Notes |
|-------------|-----------------|-------|
| Docker Desktop | 24+ | Enable WSL2 backend on Windows |
| Docker Compose | v2.20+ | Bundled with Docker Desktop |
| Make | any | Git Bash / WSL / native on Mac/Linux |
| 8 GB RAM free | — | ML models + TimescaleDB are memory-heavy |
| 20 GB disk | — | Docker images + historical data |

> **Windows users:** All `make` commands work in Git Bash or WSL2. PowerShell does not support Makefiles natively.

---

## 1. Clone and Configure

```bash
git clone <repo-url>
cd stock_quant_recommendation_system
```

Copy the example environment file and fill in secrets:

```bash
cp .env.example .env   # if provided, otherwise edit config.py defaults
```

Key settings in `config.py` (or override via environment variables):

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | `changeme...` | JWT signing key — **change in production** |
| `DATABASE_URL` | `postgresql+asyncpg://...` | TimescaleDB connection |
| `REDIS_URL` | `redis://redis:6379` | Redis connection |
| `KAFKA_BOOTSTRAP_SERVERS` | `kafka:29092` | Kafka broker |
| `MINIO_ENDPOINT` | `minio:9000` | MinIO object storage |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | Experiment tracker |
| `ALGORITHM` | `HS256` | JWT algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | Token lifetime |

---

## 2. Bootstrap Everything

The `bootstrap` target starts all services, waits for them to be healthy, runs migrations, and creates infrastructure resources:

```bash
make bootstrap
```

This executes the following steps in order:

```
make up           → docker-compose up -d (10 services)
make wait-healthy → poll until all containers pass health checks
make db-init      → alembic upgrade head (creates all tables + hypertables)
make kafka-topics → create price-updates and quant-signals topics
make minio-buckets→ create models and market-data buckets
```

Watch the logs while services start:

```bash
make logs         # tail all containers
make logs-backend # tail only the FastAPI container
```

---

## 3. Load Initial Data

### Seed Stock Universe

Populates the `stocks` table with HOSE/HNX constituents (35 stocks):

```bash
make seed
# or directly:
docker-compose exec backend python scripts/seed_stock_universe.py
```

### Backfill Historical EOD Prices

Downloads 1 year of end-of-day OHLCV data via vnstock:

```bash
make backfill
# or with custom range:
docker-compose exec backend python scripts/backfill_eod.py --days 365
docker-compose exec backend python scripts/backfill_eod.py --start 2020-01-01 --end 2024-01-01
```

---

## 4. Access the Platform

| Interface | URL | Default Credentials |
|-----------|-----|---------------------|
| **Frontend dashboard** | http://localhost:3000 | register first |
| **API documentation** | http://localhost:8000/docs | Bearer token |
| **Airflow scheduler** | http://localhost:8080 | admin / admin |
| **MLflow experiments** | http://localhost:5000 | none |
| **MinIO console** | http://localhost:9001 | minioadmin / minioadmin |

### Create Your First User

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "yourpassword", "email": "admin@example.com"}'
```

Then log in:

```bash
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=yourpassword"
```

The response includes an `access_token`. Use it as `Authorization: Bearer <token>` on all subsequent requests.

---

## 5. Daily Operating Workflow

The **Airflow scheduler** handles everything automatically after hours. For manual runs:

### Re-score stocks now

```bash
docker-compose exec backend python scripts/score_stocks.py
# Specific strategy only:
docker-compose exec backend python scripts/score_stocks.py --strategy factor_model
# Specific tickers:
docker-compose exec backend python scripts/score_stocks.py --tickers VNM,VIC,HPG
```

### Trigger a DAG manually in Airflow

1. Open http://localhost:8080
2. Enable `daily_market_pipeline` or `quant_daily_signals`
3. Click ▶ Trigger DAG

### Scheduled pipelines

| DAG | Schedule (ICT) | What it does |
|-----|---------------|--------------|
| `daily_market_pipeline` | Weekdays 18:30 | Ingest EOD, compute features, score stocks, emit signals |
| `quant_daily_signals` | Weekdays 16:00 | Update regime, factor scores, stat-arb pairs, rebalance |

---

## 6. Stopping and Restarting

```bash
make down         # stop all containers (data preserved)
make down-v       # stop + delete all volumes (full reset)
make up           # start again
```

---

## 7. Common Issues

### Port conflict

If port 5432 is already in use (local PostgreSQL), edit `docker-compose.yml` to map a different host port:

```yaml
ports:
  - "5433:5432"   # map to 5433 on host
```

And update `DATABASE_URL` accordingly.

### Out of memory

If Docker crashes during ML model inference, increase Docker Desktop's memory limit to at least 8 GB under Settings → Resources.

### Database migration fails

```bash
make db-init      # re-runs alembic upgrade head (idempotent)
```

If the migration is in a broken state:

```bash
docker-compose exec backend alembic downgrade base
docker-compose exec backend alembic upgrade head
```

### Kafka topics not found

```bash
make kafka-topics   # idempotent — safe to run again
```
