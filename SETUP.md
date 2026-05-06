# Vietnam Stock Quant Intelligence Platform — Setup & Usage Guide

> **Audience**: Developers setting up or operating this platform locally or in production.
> **Status**: All live-trading features are disabled by default. The system runs in simulation/paper-trading mode unless `LIVE_TRADING_ENABLED=true` is explicitly set and **legal clearance** from Vietnamese regulators is obtained.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [First-Time Setup](#2-first-time-setup)
3. [Environment Configuration (`.env`)](#3-environment-configuration-env)
4. [Service URLs & Ports](#4-service-urls--ports)
5. [Docker Services Overview](#5-docker-services-overview)
6. [Database Migrations & Seeding](#6-database-migrations--seeding)
7. [Data Ingestion & Backfill](#7-data-ingestion--backfill)
8. [Airflow Pipelines](#8-airflow-pipelines)
9. [Using the REST API](#9-using-the-rest-api)
10. [WebSocket Streams](#10-websocket-streams)
11. [Frontend Application](#11-frontend-application)
12. [ML Model Training](#12-ml-model-training)
13. [Backtesting](#13-backtesting)
14. [Quant Strategies](#14-quant-strategies)
15. [Monitoring & Observability](#15-monitoring--observability)
16. [Makefile Command Reference](#16-makefile-command-reference)
17. [Kubernetes Deployment (Production)](#17-kubernetes-deployment-production)
18. [Development Workflows](#18-development-workflows)
19. [Troubleshooting](#19-troubleshooting)
20. [Regulatory Notes (Vietnam)](#20-regulatory-notes-vietnam)

---

## 1. Prerequisites

### Required Software

| Tool | Minimum Version | Purpose |
|---|---|---|
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | 24.x | Runs all services |
| [Git](https://git-scm.com/) | 2.x | Source control |
| Python | 3.11+ | Scripts & local dev |
| Node.js | 20 LTS | Frontend development |
| GNU Make | 4.x (Linux/Mac) or via WSL/Chocolatey (Windows) | Build automation |

### Hardware Recommendations

| Scenario | RAM | CPU | Disk |
|---|---|---|---|
| Basic (no ML training) | 8 GB | 4 cores | 20 GB |
| Full stack with ML | 16 GB | 8 cores | 50 GB |
| ML Training (GPU) | 32 GB | 8+ cores | 100 GB + NVIDIA GPU |

### Windows Notes

- Use **WSL 2** (Ubuntu 22.04+) for best Docker performance and full `make` support.
- Alternatively, install Make via [Chocolatey](https://chocolatey.org/): `choco install make`
- All `make` commands below work inside WSL 2 or Git Bash.

---

## 2. First-Time Setup

### Option A: One-Command Bootstrap (Recommended)

```bash
# 1. Clone the repository
git clone <repository-url> stock_quant_recommendation_system
cd stock_quant_recommendation_system

# 2. Copy and configure environment variables
cp .env.example .env
# Edit .env with your values (see Section 3)

# 3. Full bootstrap: builds images, starts services,
#    runs migrations, seeds universe, backfills 365 days of EOD data
make bootstrap
```

`make bootstrap` executes these steps sequentially:
1. `make setup` — validates `.env` exists and builds Docker images
2. `make up` — starts all 13 Docker services
3. `make db-init` — runs Alembic migrations
4. `make seed` — seeds the Vietnam stock universe (~450 symbols)
5. `make backfill` — fetches 365 days of EOD prices via vnstock

> **Expected time**: 10–20 minutes (first pull of Docker images is slow).

### Option B: Manual Step-by-Step

```bash
# 1. Start all services
docker-compose up -d

# 2. Wait for TimescaleDB and Redis to be healthy (≈30 seconds)
docker-compose ps

# 3. Run database migrations
docker-compose exec backend alembic upgrade head

# 4. Create the TimescaleDB hypertables
docker-compose exec backend python -c "
from db.session import engine
from db.models import Base
import asyncio
asyncio.run(engine.run_sync(Base.metadata.create_all))
"

# 5. Seed the stock universe
docker-compose exec backend python scripts/seed_stock_universe.py

# 6. Backfill 365 days of EOD prices
docker-compose exec backend python scripts/backfill_eod.py --days 365

# 7. Install frontend dependencies and start
cd frontend && npm install && npm run dev
```

---

## 3. Environment Configuration (`.env`)

Copy `.env.example` to `.env` and fill in the values below.

### Application

```env
APP_ENV=development         # development | production
APP_SECRET_KEY=             # 32-byte hex: python -c "import secrets; print(secrets.token_hex(32))"
APP_DEBUG=true              # Set false in production
LOG_LEVEL=INFO
ALLOWED_ORIGINS=http://localhost:3000
```

### Database (TimescaleDB / PostgreSQL)

```env
POSTGRES_HOST=timescaledb
POSTGRES_PORT=5432
POSTGRES_DB=hft_platform
POSTGRES_USER=hft_user
POSTGRES_PASSWORD=          # Choose a strong password
```

### Redis

```env
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=             # Choose a strong password
REDIS_DB=0
```

### Object Storage (MinIO / S3)

```env
S3_ENDPOINT=http://minio:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=              # Change from default in production
S3_BUCKET_RAW=hft-raw
S3_BUCKET_FEATURES=hft-features
S3_BUCKET_MODELS=hft-models
```

### Data Provider — vnstock (Primary, Free)

```env
# Optional: get a free API key at https://vnstocks.com/login
# Leave blank to use the anonymous tier (rate-limited)
VNSTOCK_API_KEY=
```

To get a free key:
1. Register at [https://vnstocks.com/login](https://vnstocks.com/login)
2. Copy the API key from your profile
3. Paste it as `VNSTOCK_API_KEY=your_key_here`

### Future Data Providers (leave blank for now)

```env
VIETSTOCK_USERNAME=         # Requires paid Vietstock subscription
VIETSTOCK_PASSWORD=
FIINGROUP_API_KEY=          # Requires paid FiinGroup subscription
```

### Kafka

```env
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
KAFKA_TOPIC_TICKS=tick_data
KAFKA_TOPIC_NEWS=news_feed
KAFKA_TOPIC_SIGNALS=signals
KAFKA_TOPIC_ORDERS=orders
```

### MLflow

```env
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_EXPERIMENT_NAME=stock_predictions
```

### Trading (Safety)

```env
LIVE_TRADING_ENABLED=false  # NEVER set true without regulatory approval
PAPER_TRADING_ENABLED=true
```

### Monitoring

```env
SENTRY_DSN=                 # Optional — leave blank for local dev
PROMETHEUS_PORT=9090
```

---

## 4. Service URLs & Ports

Once `docker-compose up -d` is running:

| Service | URL | Credentials |
|---|---|---|
| **Frontend** | http://localhost:3000 | Register via UI |
| **Backend API** | http://localhost:8000 | JWT Bearer token |
| **API Docs (Swagger)** | http://localhost:8000/docs | — |
| **API Docs (ReDoc)** | http://localhost:8000/redoc | — |
| **Airflow** | http://localhost:8080 | admin / admin |
| **MLflow** | http://localhost:5000 | — |
| **Grafana** | http://localhost:3001 | admin / admin |
| **Prometheus** | http://localhost:9090 | — |
| **MinIO Console** | http://localhost:9001 | minioadmin / `S3_SECRET_KEY` |
| **TimescaleDB** | localhost:5432 | `POSTGRES_USER` / `POSTGRES_PASSWORD` |
| **Redis** | localhost:6379 | `REDIS_PASSWORD` |
| **Kafka** | localhost:9092 | — |

---

## 5. Docker Services Overview

The `docker-compose.yml` defines 13 services:

| Service Name | Image | Role |
|---|---|---|
| `backend` | `./Dockerfile` | FastAPI application server |
| `workers` | `./Dockerfile` | Background async workers (NLP, features, drift) |
| `timescaledb` | `timescale/timescaledb-ha:pg15` | Primary database with time-series extensions |
| `redis` | `redis:7-alpine` | Feature cache, session store (5 min TTL) |
| `kafka` | `confluentinc/cp-kafka:7.5.0` | Event streaming (4 topics) |
| `zookeeper` | `confluentinc/cp-zookeeper:7.5.0` | Kafka coordination |
| `minio` | `minio/minio:latest` | S3-compatible raw data archival |
| `mlflow` | `ghcr.io/mlflow/mlflow:v2.12.1` | Model registry and experiment tracking |
| `airflow` | `apache/airflow:2.8.1` | Pipeline orchestration |
| `grafana` | `grafana/grafana:10.2.0` | Dashboards and alerting |
| `prometheus` | `prom/prometheus:v2.50.1` | Metrics collection |
| `frontend` | `node:20-alpine` | Next.js 14 UI |
| `nginx` | `nginx:alpine` | Reverse proxy (production) |

### Useful Docker commands

```bash
# View all running services
docker-compose ps

# Follow logs for a specific service
docker-compose logs -f backend

# Restart a single service
docker-compose restart backend

# Stop everything
docker-compose down

# Stop and destroy all volumes (full reset)
docker-compose down -v
```

---

## 6. Database Migrations & Seeding

### Run migrations

```bash
# Apply all pending migrations
make db-migrate
# OR
docker-compose exec backend alembic upgrade head

# Check migration status
docker-compose exec backend alembic current

# Rollback one migration
docker-compose exec backend alembic downgrade -1
```

### Migration files

Located in `backend/db/migrations/versions/`:

| File | Description |
|---|---|
| `0001_initial_schema.py` | Core tables: User, Stock, EODPrice, etc. |
| `0002_quant_signals.py` | Quant strategy tables and signal columns |

### Add a new migration

```bash
# Auto-generate from model changes
docker-compose exec backend alembic revision --autogenerate -m "describe_change"
# Review the generated file, then apply:
docker-compose exec backend alembic upgrade head
```

### Seed the stock universe

Populates the `stocks` table with ~450 Vietnamese listed companies:

```bash
make seed
# OR
docker-compose exec backend python scripts/seed_stock_universe.py
```

Pre-seeded symbols include: `VNM`, `VIC`, `VHM`, `HPG`, `FPT`, `MWG`, `TCB`, `VPB`, `BID`, `VCB`, `CTG`, `ACB`, `HDB`, `SSI`, `GAS`, and more across HOSE, HNX, and UPCOM.

### Reset the database

```bash
# Drop all data and re-run migrations + seeds
make db-reset
```

---

## 7. Data Ingestion & Backfill

Data is fetched from **vnstock** (free, open-source). The provider wraps the vnstock v4 Unified UI and is located at `backend/data/ingestion/vnstock_provider.py`.

### Backfill historical EOD prices

```bash
# Backfill all seeded stocks for the last 365 days (default)
make backfill
# OR
docker-compose exec backend python scripts/backfill_eod.py --days 365

# Backfill a specific ticker
docker-compose exec backend python scripts/backfill_eod.py --ticker VNM --days 90

# Backfill a date range
docker-compose exec backend python scripts/backfill_eod.py --start 2023-01-01 --end 2024-01-01

# Adjust parallelism (default: 5 concurrent tickers)
docker-compose exec backend python scripts/backfill_eod.py --days 365 --concurrency 10
```

### Backfill a single stock from the REPL

```python
import asyncio
from data.ingestion.vnstock_provider import VnstockProvider

provider = VnstockProvider()
df = asyncio.run(provider.fetch_eod_prices("VNM", start="2024-01-01", end="2024-12-31"))
print(df.head())
```

### Available VnstockProvider methods

| Method | Description |
|---|---|
| `fetch_eod_prices(ticker, start, end)` | Daily OHLCV prices |
| `fetch_intraday_prices(ticker, date)` | Intraday tick/bar data |
| `fetch_order_book(ticker)` | Current order book snapshot |
| `fetch_fundamentals(ticker)` | P/E, P/B, ROE, revenue ratios |
| `fetch_news(ticker, limit)` | Recent news articles |
| `fetch_stock_list()` | Full listed symbols from exchange |
| `fetch_financial_statements(ticker)` | Balance sheet, income, cash flow |
| `fetch_corporate_events(ticker)` | Dividends, rights issues |
| `fetch_ownership(ticker)` | Foreign ownership, major shareholders |

---

## 8. Airflow Pipelines

Access the Airflow UI at **http://localhost:8080** (admin / admin).

Two DAGs are pre-configured:

### `daily_market_pipeline`

**Schedule**: Weekdays at 18:30 ICT (11:30 UTC) — after market close

| Task | Description |
|---|---|
| `ingest_eod` | Fetch OHLCV for all seeded stocks via VnstockProvider |
| `ingest_news` | Fetch latest news articles per ticker |
| `compute_features` | Build technical indicators, push to Redis feature store |
| `score_stocks` | Run ML meta-model to generate 0–100 stock scores |
| `generate_signals` | Run quant strategies and publish signals to Kafka |
| `[weekly] retrain` | Re-train ML models on Sundays |

### `quant_signals_pipeline`

**Schedule**: Weekdays at 16:00 ICT (09:00 UTC) — before close

| Task | Description |
|---|---|
| `update_regime` | HMM-based market regime detection |
| `factor_scores` | Multi-factor alpha model rankings |
| `stat_arb_scan` | Scan for co-integrated pairs |
| `momentum` | Cross-sectional momentum signals |
| `[weekly] rebalance` | Portfolio rebalancing on Fridays |
| `[weekly] retrain_rl` | Retrain RL agent on Sundays |

### Triggering a DAG manually

```bash
# Via Airflow CLI
docker-compose exec airflow airflow dags trigger daily_market_pipeline

# Via Airflow UI: Pipelines → DAGs → click the "Play" button
```

### Monitoring DAG runs

In the Airflow UI: go to **DAGs → daily_market_pipeline → Grid** to see run history and task logs.

---

## 9. Using the REST API

All API endpoints require a JWT Bearer token (except registration and login).

### Authentication

#### Register a new user

```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "yourpassword", "full_name": "Your Name"}'
```

#### Login and get a token

```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "yourpassword"}'
```

Response:
```json
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer"
}
```

Tokens expire: access = 8 hours, refresh = 30 days.

#### Refresh a token

```bash
curl -X POST http://localhost:8000/auth/refresh \
  -H "Authorization: Bearer <refresh_token>"
```

### Stocks

```bash
TOKEN="eyJ..."

# List all stocks
curl http://localhost:8000/stocks/ -H "Authorization: Bearer $TOKEN"

# Get a single stock
curl http://localhost:8000/stocks/VNM -H "Authorization: Bearer $TOKEN"

# Get EOD price history
curl "http://localhost:8000/stocks/VNM/prices?start=2024-01-01&end=2024-12-31" \
  -H "Authorization: Bearer $TOKEN"

# Get intraday prices
curl "http://localhost:8000/stocks/VNM/intraday" \
  -H "Authorization: Bearer $TOKEN"
```

### Predictions & Rankings

```bash
# Get ML-based stock rankings (scores 0–100)
curl http://localhost:8000/predictions/rankings -H "Authorization: Bearer $TOKEN"

# Get predictions for a specific ticker
curl http://localhost:8000/predictions/VNM -H "Authorization: Bearer $TOKEN"
```

### Quant Strategies

```bash
# Get current market regime (Bull / Bear / Sideways)
curl http://localhost:8000/quant/regime -H "Authorization: Bearer $TOKEN"

# Get factor model rankings
curl http://localhost:8000/quant/factor-model/rankings -H "Authorization: Bearer $TOKEN"

# Get stat arb pairs
curl http://localhost:8000/quant/stat-arb/pairs -H "Authorization: Bearer $TOKEN"

# Get strategy signals
curl http://localhost:8000/quant/signals -H "Authorization: Bearer $TOKEN"
```

### Portfolio Management

```bash
# Create a portfolio
curl -X POST http://localhost:8000/portfolio/ \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "My Portfolio", "initial_capital": 1000000000}'

# Get portfolio holdings
curl http://localhost:8000/portfolio/1 -H "Authorization: Bearer $TOKEN"

# Place a paper order
curl -X POST http://localhost:8000/portfolio/1/orders \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "VNM", "side": "buy", "quantity": 100, "order_type": "market"}'
```

### Backtesting

```bash
# Run a backtest
curl -X POST http://localhost:8000/backtest/ \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "momentum",
    "universe": ["VNM", "VIC", "HPG", "FPT"],
    "start_date": "2023-01-01",
    "end_date": "2024-01-01",
    "initial_capital": 1000000000,
    "parameters": {"lookback": 20, "top_n": 5}
  }'

# Get backtest results
curl http://localhost:8000/backtest/1/results -H "Authorization: Bearer $TOKEN"
```

### Interactive API Docs

Visit **http://localhost:8000/docs** for the full interactive Swagger UI — you can test every endpoint directly in the browser after clicking "Authorize" and pasting your JWT token.

---

## 10. WebSocket Streams

The backend exposes two real-time WebSocket endpoints.

### Live price stream

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/prices/VNM");
ws.onmessage = (event) => {
  const tick = JSON.parse(event.data);
  console.log(tick); // { ticker, price, volume, timestamp }
};
```

### Live signal stream

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/signals");
ws.onmessage = (event) => {
  const signal = JSON.parse(event.data);
  console.log(signal); // { ticker, strategy, direction, score, timestamp }
};
```

Both streams require a JWT token passed as a query parameter:

```
ws://localhost:8000/ws/prices/VNM?token=eyJ...
```

---

## 11. Frontend Application

The Next.js 14 frontend is available at **http://localhost:3000**.

### Pages

| Route | Description |
|---|---|
| `/login` | Authentication page |
| `/dashboard` | Market overview, sector heatmap, live ticker strip |
| `/stocks/[ticker]` | Individual stock detail with charts and signals |
| `/portfolio` | Portfolio holdings and performance |
| `/backtest` | Backtest configuration and results |
| `/quant` | Quant strategy signals and regime info |
| `/models` | ML model performance and experiment tracking |
| `/news` | News feed with sentiment scoring |

### Running the frontend in development mode

```bash
cd frontend
npm install
npm run dev
# Access at http://localhost:3000
```

### Building for production

```bash
cd frontend
npm run build
npm start
```

---

## 12. ML Model Training

### Available models

| Model | Location | Use Case |
|---|---|---|
| TFT (Temporal Fusion Transformer) | `backend/models/tft.py` | Multi-horizon price forecasting (1–20 day) |
| N-BEATS | `backend/models/nbeats.py` | Univariate time-series forecasting |
| TCN + GARCH | `backend/models/tcn_garch.py` | Intraday volatility and price prediction |
| GNN | `backend/models/gnn.py` | Cross-sectional stock ranking with graph structure |
| NLP Pipeline | `backend/models/nlp_pipeline.py` | News sentiment (FinBERT + T5) |
| Meta-model | `backend/models/meta_model.py` | LightGBM ensemble of all model outputs (final 0–100 score) |

### Trigger a full retrain

```bash
make retrain
# OR
docker-compose exec backend python -m models.training_pipeline --all
```

### Retrain a specific model

```bash
docker-compose exec backend python -m models.training_pipeline --model tft
docker-compose exec backend python -m models.training_pipeline --model meta_model
```

### Track experiments in MLflow

Open **http://localhost:5000** to browse:
- All past training runs
- Hyperparameters, metrics (Sharpe, MSE, IC)
- Registered model versions
- Artifact storage (model weights, plots)

---

## 13. Backtesting

The backtest engine (`backend/backtest/engine.py`) is powered by **vectorbt** with the following strategies available:

| Strategy | Key | Description |
|---|---|---|
| Momentum Regime | `momentum` | Cross-sectional momentum with HMM regime filter |
| Statistical Arbitrage | `stat_arb` | Pairs trading on co-integrated pairs |
| Factor Model | `factor_model` | Multi-factor alpha score (5-factor + momentum) |
| Mean Reversion | `mean_reversion` | RSI/Bollinger band mean reversion |
| RL Agent | `rl_agent` | Deep PPO-based portfolio agent |

### Run a backtest via the API (see Section 9)

### Run a backtest via the script

```bash
docker-compose exec backend python scripts/score_stocks.py \
  --strategy momentum \
  --universe HOSE \
  --start 2022-01-01 \
  --end 2024-01-01
```

### Key backtest metrics reported

- Annualized Return, Sharpe Ratio, Max Drawdown
- Calmar Ratio, Sortino Ratio
- Win Rate, Profit Factor
- Turnover, Transaction Cost Impact

---

## 14. Quant Strategies

Located in `backend/quant/strategies/`. Each strategy inherits a common interface.

### Strategy details

| Strategy | File | Signal Frequency | Notes |
|---|---|---|---|
| Momentum + Regime | `momentum_regime.py` | Daily | Disabled during Bear regime (HMM) |
| Statistical Arbitrage | `stat_arb.py` | Daily/Intraday | Johansen cointegration; long-only adapted |
| Factor Model | `factor_model.py` | Weekly | Value, Quality, Low-vol, Momentum, Growth |
| Mean Reversion | `mean_reversion.py` | Daily | RSI + Bollinger, ±7% limit aware |
| RL Portfolio Agent | `rl_agent.py` | Daily | PPO with Dirichlet policy, GRU memory |
| Order Flow | `order_flow.py` | Intraday | Order imbalance signals |

### Vietnam-specific constraints applied to all strategies

- **Long-only**: No short-selling on HOSE/HNX/UPCOM
- **Price limits**: Orders respect the ±7% daily price band
- **Settlement**: T+2.5 — minimum weekly rebalancing frequency
- **Liquidity**: Position sizing capped vs. Average Daily Volume (ADV)

See `backend/quant/RESEARCH.md` for full academic references.

---

## 15. Monitoring & Observability

### Grafana Dashboards

Access **http://localhost:3001** (admin / admin).

Pre-built dashboards in `monitoring/grafana/dashboard/overview.json`:
- API request rates and latency (p50, p95, p99)
- Database query performance
- Model prediction drift (PSI metrics)
- Kafka consumer lag
- Redis hit rates
- Business metrics (predictions served, orders placed)

### Prometheus Metrics

Access raw metrics at **http://localhost:9090**.

Scraped targets:
- `backend:8000/metrics` — FastAPI custom business metrics
- `timescaledb:9187` — PostgreSQL metrics via postgres_exporter
- `redis:9121` — Redis metrics via redis-exporter
- `kafka:7071` — Kafka JMX metrics
- `node-exporter:9100` — Host system metrics

### Alert Rules

Defined in `monitoring/alert_rules.yml`. Alerts fire for:
- API error rate > 5%
- Model drift PSI > 0.2 (retraining needed)
- Kafka consumer lag > 1000 messages
- Database connection pool saturation

### Application Logging

Structured JSON logs via `backend/utils/logging.py`. In production, ship to your log aggregation service (ELK, Grafana Loki, etc.).

---

## 16. Makefile Command Reference

Run `make help` or see the full Makefile. Key targets:

### Setup & Docker

| Command | Description |
|---|---|
| `make bootstrap` | Full first-time setup (images + up + migrate + seed + backfill) |
| `make setup` | Validate `.env` and build Docker images |
| `make up` | Start all services (`docker-compose up -d`) |
| `make down` | Stop all services |
| `make restart` | Restart all services |
| `make logs` | Follow all service logs |
| `make ps` | Show service status |

### Database

| Command | Description |
|---|---|
| `make db-init` | Run all Alembic migrations |
| `make db-migrate` | Apply pending migrations |
| `make db-reset` | Drop and recreate schema |
| `make seed` | Seed Vietnam stock universe |
| `make backfill` | Backfill 365 days of EOD prices |

### Development

| Command | Description |
|---|---|
| `make test` | Run pytest test suite |
| `make lint` | Run ruff + mypy linting |
| `make format` | Auto-format with black + ruff |
| `make retrain` | Trigger full ML model retraining |
| `make shell` | Open a Python shell in the backend container |

### Production

| Command | Description |
|---|---|
| `make k8s-deploy` | Apply Kubernetes manifests |
| `make k8s-rollback` | Rollback Kubernetes deployment |

---

## 17. Kubernetes Deployment (Production)

Manifests are in `infra/k8s/`.

### Backend Deployment

- **3 replicas** (rolling update strategy)
- **4 Uvicorn workers** with uvloop
- Resources: 500m–2000m CPU, 1–4 Gi RAM per pod
- Health checks: `/ready` (readiness), `/health` (liveness)

### Horizontal Pod Autoscaling

- Min: 2 replicas, Max: 10 replicas
- Scales on CPU > 70% or Memory > 80%

### Deploy

```bash
# Apply all manifests
make k8s-deploy

# Or manually
kubectl apply -f infra/k8s/deployments/
kubectl apply -f infra/k8s/services/

# Check rollout status
kubectl rollout status deployment/backend
```

### GPU Support for ML Training

The model server deployment includes GPU tolerations. Set `nvidia.com/gpu: "1"` in resource limits if GPU nodes are available.

---

## 18. Development Workflows

### Adding a new quant strategy

1. Create `backend/quant/strategies/my_strategy.py` extending the base strategy interface
2. Register the strategy in `backend/strategy/orchestrator.py`
3. Add a backtest endpoint mapping in `backend/api/routes/backtest.py`
4. Write tests in `backend/tests/test_quant.py`
5. Add a DAG task to `airflow/dags/quant_pipeline.py` if it needs daily execution

### Adding a new API route

1. Create `backend/api/routes/my_route.py` with an `APIRouter`
2. Register it in `backend/main.py`:
   ```python
   from api.routes.my_route import router as my_router
   app.include_router(my_router, prefix="/my-route", tags=["My Route"])
   ```

### Running tests

```bash
make test
# OR
docker-compose exec backend pytest backend/tests/ -v

# Run a specific test file
docker-compose exec backend pytest backend/tests/test_quant.py -v

# With coverage
docker-compose exec backend pytest backend/tests/ --cov=backend --cov-report=html
```

### Linting and formatting

```bash
make lint      # ruff + mypy type check
make format    # black + ruff --fix auto-format
```

### Adding a new database model

1. Define the SQLAlchemy model in `backend/db/models.py`
2. Generate a migration:
   ```bash
   docker-compose exec backend alembic revision --autogenerate -m "add_my_table"
   ```
3. Review and apply:
   ```bash
   docker-compose exec backend alembic upgrade head
   ```

---

## 19. Troubleshooting

### Services won't start

```bash
# Check for port conflicts
docker-compose ps
netstat -an | grep 5432   # TimescaleDB
netstat -an | grep 6379   # Redis

# Check service logs for errors
docker-compose logs timescaledb
docker-compose logs backend
```

### Database connection errors

```bash
# Verify TimescaleDB is healthy
docker-compose exec timescaledb pg_isready -U hft_user -d hft_platform

# Test connection from backend container
docker-compose exec backend python -c "
import asyncio
from db.session import get_session
async def test():
    async with get_session() as s:
        result = await s.execute('SELECT 1')
        print('DB OK:', result.scalar())
asyncio.run(test())
"
```

### vnstock data fetching fails

```bash
# Test vnstock directly
docker-compose exec backend python -c "
from vnstock import Vnstock
s = Vnstock().stock(symbol='VNM', source='VCI')
print(s.quote.history(start='2024-01-01', end='2024-01-31'))
"
```

If you get rate limit errors, add a `VNSTOCK_API_KEY` to your `.env` (free at [vnstocks.com/login](https://vnstocks.com/login)).

### Migrations fail

```bash
# Check current migration state
docker-compose exec backend alembic current

# Stamp to a specific revision without running SQL (if state is out of sync)
docker-compose exec backend alembic stamp head
```

### Frontend cannot reach the backend

1. Ensure the backend is running: `docker-compose ps backend`
2. Check CORS: `ALLOWED_ORIGINS=http://localhost:3000` must be in `.env`
3. Check the frontend API base URL in `frontend/src/utils/api.ts`

### Memory issues during ML training

- Reduce batch size in `backend/models/training_pipeline.py`
- Reduce the number of training epochs
- Use `--concurrency 2` in backfill scripts to reduce parallelism
- Upgrade Docker Desktop memory limit to 12+ GB (Docker Desktop → Settings → Resources)

---

## 20. Regulatory Notes (Vietnam)

> **This platform is built for research and simulation only. Live trading requires regulatory approval.**

### Key Vietnamese market rules

| Rule | Detail |
|---|---|
| Short-selling | **Prohibited** on HOSE, HNX, UPCOM |
| Daily price limits | ±7% for HOSE, ±10% for HNX, ±15% for UPCOM |
| Settlement | T+2.5 (no same-day sell of newly bought shares) |
| Foreign ownership | Caps vary by sector (typically 49–100%) |
| Market hours | 09:00–11:30 and 13:00–14:45 ICT |

### Before enabling live trading

1. Complete **Phase 0 legal clearance** with a licensed Vietnamese broker (2–4 weeks minimum)
2. Obtain API access from the broker for programmatic order submission
3. Only then set `LIVE_TRADING_ENABLED=true` in `.env`
4. Start with **paper trading** (`PAPER_TRADING_ENABLED=true`) for at least 3 months

**The system enforces `LIVE_TRADING_ENABLED=false` as the default and will not submit real orders unless this flag is explicitly changed.**

---

*For questions, feature requests, or bug reports, open an issue in the repository.*
