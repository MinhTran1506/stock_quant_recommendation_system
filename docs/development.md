# Development Guide

Reference for contributors and developers extending the platform.

---

## Running Tests

```bash
make test
```

This runs `pytest` inside the backend container. All 64 tests should pass.

```bash
# Run a specific test file
docker-compose exec backend pytest backend/tests/test_quant.py -v

# Run a specific test
docker-compose exec backend pytest backend/tests/test_platform.py::test_backtest -v

# Run with coverage report
docker-compose exec backend pytest --cov=backend --cov-report=term-missing
```

### Test Configuration

- **File:** `pytest.ini` (project root)
- **Mode:** `asyncio_mode = strict` — all async tests must be decorated with `@pytest.mark.asyncio`
- **Test files:** `backend/tests/test_platform.py` (API + backtest) and `backend/tests/test_quant.py` (strategies)

### Known Caveats in Tests

| Issue | Resolution Applied |
|-------|-------------------|
| `ModuleNotFoundError: jose` | PyJWT shim in `api/middleware/auth.py` and `api/routes/auth.py` |
| `ValueError: invalid unit abbreviation: B` | `freq="D"` in backtest engine (pandas 2.x dropped "B" for Timedelta) |
| DB initialized before auth | `_user` listed before `db` in route function parameters |
| `momentum_scalar` > 0.6 | `return_weight` dampening factor in `momentum_regime.py` |
| `PrintLogger has no .name` | `structlog.stdlib.LoggerFactory()` in `utils/logging.py` |

---

## Code Style

```bash
# Format with black and isort
make format

# Lint with flake8 and mypy
make lint
```

The project follows:
- **black** formatting (88 char line length)
- **isort** for import ordering
- **flake8** for linting
- **mypy** for type checking (partial coverage)

---

## Project Layout

```
backend/
├── api/            # FastAPI routes and middleware
│   ├── middleware/ # JWT auth, rate limiting
│   └── routes/     # One file per route group
├── backtest/       # Vectorbt + Backtrader engines
├── data/           # Ingestion, feature store, Kafka consumer
├── db/             # SQLAlchemy models, Alembic migrations, session
├── models/         # ML forecasters (TFT, N-BEATS, TCN, GNN, meta-model)
├── quant/          # Strategies, portfolio optimizers, risk manager
│   ├── strategies/ # factor_model, stat_arb, momentum_regime, etc.
│   ├── portfolio/  # optimizer.py
│   └── risk/       # risk_manager.py
├── strategy/       # Multi-strategy orchestrator
├── tests/          # pytest test suite
├── utils/          # Logging, metrics, connection manager
└── workers/        # Celery worker entrypoint
airflow/dags/       # Airflow DAG definitions
frontend/src/       # Next.js pages and components
scripts/            # Utility scripts (seed, backfill, score)
infra/k8s/          # Kubernetes manifests
monitoring/         # Prometheus + Grafana configuration
```

---

## Adding a New Quant Strategy

1. **Create the strategy file** in `backend/quant/strategies/`:

```python
# backend/quant/strategies/my_strategy.py
from dataclasses import dataclass
from typing import List
import pandas as pd

@dataclass
class MySignal:
    ticker: str
    score: float
    signal: str  # "BUY" | "HOLD" | "AVOID"

class MyStrategy:
    def __init__(self, lookback: int = 60):
        self.lookback = lookback

    def generate_signals(self, prices: pd.DataFrame) -> List[MySignal]:
        # prices: DataFrame indexed by date, columns = tickers
        ...
```

2. **Register with the orchestrator** in `backend/strategy/orchestrator.py`:

```python
from backend.quant.strategies.my_strategy import MyStrategy

# Add to the strategies dict:
self.strategies["my_strategy"] = MyStrategy()
```

3. **Add an API endpoint** (optional) in `backend/api/routes/quant.py`:

```python
@router.get("/my-strategy")
async def get_my_strategy_signals(db: AsyncSession = Depends(get_db), ...):
    ...
```

4. **Wire into Airflow** (optional) in `airflow/dags/quant_pipeline.py`:

```python
run_my_strategy = PythonOperator(
    task_id="run_my_strategy",
    python_callable=run_my_strategy_task,
)
compute_factor_scores >> run_my_strategy
```

5. **Write tests** in `backend/tests/test_quant.py`:

```python
def test_my_strategy():
    strategy = MyStrategy()
    prices = generate_test_prices(n_tickers=10, n_days=120)
    signals = strategy.generate_signals(prices)
    assert len(signals) > 0
    assert all(0 <= s.score <= 100 for s in signals)
```

---

## Adding a New API Route

1. Create `backend/api/routes/my_route.py`:

```python
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from backend.db.session import get_db
from backend.api.middleware.auth import get_current_user

router = APIRouter()

@router.get("")
async def list_items(
    _user=Depends(get_current_user),   # auth must come BEFORE db
    db: AsyncSession = Depends(get_db),
):
    ...
```

> **Important:** Always list `_user=Depends(get_current_user)` **before** `db: AsyncSession = Depends(get_db)` in the function signature. FastAPI resolves dependencies in parameter order, and `get_db` raises `RuntimeError` if the auth check hasn't run yet.

2. Register in `backend/main.py`:

```python
from backend.api.routes.my_route import router as my_router
app.include_router(my_router, prefix="/api/v1/my-route", tags=["my-route"])
```

3. Do not add `"/"` routes — the app uses `redirect_slashes=False` to avoid 307 redirects that break auth testing. Use `""` (empty string) as the root path:

```python
@router.get("")    # correct
@router.get("/")   # will cause test failures
```

---

## Database Migrations

```bash
# Create a new migration
docker-compose exec backend alembic revision --autogenerate -m "add my table"

# Apply migrations
make db-init   # or: docker-compose exec backend alembic upgrade head

# Rollback one step
docker-compose exec backend alembic downgrade -1

# View history
docker-compose exec backend alembic history
```

### Migration Notes

- Migrations live in `backend/db/migrations/versions/`
- TimescaleDB hypertables (`stock_prices`, `predictions`) require `execute_each_statement=True` in `env.py` because `SELECT create_hypertable(...)` must run as a standalone statement

---

## Environment Variables Reference

Define these in `.env` (or pass directly to Docker Compose):

| Variable | Default | Required |
|----------|---------|----------|
| `SECRET_KEY` | `changeme-in-production` | **Yes** (change this) |
| `ALGORITHM` | `HS256` | No |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | No |
| `DATABASE_URL` | `postgresql+asyncpg://quant:quantpass@timescaledb:5432/quantdb` | No |
| `REDIS_URL` | `redis://redis:6379` | No |
| `KAFKA_BOOTSTRAP_SERVERS` | `kafka:29092` | No |
| `MINIO_ENDPOINT` | `minio:9000` | No |
| `MINIO_ACCESS_KEY` | `minioadmin` | No |
| `MINIO_SECRET_KEY` | `minioadmin` | No |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | No |
| `SENTRY_DSN` | `""` | No (optional error tracking) |
| `LOG_LEVEL` | `INFO` | No |
| `CORS_ORIGINS` | `["http://localhost:3000"]` | No |
| `MAX_POSITIONS` | `20` | No |
| `INITIAL_CAPITAL` | `1000000000` | No |

---

## Docker Tips

### Rebuild after dependency changes

```bash
# After editing requirements.txt:
docker-compose build backend
docker-compose up -d backend
```

### Access a running container

```bash
docker-compose exec backend bash
docker-compose exec timescaledb psql -U quant -d quantdb
docker-compose exec redis redis-cli
```

### View logs

```bash
make logs              # all containers
docker-compose logs -f backend
docker-compose logs -f worker
docker-compose logs -f airflow
```

### Reset everything

```bash
make down-v            # stops containers and deletes all volumes
make bootstrap         # fresh start
```

---

## Monitoring

### Prometheus Metrics

The FastAPI backend exposes Prometheus metrics at `GET /metrics`.

Custom metrics defined in `backend/utils/metrics.py`:
- `http_requests_total` — request counts by route and status
- `http_request_duration_seconds` — latency histogram
- `model_inference_duration_seconds` — ML inference latency
- `prediction_drift_psi` — Population Stability Index for drift detection
- `portfolio_drawdown_pct` — Live drawdown metric

### Grafana

Import the pre-built dashboard from `monitoring/grafana/dashboard/overview.json`.

Access at http://localhost:3000 (Grafana default port, separate from Next.js if running standalone — check your docker-compose mapping).

### Alerts

Alert rules are in `monitoring/alert_rules.yml`. Key production alerts:

| Alert | Severity | Condition |
|-------|----------|-----------|
| `EODIngestionLag` | Warning | No EOD prices for > 2h |
| `IntradayIngestionStopped` | Critical | No intraday data for > 10min |
| `ModelDirectionalAccuracyLow` | Warning | Model accuracy < 50% |
| `MaxDrawdownBreached` | Critical | Portfolio drawdown > hard stop |
| `APIErrorRateHigh` | Warning | HTTP error rate > 5% |

---

## Kubernetes Deployment

Kubernetes manifests are in `infra/k8s/`:

```bash
# Deploy backend
kubectl apply -f infra/k8s/deployments/backend.yaml

# Deploy services
kubectl apply -f infra/k8s/services/services.yaml
```

The backend deployment mounts environment variables from a Kubernetes Secret. Create the secret before deploying:

```bash
kubectl create secret generic quant-secrets \
  --from-literal=SECRET_KEY='your-production-key' \
  --from-literal=DATABASE_URL='postgresql+asyncpg://...'
```

---

## Structured Logging

All log output uses `structlog` configured in `backend/utils/logging.py`.

```python
from backend.utils.logging import get_logger

logger = get_logger(__name__)

logger.info("backtest_complete", job_id=job_id, sharpe=results.sharpe_ratio)
logger.error("ingestion_failed", ticker=ticker, error=str(e))
```

Structured logs are JSON-formatted in production, human-readable in development (`LOG_LEVEL=DEBUG`).

> **Note:** The logger factory is `structlog.stdlib.LoggerFactory()`. Do not use `structlog.PrintLoggerFactory()` — it produces loggers without a `.name` attribute which breaks the `add_logger_name` processor.
