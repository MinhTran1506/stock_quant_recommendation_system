# Architecture

System design, infrastructure topology, and data flow for the Vietnam Stock Quant Recommendation System.

---

## Service Map

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DOCKER COMPOSE NETWORK                          │
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │  Next.js     │    │  FastAPI     │    │  Celery Worker           │  │
│  │  :3000       │◄──►│  :8000       │◄──►│  (async task queue)      │  │
│  └──────────────┘    └──────┬───────┘    └──────────────────────────┘  │
│                             │                                            │
│              ┌──────────────┼──────────────┐                            │
│              ▼              ▼              ▼                             │
│  ┌───────────────┐  ┌────────────┐  ┌──────────────┐                   │
│  │ TimescaleDB   │  │   Redis    │  │    Kafka     │                   │
│  │ :5432         │  │   :6379    │  │   :29092     │                   │
│  │ (OHLCV, users,│  │ (signals,  │  │ (price-updates│                  │
│  │  predictions) │  │  sessions) │  │  quant-signals│                  │
│  └───────────────┘  └────────────┘  └──────────────┘                   │
│                                                                          │
│  ┌───────────┐  ┌────────────┐  ┌────────────┐                         │
│  │  MinIO    │  │  MLflow    │  │  Airflow   │                         │
│  │:9000/9001 │  │  :5000     │  │  :8080     │                         │
│  │ (model    │  │ (experiment│  │ (pipeline  │                         │
│  │  artifacts│  │  tracking) │  │  scheduler)│                         │
│  └───────────┘  └────────────┘  └────────────┘                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### 1. EOD Ingestion Pipeline (runs daily at 18:30 ICT)

```
vnstock API
    │
    ▼
Airflow: daily_market_pipeline
    │
    ├─► ingest_eod ──────────────────► TimescaleDB (stock_prices)
    │                                        │
    ├─► ingest_news ──► Kafka topic          │
    │                  (price-updates)       │
    │                                        ▼
    ├─► compute_features ────────────► Redis cache
    │                    └──────────► MinIO S3 archive
    │                                        │
    ├─► score_stocks ◄────────────────────── │
    │       │  (meta-model inference)
    │       └──────────────────────────► TimescaleDB (stock_scores)
    │                                    TimescaleDB (predictions)
    │
    └─► generate_signals ────────────► Kafka (quant-signals)
                                            │
                                            ▼
                                     FastAPI WebSocket
                                            │
                                            ▼
                                     Next.js LiveTickerStrip
```

### 2. Quant Signals Pipeline (runs daily at 16:00 ICT)

```
Airflow: quant_daily_signals
    │
    ├─► update_market_regime ────────► Redis (regime_state)
    │       (HMM 3-state)
    │
    ├─► compute_factor_scores ──────► TimescaleDB (factor_scores)
    │       (7-factor model)
    │
    ├─► run_stat_arb_scan ──────────► TimescaleDB (stat_arb_pairs)
    │       (cointegration pairs)
    │
    ├─► run_momentum ───────────────► TimescaleDB (momentum_signals)
    │       (TSMOM + cross-sectional)
    │
    └─► rebalance_portfolios ───────► TimescaleDB (portfolio_weights)
            (weekly)
```

### 3. Real-time Request Flow

```
Browser
  │
  ├─► HTTP Request ──► FastAPI ──► JWT validation (middleware)
  │                        │
  │                        ├─► TimescaleDB (reads)
  │                        ├─► Redis (cache hit)
  │                        └─► Celery (async tasks)
  │
  └─► WebSocket ──────► FastAPI /ws/prices ──► Kafka consumer
                              │
                              └─► broadcast to all connected clients
```

---

## Component Responsibilities

### FastAPI Backend (`backend/`)

| Module | Responsibility |
|--------|---------------|
| `main.py` | App factory, router registration, lifespan hooks |
| `api/routes/` | 9 route groups: auth, stocks, predictions, backtest, portfolio, strategy, universe, quant, news |
| `api/middleware/auth.py` | JWT validation middleware + `get_current_user` dependency |
| `api/middleware/rate_limit.py` | Per-IP rate limiting |
| `backtest/engine.py` | VectorBTEngine + BacktraderEngine |
| `data/ingestion/` | vnstock + FiinGroup data fetchers |
| `data/feature_store/` | Technical/fundamental feature computation |
| `data/kafka/consumer.py` | Kafka consumer for real-time signal processing |
| `db/` | SQLAlchemy models, async session, Alembic migrations |
| `models/` | TFT, N-BEATS/N-HiTS, TCN, GNN, LightGBM meta-model |
| `quant/strategies/` | Factor model, StatArb, Momentum+Regime, Mean Reversion, Order Flow, RL agent |
| `quant/portfolio/` | MVO, Black-Litterman, Risk Parity, Max Diversification |
| `quant/risk/` | VaR/CVaR, drawdown monitoring, pre-trade checks |
| `strategy/orchestrator.py` | Coordinates multi-strategy signal combination |
| `utils/` | Connection management, structured logging, Prometheus metrics |
| `workers/main.py` | Celery worker entrypoint |

### Databases

| Store | Technology | Data Stored |
|-------|------------|-------------|
| TimescaleDB | PostgreSQL 15 + TimescaleDB | OHLCV prices, predictions, scores, portfolio state, users |
| Redis | Redis 7 | Market regime state, feature cache, JWT session store |
| MinIO | S3-compatible object store | Trained model artifacts, feature archives, backtest reports |
| MLflow | PostgreSQL-backed | Experiment runs, model metrics, model registry |

### Message Bus

Two Kafka topics:

| Topic | Producers | Consumers |
|-------|-----------|-----------|
| `price-updates` | EOD ingestion task, intraday scraper | Celery worker, WebSocket broadcaster |
| `quant-signals` | Quant pipeline DAG, score_stocks script | Celery worker, strategy orchestrator |

---

## Database Schema (Core Tables)

```
stocks             — Universe of HOSE/HNX stocks (ticker, name, sector, exchange)
stock_prices       — TimescaleDB hypertable: OHLCV (stock_id, timestamp, open, high, low, close, volume)
predictions        — ML model forecasts (stock_id, model_name, horizon, timestamp, q10, q50, q90)
stock_scores       — Composite quant score per stock per day
factor_scores      — Per-factor breakdown (stock_id, date, factor_name, score)
portfolios         — Portfolio definitions (user_id, name, optimizer_type)
portfolio_weights  — Historical target weights (portfolio_id, stock_id, date, weight)
orders             — Trade orders (portfolio_id, stock_id, side, qty, price, status)
backtest_results   — Stored backtest runs with full metrics JSON
users              — Authenticated users (username, hashed_password, email)
model_versions     — MLflow-aligned model registry (model_name, run_id, champion flag)
```

Tables `stock_prices` and `predictions` are TimescaleDB **hypertables** partitioned by time (7-day chunks).

---

## ML Model Architecture

```
Raw OHLCV + Fundamentals + Sentiment
          │
          ▼
  Feature Engineering
  (technical indicators, factor scores)
          │
    ┌─────┴────────────┐
    │                  │
    ▼                  ▼
  TFT              N-BEATS / N-HiTS
  (multi-horizon   (trend/seasonality
   transformer)     stacks)
    │                  │
    └──────┬───────────┘
           │  (base model predictions as features)
           │
           ▼
       LightGBM                   GNN (GraphSAGE)
    Meta-Ranking Model     ◄──── (sector + correlation
    (LambdaRank objective)         graph embeddings)
           │
           ▼
    Stock Scores 0–100
    (top 20 = BUY signals)
```

The meta-model uses 48 features:
- 15 TFT quantile outputs (q10/q50/q90 × 5 horizons)
- 5 N-BEATS outputs
- 19 technical features (returns, RSI, MACD, Bollinger %B, volume)
- 6 fundamental features (P/E, P/B, ROE, ROA, D/E, dividend yield)
- 3 sentiment features (1d/7d sentiment score, 7d news count)

---

## Security Architecture

- **Authentication:** OAuth2 password flow → JWT (HS256, 30-min expiry)
- **Authorization:** `JWTAuthMiddleware` validates every non-public route
- **Rate limiting:** Per-IP sliding-window limiter (`api/middleware/rate_limit.py`)
- **Database:** Parameterized queries via SQLAlchemy ORM (no raw SQL injection)
- **Secrets:** All secrets via environment variables, never hardcoded
- **Network:** All services communicate on internal Docker network; only ports 3000, 8000, 8080, 5000, 9001 are exposed to the host

---

## Monitoring Stack

```
FastAPI /metrics ──────────────┐
TimescaleDB exporter ──────────┤
Redis exporter ────────────────┼──► Prometheus ──► Grafana dashboards
Kafka JMX exporter ────────────┤         │
Node exporter ─────────────────┘         └──► Alertmanager ──► PagerDuty/Slack
```

Scrape interval: 10s for backend, 15s for all other services.

48+ alert rules across four groups: Data Pipeline, Model Health, API Health, Portfolio/Risk.
