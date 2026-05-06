# Vietnam Stock Quant Recommendation System — Documentation

A full-stack quantitative trading platform for Vietnamese equity markets (HOSE/HNX), combining deep learning forecasters, classical quant strategies, and real-time market data infrastructure.

---

## Documentation Index

| Document | Contents |
|----------|----------|
| [Getting Started](getting-started.md) | Prerequisites, installation, first-time setup, accessing UIs |
| [Architecture](architecture.md) | System design, service map, data flow diagrams |
| [API Reference](api-reference.md) | All REST endpoints, WebSocket channels, auth flow |
| [Quant Strategies](quant-strategies.md) | Factor model, stat arb, momentum/regime, portfolio optimizers, risk manager |
| [Development Guide](development.md) | Testing, adding strategies, environment variables, Docker tips |

---

## Tech Stack

| Layer | Technology | Version |
|-------|------------|---------|
| **API** | FastAPI + Uvicorn | 0.111.0 |
| **Language** | Python | 3.11 |
| **Database** | TimescaleDB (PostgreSQL 15) | 2.14 |
| **ORM** | SQLAlchemy | 2.0 |
| **Migrations** | Alembic | — |
| **Cache** | Redis | 7 |
| **Message Bus** | Apache Kafka (Confluent) | 7.5.0 |
| **Object Storage** | MinIO (S3-compatible) | — |
| **ML Experiments** | MLflow | 2.13.0 |
| **Deep Learning** | PyTorch | 2.3.0 |
| **Time-series DL** | NeuralForecast (N-BEATS, N-HiTS, TFT) | 1.7.4 |
| **Gradient Boosting** | LightGBM + CatBoost + XGBoost | — |
| **Graph ML** | torch-geometric (GraphSAGE) | — |
| **Backtesting** | vectorbt + Backtrader | 0.26.1 / 1.9.78 |
| **Orchestration** | Airflow | 2.8.1 |
| **Task Queue** | Celery | 5.4.0 |
| **Monitoring** | Prometheus + Grafana | — |
| **Frontend** | Next.js + Tailwind CSS | — |

---

## System Overview

```
Data Sources                Platform Core               Consumers
───────────────             ──────────────────          ──────────────────
vnstock (HOSE/HNX)    ──►  TimescaleDB (OHLCV)   ──►  Next.js Dashboard
FiinGroup (fundamentals)──► Redis (signals cache)  ──►  REST API (FastAPI)
News / NLP feeds      ──►  Kafka (event stream)   ──►  WebSocket (live)
                            MinIO (model artifacts)──►  Airflow (scheduler)
                            MLflow (experiments)   ──►  Grafana (monitoring)
```

### Six Core Capabilities

1. **Real-time data ingestion** — EOD + intraday prices via vnstock/FiinGroup, published to Kafka
2. **Deep learning forecasing** — TFT, N-BEATS/N-HiTS, TCN, GNN ensemble → 5-day price quantiles
3. **Quantitative strategies** — Factor model, statistical arbitrage, momentum+regime, mean reversion, order flow, RL agent
4. **Portfolio construction** — Mean-variance, Black-Litterman, Risk Parity, Max Diversification
5. **Risk management** — VaR/CVaR monitoring, drawdown halts, pre-trade checks
6. **Backtesting** — Vectorized (vectorbt) + realistic fill simulation (Backtrader)

---

## Service Ports (Default)

| Service | Port | UI |
|---------|------|----|
| FastAPI backend | 8000 | http://localhost:8000/docs |
| Next.js frontend | 3000 | http://localhost:3000 |
| Airflow | 8080 | http://localhost:8080 |
| MLflow | 5000 | http://localhost:5000 |
| MinIO console | 9001 | http://localhost:9001 |
| TimescaleDB | 5432 | — |
| Redis | 6379 | — |
| Kafka | 29092 | — |

---

## Quick Start (TL;DR)

```bash
git clone <repo>
cd stock_quant_recommendation_system

# One-command bootstrap
make bootstrap

# Load data
make seed          # stock universe
make backfill      # 1-year EOD history

# Open the app
open http://localhost:3000
```

See [Getting Started](getting-started.md) for the full walkthrough.
