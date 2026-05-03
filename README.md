# Vietnam HFT / Stock Intelligence Platform

A full-stack, ML-powered stock recommendation and high-frequency trading research platform targeting the Vietnam equity market (HOSE & HNX).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                           FRONTEND (React/Next.js)                  │
│   Dashboard · Stock Pages · Backtest UI · Portfolio Simulator       │
└────────────────────────────┬────────────────────────────────────────┘
                             │ REST / WebSocket
┌────────────────────────────▼────────────────────────────────────────┐
│                      FastAPI Backend (Python)                        │
│  ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌──────────────────────┐  │
│  │  Stocks  │ │Prediction│ │ Backtest  │ │  Portfolio / Orders  │  │
│  │   API    │ │   API    │ │    API    │ │        API           │  │
│  └────┬─────┘ └────┬─────┘ └─────┬─────┘ └──────────┬───────────┘  │
└───────┼────────────┼─────────────┼──────────────────┼──────────────┘
        │            │             │                  │
┌───────▼────────────▼─────────────▼──────────────────▼──────────────┐
│                        Core Services Layer                           │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────────┐  │
│  │Feature Store│  │ Model Server │  │   Strategy Orchestrator   │  │
│  │(Redis+Feast)│  │  (MLflow)    │  │   (Kafka-driven orders)   │  │
│  └─────────────┘  └──────────────┘  └───────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
        │                   │                        │
┌───────▼───────┐  ┌────────▼──────┐  ┌─────────────▼───────────────┐
│  Data Ingestion│  │  ML Training  │  │     Execution Layer         │
│  (Kafka+S3)   │  │ (Airflow+GPU) │  │  (Paper / Live Adapters)    │
└───────┬───────┘  └───────────────┘  └─────────────────────────────┘
        │
┌───────▼───────────────────────────────────────────────────────────┐
│                        Data Sources                                │
│  HOSE · HNX · Vietstock · FiinGroup · Local Broker APIs · News   │
└───────────────────────────────────────────────────────────────────┘
```

## Model Stack

| Layer | Models | Horizon |
|---|---|---|
| Multi-horizon | TFT, N-BEATS, N-HiTS | 1–20 days |
| Short-horizon | TCN, ARIMA/GARCH | Intraday/tick |
| Cross-sectional | GNN, Factor models | Daily rank |
| NLP | FinBERT, T5 summarizer | Event-driven |
| Meta/Ranking | LightGBM / CatBoost | Stock scoring |

## Phases

- **Phase 0** — Legal check, data provider contracts (2–4 weeks)
- **Phase 1** — MVP: EOD data + baseline models + paper sandbox (8–12 weeks)
- **Phase 2** — Intraday, TFT/N-BEATS ensemble, autoscaling (12–20 weeks)
- **Phase 3** — HFT R&D + regulated live deployment (TBD)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| ML Framework | PyTorch, PyTorch Lightning, scikit-learn |
| Boosting | LightGBM, CatBoost |
| NLP | Hugging Face Transformers |
| Feature Store | Feast + Redis |
| Experiment Tracking | MLflow |
| Streaming | Apache Kafka |
| Database | TimescaleDB (PostgreSQL) |
| Object Storage | MinIO (S3-compatible) |
| Orchestration | Apache Airflow |
| Backtesting | vectorbt, Backtrader |
| Frontend | React + Recharts + Tailwind |
| Containers | Docker + Kubernetes |
| Monitoring | Prometheus + Grafana + Sentry |

---

## Quick Start

```bash
# 1. Clone and configure environment
cp .env.example .env
# Edit .env with your API keys and secrets

# 2. Start all services (development)
docker-compose up -d

# 3. Run database migrations
docker-compose exec backend alembic upgrade head

# 4. Seed initial stock universe
docker-compose exec backend python scripts/seed_stock_universe.py

# 5. Trigger initial data backfill
docker-compose exec backend python scripts/backfill_eod.py --days 365

# 6. Access services:
# Frontend:   http://localhost:3000
# API docs:   http://localhost:8000/docs
# MLflow:     http://localhost:5000
# Airflow:    http://localhost:8080
# Grafana:    http://localhost:3001
```

## Project Structure

```
hft_platform/
├── backend/                    # FastAPI application
│   ├── api/                    # Route handlers
│   │   ├── routes/             # Endpoint modules
│   │   └── middleware/         # Auth, CORS, rate limiting
│   ├── data/                   # Data layer
│   │   ├── ingestion/          # Provider connectors
│   │   ├── feature_store/      # Feature computation & serving
│   │   └── kafka/              # Streaming producer/consumer
│   ├── models/                 # ML model definitions & training
│   ├── backtest/               # Backtesting engine
│   ├── strategy/               # Strategy orchestration
│   ├── db/                     # SQLAlchemy models & migrations
│   └── utils/                  # Shared utilities
├── frontend/                   # React dashboard
├── airflow/dags/               # ETL & training schedules
├── monitoring/                 # Prometheus & Grafana configs
├── infra/k8s/                  # Kubernetes manifests
└── scripts/                    # Utility & seed scripts
```

## ⚠️ Regulatory Notice

Automated order placement on Vietnamese exchanges requires regulatory approval.
**Do not run live trading without completing Phase 0 legal checks.**
All live-trading features are disabled by default (`LIVE_TRADING_ENABLED=false`).