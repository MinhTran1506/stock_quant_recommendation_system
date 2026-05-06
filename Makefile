# ─── HFT Platform Makefile ───────────────────────────────────────────────────
# Usage: make <target>
# Requires: Docker, Docker Compose, Python 3.11+, Node 20+

DOCKER_COMPOSE = docker compose
BACKEND        = $(DOCKER_COMPOSE) exec backend
ALEMBIC        = $(BACKEND) alembic
PYTEST         = $(BACKEND) pytest

.PHONY: help setup up down restart logs \
        db-init db-migrate db-rollback db-reset \
        seed backfill \
        test test-cov lint format \
        train-pipeline retrain \
        kafka-topics \
        build push \
        k8s-deploy k8s-status k8s-rollback \
        clean prune

# ─── Default ──────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  VN-HFT Platform — Developer Commands"
	@echo "  ─────────────────────────────────────"
	@echo "  Setup & run:"
	@echo "    make setup      — Copy .env.example → .env (first-time)"
	@echo "    make up         — Start all services (detached)"
	@echo "    make down       — Stop all services"
	@echo "    make restart    — Restart backend only"
	@echo "    make logs       — Tail all logs"
	@echo ""
	@echo "  Database:"
	@echo "    make db-init    — Apply all Alembic migrations"
	@echo "    make db-migrate — Create new Alembic migration"
	@echo "    make db-reset   — DROP all tables + re-migrate (destructive!)"
	@echo "    make seed       — Seed stock universe"
	@echo "    make backfill   — Backfill 1 year of EOD prices"
	@echo ""
	@echo "  Testing & quality:"
	@echo "    make test       — Run pytest suite"
	@echo "    make test-cov   — Run pytest with coverage report"
	@echo "    make lint       — Run ruff linter"
	@echo "    make format     — Run black + isort"
	@echo ""
	@echo "  ML:"
	@echo "    make retrain    — Trigger full weekly retraining"
	@echo ""
	@echo "  Infra:"
	@echo "    make build      — Build Docker images"
	@echo "    make k8s-deploy — Deploy to Kubernetes (kubectl required)"
	@echo ""

# ─── Setup ────────────────────────────────────────────────────────────────────
setup:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✅  .env created from .env.example — fill in your secrets"; \
	else \
		echo "⚠️  .env already exists, skipping copy"; \
	fi

# ─── Docker Compose ───────────────────────────────────────────────────────────
up:
	$(DOCKER_COMPOSE) up -d
	@echo "✅  Services started"
	@echo "    Frontend:  http://localhost:3000"
	@echo "    API docs:  http://localhost:8000/docs"
	@echo "    MLflow:    http://localhost:5000"
	@echo "    Airflow:   http://localhost:8080"
	@echo "    Grafana:   http://localhost:3001"
	@echo "    MinIO:     http://localhost:9001"

down:
	$(DOCKER_COMPOSE) down

restart:
	$(DOCKER_COMPOSE) restart backend worker

logs:
	$(DOCKER_COMPOSE) logs -f --tail=100

logs-backend:
	$(DOCKER_COMPOSE) logs -f backend

logs-worker:
	$(DOCKER_COMPOSE) logs -f worker

ps:
	$(DOCKER_COMPOSE) ps

# ─── Database ─────────────────────────────────────────────────────────────────
db-init:
	$(ALEMBIC) upgrade head
	@echo "✅  Migrations applied"

db-migrate:
	@read -p "Migration message: " msg; \
	$(ALEMBIC) revision --autogenerate -m "$$msg"

db-rollback:
	$(ALEMBIC) downgrade -1

db-reset:
	@echo "⚠️  This will DROP all tables. Press Ctrl+C to cancel."
	@sleep 3
	$(ALEMBIC) downgrade base
	$(ALEMBIC) upgrade head
	@echo "✅  Database reset complete"

db-shell:
	$(DOCKER_COMPOSE) exec timescaledb psql -U hft_user -d hft_platform

# ─── Data ─────────────────────────────────────────────────────────────────────
seed:
	$(BACKEND) python scripts/seed_stock_universe.py
	@echo "✅  Stock universe seeded"

backfill:
	$(BACKEND) python scripts/backfill_eod.py --days 365
	@echo "✅  EOD backfill complete"

backfill-intraday:
	$(BACKEND) python scripts/backfill_intraday.py --days 30
	@echo "✅  Intraday backfill complete"

kafka-topics:
	$(DOCKER_COMPOSE) exec kafka kafka-topics \
		--bootstrap-server kafka:9092 \
		--create --if-not-exists --topic tick_data --partitions 6 --replication-factor 1
	$(DOCKER_COMPOSE) exec kafka kafka-topics \
		--bootstrap-server kafka:9092 \
		--create --if-not-exists --topic order_events --partitions 3 --replication-factor 1
	$(DOCKER_COMPOSE) exec kafka kafka-topics \
		--bootstrap-server kafka:9092 \
		--create --if-not-exists --topic model_signals --partitions 3 --replication-factor 1
	$(DOCKER_COMPOSE) exec kafka kafka-topics \
		--bootstrap-server kafka:9092 \
		--create --if-not-exists --topic news_feed --partitions 3 --replication-factor 1
	@echo "✅  Kafka topics created"

minio-buckets:
	@. ./.env && \
	docker compose exec minio mc alias set local http://localhost:9000 $$S3_ACCESS_KEY $$S3_SECRET_KEY && \
	docker compose exec minio mc mb --ignore-existing local/hft-raw && \
	docker compose exec minio mc mb --ignore-existing local/hft-features && \
	docker compose exec minio mc mb --ignore-existing local/hft-models
	@echo "✅  MinIO buckets created"

# ─── Testing & quality ────────────────────────────────────────────────────────
test:
	$(PYTEST) tests/ -v --tb=short

test-cov:
	$(PYTEST) tests/ -v --cov=. --cov-report=term-missing --cov-report=html

lint:
	$(BACKEND) ruff check .
	$(BACKEND) mypy . --ignore-missing-imports

format:
	$(BACKEND) black .
	$(BACKEND) isort .
	$(BACKEND) ruff check . --fix

# ─── ML ───────────────────────────────────────────────────────────────────────
retrain:
	$(BACKEND) python -c "from models.training_pipeline import TrainingPipeline; TrainingPipeline().run_full_retrain()"
	@echo "✅  Retraining complete"

score:
	$(BACKEND) python scripts/score_stocks.py
	@echo "✅  Stock scoring complete"

# ─── Docker build ─────────────────────────────────────────────────────────────
build:
	$(DOCKER_COMPOSE) build

build-backend:
	$(DOCKER_COMPOSE) build backend

build-frontend:
	$(DOCKER_COMPOSE) build frontend

build-clean:
	$(DOCKER_COMPOSE) build --no-cache

# ─── Kubernetes ───────────────────────────────────────────────────────────────
k8s-apply:
	kubectl apply -f infra/k8s/

k8s-deploy:
	kubectl apply -f infra/k8s/deployments/
	kubectl apply -f infra/k8s/services/
	kubectl apply -f infra/k8s/configmaps/
	kubectl rollout status deployment/hft-backend -n hft-platform

k8s-status:
	kubectl get pods -n hft-platform
	kubectl get services -n hft-platform

k8s-rollback:
	kubectl rollout undo deployment/hft-backend -n hft-platform

k8s-logs:
	kubectl logs -n hft-platform -l app=hft-backend --tail=100 -f

# ─── Cleanup ──────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true

prune:
	$(DOCKER_COMPOSE) down -v
	docker system prune -f
	@echo "✅  All containers, volumes, and images pruned"

# ─── Full first-time setup shortcut ───────────────────────────────────────────
bootstrap: setup up
	@echo "Waiting 20s for services to be healthy…"
	@sleep 20
	@$(MAKE) db-init
	@$(MAKE) kafka-topics
	@$(MAKE) minio-buckets
	@$(MAKE) seed
	@echo ""
	@echo "🚀  Platform ready! Open http://localhost:3000"

# ─── Quant Trading Targets ─────────────────────────────────────────────────────
quant-score:
	$(BACKEND) python scripts/score_stocks.py
	@echo "✅  Quant scoring complete"

quant-score-factor:
	$(BACKEND) python scripts/score_stocks.py --strategy factor_model

quant-score-statarb:
	$(BACKEND) python scripts/score_stocks.py --strategy stat_arb

quant-score-momentum:
	$(BACKEND) python scripts/score_stocks.py --strategy momentum

quant-dag-trigger:
	$(DOCKER_COMPOSE) exec airflow airflow dags trigger quant_daily_signals

test-quant:
	$(PYTEST) tests/test_quant.py -v --tb=short

train-rl:
	$(BACKEND) python -c "from quant.strategies.rl_agent import RLPortfolioAgent; import pandas as pd, numpy as np; tickers=['VNM','VIC','HPG','FPT','TCB','MBB','VPB','STB']; dates=pd.bdate_range('2020-01-01','2024-01-01'); prices=pd.DataFrame(np.exp(np.random.normal(0,.015,size=(len(dates),len(tickers))).cumsum(0))*10000,index=dates,columns=tickers); agent=RLPortfolioAgent(len(tickers),tickers); agent.train(prices,tickers,n_episodes=50)"
	@echo "✅  RL agent training complete"