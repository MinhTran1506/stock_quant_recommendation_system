"""
utils/metrics.py — Prometheus custom metrics for business KPIs.

Exported at /metrics by the FastAPI Prometheus ASGI app.

Categories:
  - API request metrics (latency, error rate)
  - Model inference metrics (latency, prediction distribution)
  - Data pipeline metrics (ingestion lag, row counts)
  - Trading metrics (paper P&L, order fill rate)
  - Model drift metrics (prediction mean, std deviation)
"""
from prometheus_client import (
    Counter, Gauge, Histogram, Summary, CollectorRegistry, REGISTRY
)

# ─── API Metrics ──────────────────────────────────────────────────────────────
API_REQUESTS = Counter(
    "hft_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status_code"],
)
API_LATENCY = Histogram(
    "hft_api_request_duration_seconds",
    "API request duration",
    ["method", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# ─── Model Metrics ────────────────────────────────────────────────────────────
MODEL_INFERENCE_LATENCY = Histogram(
    "hft_model_inference_duration_seconds",
    "Model inference latency",
    ["model_name", "horizon_days"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
)
MODEL_PREDICTION_SCORE = Histogram(
    "hft_model_prediction_score",
    "Distribution of meta-model stock scores",
    ["model_name"],
    buckets=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
)
MODEL_DIRECTIONAL_ACCURACY = Gauge(
    "hft_model_directional_accuracy",
    "Rolling 30-day directional accuracy",
    ["model_name", "horizon_days"],
)
MODEL_MAE = Gauge(
    "hft_model_mae",
    "Rolling 30-day Mean Absolute Error",
    ["model_name", "horizon_days"],
)
MODEL_PREDICTION_DRIFT = Gauge(
    "hft_model_prediction_drift",
    "PSI drift score vs training distribution",
    ["model_name"],
)

# ─── Data Pipeline Metrics ────────────────────────────────────────────────────
DATA_INGEST_LAG_SECONDS = Gauge(
    "hft_data_ingest_lag_seconds",
    "Seconds since last successful data ingestion",
    ["provider", "data_type"],
)
DATA_ROWS_INGESTED = Counter(
    "hft_data_rows_ingested_total",
    "Total rows ingested",
    ["provider", "data_type"],
)
KAFKA_CONSUMER_LAG = Gauge(
    "hft_kafka_consumer_lag_messages",
    "Kafka consumer lag in messages",
    ["topic", "consumer_group"],
)

# ─── Trading Metrics ──────────────────────────────────────────────────────────
PAPER_PORTFOLIO_VALUE = Gauge(
    "hft_paper_portfolio_value_vnd",
    "Current paper portfolio total value in VND",
    ["portfolio_id"],
)
PAPER_PORTFOLIO_PNL = Gauge(
    "hft_paper_portfolio_pnl_pct",
    "Paper portfolio cumulative P&L percentage",
    ["portfolio_id"],
)
ORDERS_SUBMITTED = Counter(
    "hft_orders_submitted_total",
    "Total orders submitted",
    ["strategy_id", "side", "is_paper"],
)
ORDER_FILL_RATE = Gauge(
    "hft_order_fill_rate",
    "Ratio of filled orders to submitted orders",
    ["strategy_id"],
)

# ─── Feature Store Metrics ────────────────────────────────────────────────────
FEATURE_CACHE_HIT_RATE = Gauge(
    "hft_feature_cache_hit_rate",
    "Redis feature cache hit rate (0-1)",
)
FEATURE_COMPUTE_LATENCY = Histogram(
    "hft_feature_compute_duration_seconds",
    "Feature computation latency per stock",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
)

# ─── WebSocket Metrics ────────────────────────────────────────────────────────
WS_ACTIVE_CONNECTIONS = Gauge(
    "hft_websocket_active_connections",
    "Active WebSocket connections",
    ["group"],
)


# ─── Middleware helper ────────────────────────────────────────────────────────
import time
from functools import wraps
from typing import Callable


def track_inference(model_name: str, horizon_days: int = 0):
    """Decorator: records latency and score distribution for model inference."""
    def decorator(fn: Callable):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            start = time.monotonic()
            result = await fn(*args, **kwargs)
            elapsed = time.monotonic() - start
            MODEL_INFERENCE_LATENCY.labels(
                model_name=model_name, horizon_days=str(horizon_days)
            ).observe(elapsed)
            return result
        return wrapper
    return decorator
