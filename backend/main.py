"""
main.py — FastAPI application entrypoint.

Registers all routers, middleware, startup/shutdown lifecycle hooks,
and WebSocket endpoints for real-time price streaming.
"""
import asyncio
import logging
from contextlib import asynccontextmanager

import sentry_sdk
import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_client import make_asgi_app

from api.middleware.auth import JWTAuthMiddleware
from api.middleware.rate_limit import RateLimitMiddleware
from api.routes import (
    auth,
    backtest,
    news,
    portfolio,
    predictions,
    quant,
    stocks,
    strategy,
    universe,
)
from config import get_settings
from data.kafka.consumer import KafkaConsumerManager
from db.session import init_db, close_db
from utils.connection_manager import WebSocketConnectionManager
from utils.logging import configure_logging

settings = get_settings()
configure_logging(settings.log_level)
logger = structlog.get_logger(__name__)

# ─── Sentry (production error tracking) ───────────────────────────────────────
if settings.sentry_dsn:
    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.app_env,
        traces_sample_rate=0.1,
    )

# ─── WebSocket connection manager (shared across routes) ──────────────────────
ws_manager = WebSocketConnectionManager()


# ─── Lifespan (startup / shutdown) ────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle: connect to all external services."""
    logger.info("Starting HFT Platform", env=settings.app_env)

    # Initialize DB connection pool
    await init_db()

    # Start Kafka consumer for real-time tick streaming to WebSocket clients
    kafka_manager = KafkaConsumerManager(ws_manager=ws_manager)
    consumer_task = asyncio.create_task(kafka_manager.start())

    logger.info("All services initialised — platform ready")
    yield

    # Graceful shutdown
    logger.info("Shutting down HFT Platform")
    consumer_task.cancel()
    await kafka_manager.stop()
    await close_db()
    logger.info("Shutdown complete")


# ─── App factory ──────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description="Vietnam equity market intelligence & HFT research platform",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        redirect_slashes=False,
    )

    # ── Middleware (order matters — outermost runs first) ──────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(RateLimitMiddleware, calls=120, period=60)  # 120 req/min
    # JWT auth is applied per-router via dependency injection, not globally,
    # so public endpoints (auth, health) remain accessible.

    # ── Prometheus metrics endpoint ────────────────────────────────────────
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # ── API routers ────────────────────────────────────────────────────────
    prefix = settings.api_prefix
    app.include_router(auth.router,        prefix=f"{prefix}/auth",        tags=["Auth"])
    app.include_router(universe.router,    prefix=f"{prefix}/universe",    tags=["Universe"])
    app.include_router(stocks.router,      prefix=f"{prefix}/stocks",      tags=["Stocks"])
    app.include_router(predictions.router, prefix=f"{prefix}/predictions", tags=["Predictions"])
    app.include_router(backtest.router,    prefix=f"{prefix}/backtest",    tags=["Backtest"])
    app.include_router(portfolio.router,   prefix=f"{prefix}/portfolio",   tags=["Portfolio"])
    app.include_router(strategy.router,    prefix=f"{prefix}/strategy",    tags=["Strategy"])
    app.include_router(news.router,        prefix=f"{prefix}/news",        tags=["News"])
    app.include_router(quant.router,       prefix=f"{prefix}/quant",       tags=["Quant"])

    # ── WebSocket: real-time price stream ──────────────────────────────────
    @app.websocket("/ws/prices/{ticker}")
    async def ws_prices(websocket: WebSocket, ticker: str):
        """
        Subscribe to live price / order-book updates for a ticker.
        Internally the Kafka consumer publishes to this WebSocket group.
        """
        await ws_manager.connect(websocket, group=ticker.upper())
        try:
            while True:
                # Keep connection alive; server pushes data via ws_manager
                await websocket.receive_text()
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket, group=ticker.upper())

    @app.websocket("/ws/signals")
    async def ws_signals(websocket: WebSocket):
        """Subscribe to model signal events (new predictions, alerts)."""
        await ws_manager.connect(websocket, group="signals")
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket, group="signals")

    # ── Health & readiness ─────────────────────────────────────────────────
    @app.get("/health", tags=["Health"])
    async def health():
        return {"status": "ok", "env": settings.app_env}

    @app.get("/ready", tags=["Health"])
    async def ready():
        # TODO: check DB, Redis, Kafka connectivity
        return {"status": "ready"}

    return app


app = create_app()
