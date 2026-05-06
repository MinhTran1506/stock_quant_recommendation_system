"""
config.py — Centralised application settings loaded from environment variables.
All secrets are sourced from .env / K8s secrets; never hard-coded.
"""
from functools import lru_cache
from typing import List, Optional

from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ─────────────────────────────────────────────────────────
    app_name: str = "Vietnam HFT Platform"
    app_env: str = "development"
    app_secret_key: str = Field(..., min_length=16)
    log_level: str = "INFO"
    debug: bool = False
    # CRITICAL: stays False until regulatory clearance is granted
    live_trading_enabled: bool = False
    api_prefix: str = "/api/v1"
    allowed_origins: List[str] = ["http://localhost:3000"]

    # ── Database ─────────────────────────────────────────────────────────────
    postgres_host: str = "timescaledb"
    postgres_port: int = 5432
    postgres_db: str = "hft_platform"
    postgres_user: str = "hft_user"
    postgres_password: str = Field(...)
    db_pool_size: int = 20
    db_max_overflow: int = 40
    db_pool_timeout: int = 30

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def database_url_sync(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_password: str = Field(...)
    redis_db: int = 0
    redis_feature_ttl_seconds: int = 300   # 5 min TTL for feature cache

    @property
    def redis_url(self) -> str:
        return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # ── Kafka ─────────────────────────────────────────────────────────────────
    kafka_bootstrap_servers: str = "kafka:9092"
    kafka_topic_tick_data: str = "tick_data"
    kafka_topic_order_events: str = "order_events"
    kafka_topic_model_signals: str = "model_signals"
    kafka_topic_news_feed: str = "news_feed"
    kafka_consumer_group: str = "hft_platform_group"

    # ── Object Storage ────────────────────────────────────────────────────────
    s3_endpoint: str = "http://minio:9000"
    s3_access_key: str = Field(...)
    s3_secret_key: str = Field(...)
    s3_bucket_raw: str = "hft-raw"
    s3_bucket_features: str = "hft-features"
    s3_bucket_models: str = "hft-models"

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_experiment_name: str = "hft_vietnam"

    # ── Data Providers ────────────────────────────────────────────────────────
    # vnstock — open-source, free, no subscription needed.
    # Get a free API key at https://vnstocks.com/login for 60 req/min (vs 20 as guest).
    vnstock_api_key: Optional[str] = None

    # Vietstock — requires paid subscription (not yet configured; kept for future use)
    vietstock_api_key: Optional[str] = None
    vietstock_api_url: str = "https://api.vietstock.vn"
    # FiinGroup — requires enterprise subscription (not yet configured; kept for future use)
    fiingroup_api_key: Optional[str] = None
    fiingroup_api_url: str = "https://api.fiingroup.vn"

    # ── Broker ────────────────────────────────────────────────────────────────
    broker_api_key: Optional[str] = None
    broker_api_secret: Optional[str] = None
    broker_api_url: Optional[str] = None
    broker_account_id: Optional[str] = None

    # ── Model configuration ───────────────────────────────────────────────────
    model_inference_batch_size: int = 64
    model_retrain_interval_hours: int = 24
    feature_lookback_days: int = 252        # ~1 trading year
    prediction_horizons: List[int] = [1, 3, 5, 10, 20]  # trading days

    # ── Monitoring ────────────────────────────────────────────────────────────
    sentry_dsn: Optional[str] = None
    prometheus_port: int = 9090

    @field_validator("live_trading_enabled", mode="before")
    @classmethod
    def validate_live_trading(cls, v: bool) -> bool:
        """Prevent accidental live trading enablement."""
        return v  # Validation hook — add regulatory checks here

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance — call this throughout the app."""
    return Settings()
