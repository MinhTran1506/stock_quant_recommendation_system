"""
db/migrations/versions/0001_initial_schema.py
Initial database migration: creates all tables and sets up
TimescaleDB hypertables for time-series data.

Run with: alembic upgrade head
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSON
import uuid


revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── users ──────────────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("email", sa.String(255), nullable=False, unique=True),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(255)),
        sa.Column("is_active", sa.Boolean, default=True, nullable=False),
        sa.Column("is_superuser", sa.Boolean, default=False, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, onupdate=sa.func.now()),
    )
    op.create_index("ix_users_email", "users", ["email"])

    # ── stocks ─────────────────────────────────────────────────────────────────
    op.create_table(
        "stocks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("ticker", sa.String(20), nullable=False, unique=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("exchange", sa.Enum("HOSE", "HNX", "UPCOM", name="exchange"), nullable=False),
        sa.Column("sector", sa.String(100)),
        sa.Column("industry", sa.String(100)),
        sa.Column("market_cap", sa.Numeric(20, 2)),
        sa.Column("listing_date", sa.DateTime),
        sa.Column("is_active", sa.Boolean, default=True, nullable=False),
        sa.Column("metadata", JSON, default=dict),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("ix_stocks_ticker", "stocks", ["ticker"])

    # ── eod_prices (TimescaleDB hypertable) ────────────────────────────────────
    op.create_table(
        "eod_prices",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("stock_id", UUID(as_uuid=True),
                  sa.ForeignKey("stocks.id"), nullable=False),
        sa.Column("date", sa.DateTime, nullable=False),
        sa.Column("open", sa.Numeric(15, 2)),
        sa.Column("high", sa.Numeric(15, 2)),
        sa.Column("low", sa.Numeric(15, 2)),
        sa.Column("close", sa.Numeric(15, 2), nullable=False),
        sa.Column("volume", sa.BigInteger),
        sa.Column("adjusted_close", sa.Numeric(15, 2)),
        sa.Column("source", sa.String(50)),
    )
    op.create_unique_constraint("uq_eod_stock_date", "eod_prices", ["stock_id", "date"])
    op.create_index("ix_eod_stock_date", "eod_prices", ["stock_id", "date"])

    # Convert to TimescaleDB hypertable (partitioned by date, daily chunks)
    op.execute(
        "SELECT create_hypertable('eod_prices', 'date', "
        "chunk_time_interval => INTERVAL '1 month', if_not_exists => TRUE);"
    )
    op.execute(
        "ALTER TABLE eod_prices SET ("
        "  timescaledb.compress,"
        "  timescaledb.compress_orderby = 'date DESC',"
        "  timescaledb.compress_segmentby = 'stock_id'"
        ");"
    )
    op.execute(
        "SELECT add_compression_policy('eod_prices', INTERVAL '90 days');"
    )

    # ── intraday_prices (TimescaleDB hypertable) ───────────────────────────────
    op.create_table(
        "intraday_prices",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("stock_id", UUID(as_uuid=True),
                  sa.ForeignKey("stocks.id"), nullable=False),
        sa.Column("timestamp", sa.DateTime, nullable=False),
        sa.Column("interval_minutes", sa.Integer, nullable=False, default=1),
        sa.Column("open", sa.Numeric(15, 2)),
        sa.Column("high", sa.Numeric(15, 2)),
        sa.Column("low", sa.Numeric(15, 2)),
        sa.Column("close", sa.Numeric(15, 2), nullable=False),
        sa.Column("volume", sa.BigInteger),
    )
    op.create_unique_constraint(
        "uq_intraday", "intraday_prices", ["stock_id", "timestamp", "interval_minutes"]
    )
    op.create_index("ix_intraday_stock_ts", "intraday_prices", ["stock_id", "timestamp"])
    op.execute(
        "SELECT create_hypertable('intraday_prices', 'timestamp', "
        "chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);"
    )
    op.execute(
        "ALTER TABLE intraday_prices SET ("
        "  timescaledb.compress,"
        "  timescaledb.compress_orderby = 'timestamp DESC',"
        "  timescaledb.compress_segmentby = 'stock_id'"
        ");"
    )
    op.execute("SELECT add_compression_policy('intraday_prices', INTERVAL '7 days');")

    # ── orderbook_snapshots ────────────────────────────────────────────────────
    op.create_table(
        "orderbook_snapshots",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("stock_id", UUID(as_uuid=True),
                  sa.ForeignKey("stocks.id"), nullable=False),
        sa.Column("timestamp", sa.DateTime, nullable=False),
        sa.Column("bids", JSON, nullable=False, default=list),
        sa.Column("asks", JSON, nullable=False, default=list),
        sa.Column("mid_price", sa.Numeric(15, 2)),
        sa.Column("spread", sa.Numeric(15, 4)),
    )
    op.create_index("ix_ob_stock_ts", "orderbook_snapshots", ["stock_id", "timestamp"])
    op.execute(
        "SELECT create_hypertable('orderbook_snapshots', 'timestamp', "
        "chunk_time_interval => INTERVAL '1 hour', if_not_exists => TRUE);"
    )

    # ── fundamentals ──────────────────────────────────────────────────────────
    op.create_table(
        "fundamentals",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("stock_id", UUID(as_uuid=True),
                  sa.ForeignKey("stocks.id"), nullable=False),
        sa.Column("report_date", sa.DateTime, nullable=False),
        sa.Column("period", sa.String(10)),
        sa.Column("pe_ratio", sa.Float),
        sa.Column("pb_ratio", sa.Float),
        sa.Column("roe", sa.Float),
        sa.Column("roa", sa.Float),
        sa.Column("debt_to_equity", sa.Float),
        sa.Column("revenue", sa.Numeric(20, 2)),
        sa.Column("net_income", sa.Numeric(20, 2)),
        sa.Column("eps", sa.Float),
        sa.Column("dividend_yield", sa.Float),
        sa.Column("raw_data", JSON, default=dict),
    )

    # ── news_articles ──────────────────────────────────────────────────────────
    op.create_table(
        "news_articles",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("stock_id", UUID(as_uuid=True),
                  sa.ForeignKey("stocks.id"), nullable=True),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("source", sa.String(100)),
        sa.Column("url", sa.Text, unique=True),
        sa.Column("published_at", sa.DateTime, nullable=False),
        sa.Column("raw_content", sa.Text),
        sa.Column("summary", sa.Text),
        sa.Column("sentiment_score", sa.Float),
        sa.Column("sentiment_label", sa.String(20)),
        sa.Column("event_tags", JSON, default=list),
        sa.Column("embedding", JSON),
    )
    op.create_index(
        "ix_news_stock_published", "news_articles", ["stock_id", "published_at"]
    )

    # ── model_versions ─────────────────────────────────────────────────────────
    op.create_table(
        "model_versions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("version", sa.String(50), nullable=False),
        sa.Column("mlflow_run_id", sa.String(100)),
        sa.Column("mlflow_model_uri", sa.Text),
        sa.Column("model_type", sa.String(50)),
        sa.Column("horizon_days", sa.Integer),
        sa.Column("metrics", JSON, default=dict),
        sa.Column("is_champion", sa.Boolean, default=False),
        sa.Column("trained_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_unique_constraint("uq_model_version", "model_versions", ["name", "version"])

    # ── predictions (TimescaleDB hypertable) ───────────────────────────────────
    op.create_table(
        "predictions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("stock_id", UUID(as_uuid=True),
                  sa.ForeignKey("stocks.id"), nullable=False),
        sa.Column("model_version_id", UUID(as_uuid=True),
                  sa.ForeignKey("model_versions.id")),
        sa.Column("generated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("target_date", sa.DateTime, nullable=False),
        sa.Column("horizon_days", sa.Integer, nullable=False),
        sa.Column("predicted_return", sa.Float),
        sa.Column("predicted_price", sa.Float),
        sa.Column("confidence_lower", sa.Float),
        sa.Column("confidence_upper", sa.Float),
        sa.Column("score", sa.Float),
        sa.Column("feature_importances", JSON, default=dict),
        sa.Column("raw_outputs", JSON, default=dict),
    )
    op.create_index(
        "ix_pred_stock_generated", "predictions", ["stock_id", "generated_at"]
    )
    op.execute(
        "SELECT create_hypertable('predictions', 'generated_at', "
        "chunk_time_interval => INTERVAL '1 week', if_not_exists => TRUE);"
    )

    # ── portfolios ─────────────────────────────────────────────────────────────
    op.create_table(
        "portfolios",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("user_id", UUID(as_uuid=True),
                  sa.ForeignKey("users.id"), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("description", sa.Text),
        sa.Column("initial_capital", sa.Numeric(20, 2), nullable=False),
        sa.Column("currency", sa.String(10), default="VND"),
        sa.Column("is_paper", sa.Boolean, default=True, nullable=False),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, onupdate=sa.func.now()),
    )

    # ── positions ──────────────────────────────────────────────────────────────
    op.create_table(
        "positions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("portfolio_id", UUID(as_uuid=True),
                  sa.ForeignKey("portfolios.id"), nullable=False),
        sa.Column("stock_id", UUID(as_uuid=True),
                  sa.ForeignKey("stocks.id"), nullable=False),
        sa.Column("quantity", sa.BigInteger, default=0, nullable=False),
        sa.Column("avg_cost", sa.Numeric(15, 2)),
        sa.Column("opened_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("closed_at", sa.DateTime),
        sa.Column("is_open", sa.Boolean, default=True),
    )

    # ── orders ─────────────────────────────────────────────────────────────────
    op.create_table(
        "orders",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("portfolio_id", UUID(as_uuid=True),
                  sa.ForeignKey("portfolios.id"), nullable=False),
        sa.Column("stock_id", UUID(as_uuid=True),
                  sa.ForeignKey("stocks.id"), nullable=False),
        sa.Column("side", sa.Enum("BUY", "SELL", name="orderside"), nullable=False),
        sa.Column("order_type",
                  sa.Enum("MARKET", "LIMIT", "STOP", "STOP_LIMIT", name="ordertype"),
                  nullable=False),
        sa.Column("status",
                  sa.Enum("PENDING", "OPEN", "PARTIALLY_FILLED", "FILLED",
                          "CANCELLED", "REJECTED", name="orderstatus"),
                  default="PENDING", nullable=False),
        sa.Column("quantity", sa.BigInteger, nullable=False),
        sa.Column("limit_price", sa.Numeric(15, 2)),
        sa.Column("stop_price", sa.Numeric(15, 2)),
        sa.Column("filled_quantity", sa.BigInteger, default=0),
        sa.Column("avg_fill_price", sa.Numeric(15, 2)),
        sa.Column("commission", sa.Numeric(15, 4), default=0),
        sa.Column("strategy_id", sa.String(100)),
        sa.Column("is_paper", sa.Boolean, default=True),
        sa.Column("submitted_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("filled_at", sa.DateTime),
        sa.Column("raw_broker_response", JSON),
    )

    # ── strategies ─────────────────────────────────────────────────────────────
    op.create_table(
        "strategies",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("name", sa.String(100), unique=True, nullable=False),
        sa.Column("description", sa.Text),
        sa.Column("status",
                  sa.Enum("INACTIVE", "PAPER", "LIVE", name="strategystatus"),
                  default="INACTIVE"),
        sa.Column("config", JSON, default=dict),
        sa.Column("universe_filter", JSON),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, onupdate=sa.func.now()),
    )

    # ── backtest_runs ──────────────────────────────────────────────────────────
    op.create_table(
        "backtest_runs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("user_id", UUID(as_uuid=True),
                  sa.ForeignKey("users.id"), nullable=False),
        sa.Column("strategy_id", UUID(as_uuid=True),
                  sa.ForeignKey("strategies.id")),
        sa.Column("name", sa.String(200)),
        sa.Column("start_date", sa.DateTime, nullable=False),
        sa.Column("end_date", sa.DateTime, nullable=False),
        sa.Column("initial_capital", sa.Numeric(20, 2), nullable=False),
        sa.Column("config", JSON, default=dict),
        sa.Column("status", sa.String(20), default="PENDING"),
        sa.Column("summary_metrics", JSON),
        sa.Column("equity_curve", JSON),
        sa.Column("trade_log", JSON),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime),
        sa.Column("error_message", sa.Text),
    )


def downgrade() -> None:
    # Drop in reverse FK dependency order
    for table in [
        "backtest_runs", "strategies", "orders", "positions",
        "portfolios", "predictions", "model_versions",
        "news_articles", "fundamentals", "orderbook_snapshots",
        "intraday_prices", "eod_prices", "stocks", "users",
    ]:
        op.drop_table(table)

    # Drop enums
    for enum_name in [
        "exchange", "orderside", "ordertype", "orderstatus", "strategystatus"
    ]:
        op.execute(f"DROP TYPE IF EXISTS {enum_name} CASCADE;")
