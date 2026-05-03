"""
db/models.py — SQLAlchemy ORM models.

All time-series tables use TimescaleDB hypertables for efficient
range queries and compression. Standard tables use plain Postgres.
"""
from datetime import datetime
from decimal import Decimal
from typing import Optional
from uuid import uuid4

from sqlalchemy import (
    BigInteger, Boolean, Column, DateTime, Enum, Float,
    ForeignKey, Index, Integer, JSON, Numeric, String, Text, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship
import enum


class Base(DeclarativeBase):
    pass


# ─── Enums ────────────────────────────────────────────────────────────────────
class Exchange(str, enum.Enum):
    HOSE = "HOSE"
    HNX = "HNX"
    UPCOM = "UPCOM"


class OrderSide(str, enum.Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, enum.Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class OrderType(str, enum.Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class StrategyStatus(str, enum.Enum):
    INACTIVE = "INACTIVE"
    PAPER = "PAPER"
    LIVE = "LIVE"   # Only allowed after regulatory clearance


# ─── User ─────────────────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    portfolios = relationship("Portfolio", back_populates="user")
    backtests = relationship("BacktestRun", back_populates="user")


# ─── Stock Universe ────────────────────────────────────────────────────────────
class Stock(Base):
    __tablename__ = "stocks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    ticker = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    exchange = Column(Enum(Exchange), nullable=False)
    sector = Column(String(100))
    industry = Column(String(100))
    market_cap = Column(Numeric(20, 2))
    listing_date = Column(DateTime)
    is_active = Column(Boolean, default=True, nullable=False)
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    eod_prices = relationship("EODPrice", back_populates="stock")
    intraday_prices = relationship("IntradayPrice", back_populates="stock")
    predictions = relationship("Prediction", back_populates="stock")
    news_articles = relationship("NewsArticle", back_populates="stock")
    fundamentals = relationship("Fundamental", back_populates="stock")


# ─── EOD Price (TimescaleDB hypertable) ───────────────────────────────────────
class EODPrice(Base):
    __tablename__ = "eod_prices"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    stock_id = Column(UUID(as_uuid=True), ForeignKey("stocks.id"), nullable=False)
    date = Column(DateTime, nullable=False)
    open = Column(Numeric(15, 2))
    high = Column(Numeric(15, 2))
    low = Column(Numeric(15, 2))
    close = Column(Numeric(15, 2), nullable=False)
    volume = Column(BigInteger)
    adjusted_close = Column(Numeric(15, 2))
    source = Column(String(50))

    stock = relationship("Stock", back_populates="eod_prices")

    __table_args__ = (
        UniqueConstraint("stock_id", "date", name="uq_eod_stock_date"),
        Index("ix_eod_stock_date", "stock_id", "date"),
    )


# ─── Intraday Price (TimescaleDB hypertable) ───────────────────────────────────
class IntradayPrice(Base):
    __tablename__ = "intraday_prices"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    stock_id = Column(UUID(as_uuid=True), ForeignKey("stocks.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    interval_minutes = Column(Integer, nullable=False, default=1)
    open = Column(Numeric(15, 2))
    high = Column(Numeric(15, 2))
    low = Column(Numeric(15, 2))
    close = Column(Numeric(15, 2), nullable=False)
    volume = Column(BigInteger)

    stock = relationship("Stock", back_populates="intraday_prices")

    __table_args__ = (
        UniqueConstraint("stock_id", "timestamp", "interval_minutes", name="uq_intraday"),
        Index("ix_intraday_stock_ts", "stock_id", "timestamp"),
    )


# ─── Order Book Snapshot ───────────────────────────────────────────────────────
class OrderBookSnapshot(Base):
    __tablename__ = "orderbook_snapshots"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    stock_id = Column(UUID(as_uuid=True), ForeignKey("stocks.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    # Store as JSON: [{price, volume, side}, ...]
    bids = Column(JSON, nullable=False, default=list)
    asks = Column(JSON, nullable=False, default=list)
    mid_price = Column(Numeric(15, 2))
    spread = Column(Numeric(15, 4))

    __table_args__ = (Index("ix_ob_stock_ts", "stock_id", "timestamp"),)


# ─── Fundamental / Corporate Filing ───────────────────────────────────────────
class Fundamental(Base):
    __tablename__ = "fundamentals"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    stock_id = Column(UUID(as_uuid=True), ForeignKey("stocks.id"), nullable=False)
    report_date = Column(DateTime, nullable=False)
    period = Column(String(10))  # e.g., "Q1-2024"
    # Key ratios
    pe_ratio = Column(Float)
    pb_ratio = Column(Float)
    roe = Column(Float)
    roa = Column(Float)
    debt_to_equity = Column(Float)
    revenue = Column(Numeric(20, 2))
    net_income = Column(Numeric(20, 2))
    eps = Column(Float)
    dividend_yield = Column(Float)
    raw_data = Column(JSON, default=dict)

    stock = relationship("Stock", back_populates="fundamentals")


# ─── News Article ─────────────────────────────────────────────────────────────
class NewsArticle(Base):
    __tablename__ = "news_articles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    stock_id = Column(UUID(as_uuid=True), ForeignKey("stocks.id"), nullable=True)
    title = Column(Text, nullable=False)
    source = Column(String(100))
    url = Column(Text, unique=True)
    published_at = Column(DateTime, nullable=False)
    raw_content = Column(Text)
    summary = Column(Text)             # LLM-generated summary
    sentiment_score = Column(Float)    # -1.0 to 1.0
    sentiment_label = Column(String(20))  # POSITIVE | NEGATIVE | NEUTRAL
    event_tags = Column(JSON, default=list)  # ["earnings", "dividend", "merger"]
    embedding = Column(JSON)           # vector embedding for similarity search

    stock = relationship("Stock", back_populates="news_articles")
    __table_args__ = (Index("ix_news_stock_published", "stock_id", "published_at"),)


# ─── Model Registry Entry ─────────────────────────────────────────────────────
class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    mlflow_run_id = Column(String(100))
    mlflow_model_uri = Column(Text)
    model_type = Column(String(50))   # TFT | NBEATS | LSTM | LGBM | META
    horizon_days = Column(Integer)
    metrics = Column(JSON, default=dict)
    is_champion = Column(Boolean, default=False)
    trained_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    predictions = relationship("Prediction", back_populates="model_version")

    __table_args__ = (UniqueConstraint("name", "version", name="uq_model_version"),)


# ─── Prediction ───────────────────────────────────────────────────────────────
class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    stock_id = Column(UUID(as_uuid=True), ForeignKey("stocks.id"), nullable=False)
    model_version_id = Column(UUID(as_uuid=True), ForeignKey("model_versions.id"))
    generated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    target_date = Column(DateTime, nullable=False)    # date being predicted
    horizon_days = Column(Integer, nullable=False)
    predicted_return = Column(Float)
    predicted_price = Column(Float)
    confidence_lower = Column(Float)
    confidence_upper = Column(Float)
    score = Column(Float)             # Meta-model ranking score (0-100)
    feature_importances = Column(JSON, default=dict)
    raw_outputs = Column(JSON, default=dict)

    stock = relationship("Stock", back_populates="predictions")
    model_version = relationship("ModelVersion", back_populates="predictions")

    __table_args__ = (
        Index("ix_pred_stock_generated", "stock_id", "generated_at"),
    )


# ─── Portfolio ────────────────────────────────────────────────────────────────
class Portfolio(Base):
    __tablename__ = "portfolios"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    initial_capital = Column(Numeric(20, 2), nullable=False)
    currency = Column(String(10), default="VND")
    is_paper = Column(Boolean, default=True, nullable=False)  # paper=True by default
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="portfolios")
    positions = relationship("Position", back_populates="portfolio")
    orders = relationship("Order", back_populates="portfolio")


# ─── Position ─────────────────────────────────────────────────────────────────
class Position(Base):
    __tablename__ = "positions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False)
    stock_id = Column(UUID(as_uuid=True), ForeignKey("stocks.id"), nullable=False)
    quantity = Column(BigInteger, default=0, nullable=False)
    avg_cost = Column(Numeric(15, 2))
    opened_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime)
    is_open = Column(Boolean, default=True)

    portfolio = relationship("Portfolio", back_populates="positions")


# ─── Order ────────────────────────────────────────────────────────────────────
class Order(Base):
    __tablename__ = "orders"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False)
    stock_id = Column(UUID(as_uuid=True), ForeignKey("stocks.id"), nullable=False)
    side = Column(Enum(OrderSide), nullable=False)
    order_type = Column(Enum(OrderType), nullable=False)
    status = Column(Enum(OrderStatus), default=OrderStatus.PENDING, nullable=False)
    quantity = Column(BigInteger, nullable=False)
    limit_price = Column(Numeric(15, 2))
    stop_price = Column(Numeric(15, 2))
    filled_quantity = Column(BigInteger, default=0)
    avg_fill_price = Column(Numeric(15, 2))
    commission = Column(Numeric(15, 4), default=0)
    strategy_id = Column(String(100))   # which strategy generated this order
    is_paper = Column(Boolean, default=True)
    submitted_at = Column(DateTime, default=datetime.utcnow)
    filled_at = Column(DateTime)
    raw_broker_response = Column(JSON)

    portfolio = relationship("Portfolio", back_populates="orders")


# ─── Strategy ─────────────────────────────────────────────────────────────────
class Strategy(Base):
    __tablename__ = "strategies"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    status = Column(Enum(StrategyStatus), default=StrategyStatus.INACTIVE)
    config = Column(JSON, default=dict)    # strategy parameters
    universe_filter = Column(JSON)         # which stocks to trade
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)


# ─── Backtest Run ─────────────────────────────────────────────────────────────
class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    strategy_id = Column(UUID(as_uuid=True), ForeignKey("strategies.id"))
    name = Column(String(200))
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(Numeric(20, 2), nullable=False)
    config = Column(JSON, default=dict)
    status = Column(String(20), default="PENDING")  # PENDING|RUNNING|DONE|FAILED
    # Results stored as JSON for flexibility
    summary_metrics = Column(JSON)   # total_return, sharpe, max_drawdown, etc.
    equity_curve = Column(JSON)      # [{date, value}, ...]
    trade_log = Column(JSON)         # [{date, ticker, side, qty, price}, ...]
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    error_message = Column(Text)

    user = relationship("User", back_populates="backtests")
