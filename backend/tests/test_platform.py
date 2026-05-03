"""
backend/tests/test_features.py — Unit tests for feature engineering.
backend/tests/test_backtest.py  — Integration tests for backtest engine.
backend/tests/test_api.py       — API endpoint smoke tests.
"""
import asyncio
import numpy as np
import pandas as pd
import pytest
from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient

# ─── Fixtures ─────────────────────────────────────────────────────────────────
def make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2022-01-01", periods=n)
    log_returns = rng.normal(0.0005, 0.015, n)
    close = 10000 * np.exp(np.cumsum(log_returns))
    high = close * (1 + abs(rng.normal(0, 0.005, n)))
    low  = close * (1 - abs(rng.normal(0, 0.005, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    volume = rng.randint(500_000, 5_000_000, n)
    return pd.DataFrame({
        "date": dates,
        "open": np.round(open_, 0),
        "high": np.round(high, 0),
        "low": np.round(low, 0),
        "close": np.round(close, 0),
        "volume": volume,
    })


# ─── Feature Store tests ───────────────────────────────────────────────────────
class TestFeatureStore:
    def setup_method(self):
        from data.feature_store.features import FeatureStore
        self.fs = FeatureStore()
        self.df = make_ohlcv(300)

    def test_compute_all_features_returns_dict(self):
        features = self.fs.compute_all_features(self.df, ticker="VNM")
        assert isinstance(features, dict)
        assert "ticker" in features
        assert features["ticker"] == "VNM"

    def test_price_return_features(self):
        features = self.fs.compute_all_features(self.df)
        assert "return_1d" in features
        assert "return_5d" in features
        assert "return_20d" in features
        assert isinstance(features["return_1d"], float)

    def test_volatility_features(self):
        features = self.fs.compute_all_features(self.df)
        assert "realised_vol_20d" in features
        assert "parkinson_vol_20d" in features
        # Volatility should be positive
        assert features["realised_vol_20d"] > 0

    def test_technical_indicators(self):
        features = self.fs.compute_all_features(self.df)
        assert "rsi_14" in features
        assert "macd_hist" in features
        assert "bb_pct_b" in features
        # RSI should be between 0 and 100
        assert 0 <= features["rsi_14"] <= 100

    def test_microstructure_features(self):
        bids = [[10500, 1000], [10490, 2000], [10480, 3000]]
        asks = [[10510, 1000], [10520, 2000], [10530, 3000]]
        features = self.fs.compute_microstructure_features(bids, asks)
        assert "bid_ask_spread" in features
        assert "order_imbalance" in features
        assert features["bid_ask_spread"] == pytest.approx(10.0)
        assert -1 <= features["order_imbalance"] <= 1

    def test_missing_columns_raises(self):
        bad_df = pd.DataFrame({"date": [], "close": []})
        with pytest.raises(ValueError, match="Missing columns"):
            self.fs.compute_all_features(bad_df)

    def test_short_dataframe_handles_gracefully(self):
        """Should not crash with very short data (< 14 bars for RSI)."""
        short_df = make_ohlcv(10)
        features = self.fs.compute_all_features(short_df)
        assert isinstance(features, dict)


# ─── Backtest engine tests ─────────────────────────────────────────────────────
class TestVectorBTEngine:
    def setup_method(self):
        from backtest.engine import VectorBTEngine, BacktestConfig
        self.engine = VectorBTEngine()
        self.config = BacktestConfig(
            start_date="2022-01-01",
            end_date="2023-12-31",
            initial_capital=1_000_000_000,
            commission_pct=0.0015,
            slippage_pct=0.001,
        )

    def _make_multi_ticker_prices(self):
        """Create price DataFrame with 5 tickers."""
        dates = pd.bdate_range("2022-01-01", "2023-12-31")
        rng = np.random.RandomState(0)
        data = {}
        for ticker in ["VNM", "VIC", "HPG", "FPT", "TCB"]:
            log_ret = rng.normal(0.0003, 0.015, len(dates))
            data[ticker] = pd.Series(10000 * np.exp(np.cumsum(log_ret)), index=dates)
        return pd.DataFrame(data)

    def _make_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Simple 5-day momentum signal."""
        signals = prices.pct_change(5).apply(
            lambda col: col.apply(lambda x: 1 if x > 0.02 else (-1 if x < -0.02 else 0))
        )
        return signals

    def test_run_returns_results(self):
        prices = self._make_multi_ticker_prices()
        signals = self._make_signals(prices)
        results = self.engine.run(prices, signals, self.config, run_id="test_001")

        assert results is not None
        assert isinstance(results.total_return_pct, float)
        assert isinstance(results.sharpe_ratio, float)
        assert len(results.equity_curve) > 0

    def test_equity_curve_starts_at_initial_capital(self):
        prices = self._make_multi_ticker_prices()
        signals = self._make_signals(prices)
        results = self.engine.run(prices, signals, self.config, run_id="test_002")

        first_value = results.equity_curve[0]["value"]
        assert abs(first_value - self.config.initial_capital) < self.config.initial_capital * 0.01

    def test_max_drawdown_is_negative(self):
        prices = self._make_multi_ticker_prices()
        signals = self._make_signals(prices)
        results = self.engine.run(prices, signals, self.config, run_id="test_003")
        assert results.max_drawdown_pct <= 0

    def test_win_rate_between_0_and_100(self):
        prices = self._make_multi_ticker_prices()
        signals = self._make_signals(prices)
        results = self.engine.run(prices, signals, self.config, run_id="test_004")
        assert 0 <= results.win_rate <= 100


# ─── Strategy orchestrator tests ──────────────────────────────────────────────
class TestStrategyOrchestrator:
    def setup_method(self):
        from strategy.orchestrator import StrategyOrchestrator, PaperExecutionAdapter
        self.orchestrator = StrategyOrchestrator(
            strategy_id="test_strategy",
            portfolio_id="test_portfolio",
            execution_mode="paper",
        )

    @pytest.mark.asyncio
    async def test_paper_order_fill(self):
        from strategy.orchestrator import OrderRequest
        adapter = self.orchestrator._adapter
        order = OrderRequest(
            ticker="VNM",
            side="BUY",
            order_type="MARKET",
            quantity=1000,
        )
        response = await adapter.submit_order(order, current_price=80000.0)
        assert response.status == "FILLED"
        assert response.filled_qty == 1000
        assert response.is_paper is True
        assert response.avg_fill_price > 0

    def test_live_trading_blocked_by_default(self):
        from strategy.orchestrator import LiveExecutionAdapter, RegulatoryBlockError
        with pytest.raises(RegulatoryBlockError):
            LiveExecutionAdapter(broker_api_key="key", broker_api_url="http://broker.test")


# ─── NLP pipeline tests ───────────────────────────────────────────────────────
class TestNLPPipeline:
    def setup_method(self):
        from models.nlp_pipeline import NLPPipeline
        self.nlp = NLPPipeline()

    def test_event_classification(self):
        text = "Vinamilk báo lợi nhuận tăng 20% trong Q3, vượt kỳ vọng analysts"
        tags = self.nlp._classify_events(text)
        assert "earnings" in tags

    def test_event_classification_dividend(self):
        text = "HPG announces cổ tức 2000 VND per share"
        tags = self.nlp._classify_events(text)
        assert "dividend" in tags

    def test_ticker_extraction(self):
        text = "VNM và VIC tăng mạnh hôm nay, trong khi HPG giảm nhẹ."
        tickers = self.nlp._extract_ticker_mentions(text)
        assert "VNM" in tickers
        assert "VIC" in tickers
        assert "HPG" in tickers

    def test_aggregate_sentiment_empty(self):
        score = self.nlp.compute_aggregate_sentiment([])
        assert score == 0.0

    def test_aggregate_sentiment_positive(self):
        articles = [
            {"sentiment_score": 0.8, "published_at": datetime.utcnow().isoformat()},
            {"sentiment_score": 0.6, "published_at": datetime.utcnow().isoformat()},
        ]
        score = self.nlp.compute_aggregate_sentiment(articles)
        assert score > 0


# ─── API smoke tests ──────────────────────────────────────────────────────────
@pytest.mark.asyncio
class TestAPIEndpoints:
    async def test_health_endpoint(self):
        """Health check should return 200 without auth."""
        from main import app
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    async def test_openapi_schema(self):
        """OpenAPI schema should be accessible."""
        from main import app
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "paths" in schema

    async def test_protected_endpoint_requires_auth(self):
        """Stock list should return 401 without token."""
        from main import app
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/v1/stocks")
        assert response.status_code == 401


# ─── pytest configuration ─────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
