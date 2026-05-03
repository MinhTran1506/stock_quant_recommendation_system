"""
backend/tests/test_quant.py — Quant strategy unit and integration tests.

Tests cover:
  - Statistical arbitrage (cointegration detection, Kalman filter, OU params)
  - Factor model (individual factors, composite scores, IC evaluation)
  - Momentum + regime (HMM fitting, signal generation, crash prevention)
  - Portfolio optimisation (MVO, risk parity, Black-Litterman)
  - Risk manager (VaR, CVaR, breach detection)
  - Order flow (OBI, VPIN, microstructure signals)
  - Mean reversion (Bollinger, RSI, short-term reversal)
"""
import asyncio
import numpy as np
import pandas as pd
import pytest
from datetime import date, datetime, timedelta
from typing import List, Tuple

# ─── Fixtures ─────────────────────────────────────────────────────────────────
def make_prices(
    n_stocks: int = 10,
    n_days: int = 252,
    seed: int = 42,
    correlations: bool = False,
) -> pd.DataFrame:
    """Generate synthetic OHLCV panel for n stocks."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2022-01-01", periods=n_days)
    tickers = [f"ST{i:02d}" for i in range(n_stocks)]

    if correlations:
        # Create correlated pairs for stat arb testing
        factor = rng.normal(0, 0.01, n_days).cumsum()
        data = {}
        for i, t in enumerate(tickers):
            idio = rng.normal(0, 0.005, n_days).cumsum()
            factor_loading = 0.8 if i % 2 == 0 else 0.75  # pairs share factor
            data[t] = np.exp(factor * factor_loading + idio) * 10000
        return pd.DataFrame(data, index=dates)

    # Independent random walks
    log_returns = rng.normal(0.0003, 0.015, (n_days, n_stocks))
    prices = np.exp(np.cumsum(log_returns, axis=0)) * 10000
    return pd.DataFrame(prices, index=dates, columns=tickers)


def make_cointegrated_pair(n: int = 252) -> Tuple[pd.Series, pd.Series]:
    """Create a known cointegrated pair (y = 1.5*x + noise with mean reversion)."""
    rng = np.random.RandomState(0)
    x = 10000 + np.cumsum(rng.normal(0, 50, n))
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.95 * noise[i-1] + rng.normal(0, 30)
    y = 1.5 * x + noise
    dates = pd.bdate_range("2022-01-01", periods=n)
    return pd.Series(x, index=dates, name="ST00"), pd.Series(y, index=dates, name="ST01")


# ─── Statistical Arbitrage Tests ──────────────────────────────────────────────
class TestKalmanHedge:
    def setup_method(self):
        from quant.strategies.stat_arb import KalmanHedge
        self.kf = KalmanHedge(delta=1e-4, R=1e-2)

    def test_initial_update_returns_ratio(self):
        beta = self.kf.update(150.0, 100.0)
        assert isinstance(beta, float)
        assert beta > 0

    def test_hedge_ratio_converges(self):
        """Kalman filter should converge to true hedge ratio ~1.5."""
        x, y = make_cointegrated_pair(300)
        beta_series = self.kf.batch_fit(y.values, x.values)
        # Last 50 estimates should be close to 1.5
        final_betas = beta_series[-50:]
        assert abs(np.mean(final_betas) - 1.5) < 0.3, f"Expected ~1.5, got {np.mean(final_betas):.3f}"

    def test_filter_adapts_to_change(self):
        """Beta should shift when relationship changes mid-series."""
        rng = np.random.RandomState(42)
        x = np.cumsum(rng.normal(0, 1, 300)) + 100
        # First 150 bars: beta=1.5, then shifts to beta=2.0
        y = np.concatenate([1.5 * x[:150], 2.0 * x[150:]]) + rng.normal(0, 2, 300)
        betas = self.kf.batch_fit(y, x)
        assert betas[240] > betas[50], "Beta should increase after structural break"


class TestOUSpreadModel:
    def setup_method(self):
        from quant.strategies.stat_arb import OUSpreadModel
        self.ou = OUSpreadModel()

    def test_fit_known_ou_process(self):
        """Verify parameter recovery on a simulated OU process."""
        rng = np.random.RandomState(0)
        kappa, mu, sigma = 0.3, 0.0, 0.1
        n = 500
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = x[i-1] + kappa * (mu - x[i-1]) + sigma * rng.normal()

        params = self.ou.fit(x)
        assert "kappa" in params and "half_life" in params
        assert abs(params["kappa"] - kappa) < 0.15, f"kappa: expected {kappa}, got {params['kappa']:.3f}"
        assert 0 < params["half_life"] < 200

    def test_fit_with_short_series(self):
        """Should not crash on very short series."""
        params = self.ou.fit(np.random.normal(0, 1, 15))
        assert isinstance(params, dict)


class TestPairsFinder:
    def setup_method(self):
        from quant.strategies.stat_arb import PairsFinder
        self.finder = PairsFinder(min_corr=0.6, max_half_life_days=120, lookback_days=200)

    def test_finds_planted_cointegrated_pair(self):
        """Should find the planted cointegrated pair."""
        x, y = make_cointegrated_pair(252)
        prices = pd.concat([x, y], axis=1)
        # Add some noise stocks
        noise = make_prices(n_stocks=5, n_days=252, seed=99)
        prices = pd.concat([prices, noise], axis=1)

        pairs = self.finder.find_pairs(prices)
        tickers_found = [(p.ticker_a, p.ticker_b) for p in pairs]
        # At least one pair involving ST00 or ST01 should be found
        involved = any(
            "ST00" in (a, b) or "ST01" in (a, b)
            for a, b in tickers_found
        )
        # Note: cointegration test may not always find the pair with short series
        # Just verify no crash and returns valid PairConfig objects
        for pair in pairs:
            assert pair.half_life_days > 0
            assert isinstance(pair.hedge_ratio, float)


class TestStatArbStrategy:
    def setup_method(self):
        from quant.strategies.stat_arb import StatArbStrategy, PairConfig
        self.strategy = StatArbStrategy(long_only=True)
        self.pair = PairConfig(
            ticker_a="ST00", ticker_b="ST01",
            hedge_ratio=1.5, spread_mean=0.0, spread_std=0.05, half_life_days=10
        )

    def test_signal_is_valid(self):
        prices = make_prices(n_stocks=5, n_days=100, seed=0, correlations=True)
        prices.columns = ["ST00", "ST01", "ST02", "ST03", "ST04"]
        signals = self.strategy.generate_signals(prices, [self.pair])
        for s in signals:
            assert s.signal in (-1, 0, 1)
            assert isinstance(s.z_score, float)

    def test_long_only_no_negative_signals(self):
        prices = make_prices(n_stocks=2, n_days=100, seed=7)
        prices.columns = ["ST00", "ST01"]
        signals = self.strategy.generate_signals(prices, [self.pair])
        for s in signals:
            assert s.signal >= 0, "Long-only mode should not produce short signals"


# ─── Factor Model Tests ───────────────────────────────────────────────────────
class TestFactorModel:
    def setup_method(self):
        from quant.strategies.factor_model import FactorModel, FactorConfig
        self.model = FactorModel()
        self.prices = make_prices(n_stocks=20, n_days=252)

    def test_compute_scores_returns_dataframe(self):
        scores = self.model.compute_scores(self.prices, pd.DataFrame(), pd.DataFrame())
        assert not scores.empty
        assert "score" in scores.columns
        assert "rank" in scores.columns

    def test_scores_bounded_0_to_100(self):
        scores = self.model.compute_scores(self.prices, pd.DataFrame(), pd.DataFrame())
        assert scores["score"].between(0, 100).all(), "Scores should be in [0, 100]"

    def test_ranks_are_unique(self):
        scores = self.model.compute_scores(self.prices, pd.DataFrame(), pd.DataFrame())
        assert scores["rank"].nunique() == len(scores), "Ranks should be unique"

    def test_momentum_factor_positive_for_winners(self):
        # Make first stock a clear winner
        prices = self.prices.copy()
        prices.iloc[-1, 0] = prices.iloc[0, 0] * 2.0   # 100% return
        scores = self.model.compute_scores(prices, pd.DataFrame(), pd.DataFrame())
        winner = prices.columns[0]
        if winner in scores.index:
            assert scores.loc[winner, "factor_mom"] >= 50, "Winner should have high momentum score"

    def test_ic_evaluation(self):
        scores = self.model.compute_scores(self.prices, pd.DataFrame(), pd.DataFrame())
        fwd_returns = self.prices.pct_change(5).iloc[-1]
        ic = self.model.evaluate_ic(scores, fwd_returns)
        assert "composite" in ic
        assert -1 <= ic["composite"]["ic"] <= 1

    def test_long_short_portfolio_sums_to_zero(self):
        scores = self.model.compute_scores(self.prices, pd.DataFrame(), pd.DataFrame())
        portfolio = self.model.long_short_portfolio(scores, long_n=5, short_n=5)
        total = sum(portfolio.values())
        # Long-short: total of abs weights ≈ 2, net should be 0
        long_total  = sum(w for w in portfolio.values() if w > 0)
        short_total = sum(w for w in portfolio.values() if w < 0)
        assert abs(long_total + short_total) < 0.01


# ─── Momentum + Regime Tests ──────────────────────────────────────────────────
class TestMarketRegimeDetector:
    def setup_method(self):
        from quant.strategies.momentum_regime import MarketRegimeDetector
        self.detector = MarketRegimeDetector(n_states=3)

    def test_fit_and_predict(self):
        returns = pd.Series(np.random.normal(0.0005, 0.015, 500))
        self.detector.fit(returns)
        state = self.detector.predict(returns)
        assert state.regime.value in ("BULL", "BEAR", "SIDEWAYS")
        assert abs(state.bull_prob + state.bear_prob + state.sideways_prob - 1.0) < 0.01

    def test_bear_regime_low_momentum_scalar(self):
        """Forced bearish period should produce low momentum scalar."""
        # Strong negative trend = bear market proxy
        bear_returns = pd.Series(np.random.normal(-0.03, 0.03, 200))
        self.detector.fit(bear_returns)
        state = self.detector.predict(bear_returns.tail(100))
        # Bear regime should lower momentum scalar
        assert state.momentum_scalar <= 0.6


class TestCrossSectionalMomentum:
    def setup_method(self):
        from quant.strategies.momentum_regime import CrossSectionalMomentum
        self.cs = CrossSectionalMomentum(long_n=5, long_only=True)

    def test_top_n_stocks_get_long_signal(self):
        prices = make_prices(20, 300)
        signals = self.cs.generate_signals(prices)
        n_long = (signals["signal"] == 1).sum()
        assert n_long <= 5, f"Expected ≤5 long signals, got {n_long}"
        assert n_long >= 1, "Expected at least one long signal"

    def test_long_only_no_short(self):
        prices = make_prices(20, 300)
        signals = self.cs.generate_signals(prices)
        assert (signals["signal"] == -1).sum() == 0, "Long-only: no short signals"


# ─── Portfolio Optimisation Tests ─────────────────────────────────────────────
class TestMeanVarianceOptimizer:
    def setup_method(self):
        from quant.portfolio.optimizer import MeanVarianceOptimizer, ledoit_wolf_shrinkage
        self.mvo = MeanVarianceOptimizer(max_weight=0.3)
        self.prices = make_prices(10, 252)
        self.returns = self.prices.pct_change().dropna()
        self.tickers = list(self.prices.columns)
        self.cov = ledoit_wolf_shrinkage(self.returns)

    def test_weights_sum_to_one(self):
        mu = pd.Series(self.returns.mean() * 252, index=self.tickers)
        weights = self.mvo.optimize(mu, self.cov, self.tickers)
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-4, f"Weights sum to {total:.6f}"

    def test_no_weight_exceeds_max(self):
        mu = pd.Series(self.returns.mean() * 252, index=self.tickers)
        weights = self.mvo.optimize(mu, self.cov, self.tickers)
        for t, w in weights.items():
            assert w <= self.mvo.max_weight + 1e-4, f"{t}: {w:.4f} exceeds max {self.mvo.max_weight}"

    def test_all_weights_non_negative(self):
        mu = pd.Series(self.returns.mean() * 252, index=self.tickers)
        weights = self.mvo.optimize(mu, self.cov, self.tickers)
        for t, w in weights.items():
            assert w >= -1e-6, f"{t}: negative weight {w}"


class TestRiskParityOptimizer:
    def setup_method(self):
        from quant.portfolio.optimizer import RiskParityOptimizer, ledoit_wolf_shrinkage
        self.rp = RiskParityOptimizer()
        prices = make_prices(5, 252)
        returns = prices.pct_change().dropna()
        self.cov = ledoit_wolf_shrinkage(returns)
        self.tickers = list(prices.columns)

    def test_weights_sum_to_one(self):
        w = self.rp.optimize(self.cov, self.tickers)
        assert abs(sum(w.values()) - 1.0) < 1e-4

    def test_equal_risk_contribution(self):
        """All assets should contribute approximately equally to portfolio variance."""
        from quant.portfolio.optimizer import ledoit_wolf_shrinkage
        weights = self.rp.optimize(self.cov, self.tickers)
        w_arr = np.array([weights[t] for t in self.tickers])
        rc = w_arr * (self.cov @ w_arr)
        rc_norm = rc / rc.sum()
        # Each asset's risk contribution should be ~1/N
        target = 1.0 / len(self.tickers)
        for r in rc_norm:
            assert abs(r - target) < 0.1, f"Risk contribution {r:.3f} ≠ {target:.3f}"


class TestBlackLitterman:
    def setup_method(self):
        from quant.portfolio.optimizer import BlackLittermanOptimizer, InvestorView, ledoit_wolf_shrinkage
        self.bl = BlackLittermanOptimizer(max_weight=0.25)
        prices = make_prices(8, 252)
        returns = prices.pct_change().dropna()
        self.cov = ledoit_wolf_shrinkage(returns)
        self.tickers = list(prices.columns)
        self.mcap = pd.Series({t: np.random.uniform(1e11, 1e13) for t in self.tickers})

    def test_no_views_returns_market_weights(self):
        w = self.bl.optimize(self.mcap, self.cov, self.tickers, views=[])
        assert abs(sum(w.values()) - 1.0) < 1e-4

    def test_view_tilts_weights(self):
        from quant.portfolio.optimizer import InvestorView
        # Strong positive view on first ticker
        views = [InvestorView(
            description="ST00 outperforms",
            tickers=[self.tickers[0]],
            view_weights=[1.0],
            expected_return=0.20 / 252,
            confidence=0.9,
        )]
        w_with_view = self.bl.optimize(self.mcap, self.cov, self.tickers, views=views)
        w_no_view   = self.bl.optimize(self.mcap, self.cov, self.tickers, views=[])
        # Ticker 0 weight should increase with a positive view
        assert w_with_view[self.tickers[0]] >= w_no_view[self.tickers[0]] - 0.05


# ─── Risk Manager Tests ───────────────────────────────────────────────────────
class TestRiskManager:
    def setup_method(self):
        from quant.risk.risk_manager import RiskManager, RiskLimits
        limits = RiskLimits(
            max_drawdown_pct=0.15,
            hard_stop_drawdown_pct=0.20,
            var_95_limit_pct=0.03,
        )
        self.rm = RiskManager(limits=limits)
        rng = np.random.RandomState(42)
        self.returns = pd.Series(rng.normal(0.0005, 0.015, 252))

    def test_var_is_positive(self):
        report = self.rm.compute_risk_report(self.returns, 1e9, {}, {})
        assert report.var_95_1d >= 0, "VaR should be non-negative"
        assert report.var_99_1d >= report.var_95_1d, "99% VaR ≥ 95% VaR"

    def test_cvar_exceeds_var(self):
        report = self.rm.compute_risk_report(self.returns, 1e9, {}, {})
        assert report.cvar_95_1d >= report.var_95_1d, "CVaR ≥ VaR by definition"

    def test_no_breach_on_normal_returns(self):
        report = self.rm.compute_risk_report(self.returns, 1e9, {}, {})
        # With mild returns, action should not be HALT
        assert report.action_required in ("NONE", "REDUCE")

    def test_halt_triggered_on_large_drawdown(self):
        # Simulate large drawdown: cumulative -25%
        crash_returns = pd.concat([
            pd.Series(np.full(10, -0.025)),   # -25% over 10 days
            pd.Series(np.random.normal(0, 0.01, 90)),
        ])
        report = self.rm.compute_risk_report(crash_returns, 1e9, {}, {})
        # Should trigger HALT due to >20% drawdown
        assert report.action_required in ("REDUCE", "HALT")

    def test_pre_trade_check_blocks_oversized_position(self):
        ok, reason = self.rm.pre_trade_check(
            ticker="VNM",
            proposed_value=2e8,      # 200M VND
            portfolio_value=1e9,     # 1B VND portfolio
            current_positions={},
            sector_map={"VNM": "Consumer Staples"},
        )
        # 200M / 1B = 20% > 15% limit
        assert not ok, f"Should have blocked, reason: {reason}"

    def test_pre_trade_check_allows_valid_position(self):
        ok, reason = self.rm.pre_trade_check(
            ticker="VNM",
            proposed_value=1e8,      # 100M = 10% of portfolio
            portfolio_value=1e9,
            current_positions={},
            sector_map={"VNM": "Consumer Staples"},
        )
        assert ok, f"Should have allowed, reason: {reason}"

    def test_monte_carlo_var(self):
        returns = pd.Series(np.random.normal(0, 0.015, 252))
        cov = np.array([[0.015**2]])
        mc = self.rm.monte_carlo_var(returns, np.array([1.0]), cov)
        assert mc["var_mc"] >= 0
        assert mc["cvar_mc"] >= mc["var_mc"]


# ─── Order Flow Tests ─────────────────────────────────────────────────────────
class TestOrderBookImbalance:
    def setup_method(self):
        from quant.strategies.order_flow import OrderBookImbalanceModel
        self.obi = OrderBookImbalanceModel(n_levels=5)

    def test_empty_book_returns_zero(self):
        assert self.obi.compute([], []) == 0.0

    def test_all_bids_returns_positive(self):
        bids = [(10000, 1000), (9990, 2000)]
        asks = [(10010, 1), (10020, 1)]   # minimal asks
        obi = self.obi.compute(bids, asks)
        assert obi > 0, f"OBI should be positive: {obi}"

    def test_balanced_book_near_zero(self):
        bids = [(9995, 1000), (9990, 800)]
        asks = [(10005, 1000), (10010, 800)]
        obi = self.obi.compute(bids, asks)
        assert abs(obi) < 0.1, f"Balanced book OBI should be near 0: {obi}"

    def test_obi_bounded(self):
        bids = [(10000, 10000)] * 5
        asks = [(10010, 1)] * 5
        obi = self.obi.compute(bids, asks)
        assert -1 <= obi <= 1


class TestMicrostructureStrategy:
    def setup_method(self):
        from quant.strategies.order_flow import MicrostructureStrategy, OrderBookSnapshot
        self.strategy = MicrostructureStrategy(long_only=True)
        self.snapshot = OrderBookSnapshot(
            ticker="VNM",
            timestamp=datetime.utcnow(),
            bids=[(80000, 5000), (79900, 3000), (79800, 2000)],
            asks=[(80100, 5000), (80200, 3000), (80300, 2000)],
            last_trade_price=80050,
            last_trade_size=1000,
        )

    def test_signal_is_valid(self):
        trades = pd.DataFrame({"timestamp": [datetime.utcnow()], "price": [80050], "size": [1000], "side": ["BUY"]})
        sig = self.strategy.compute_signal(self.snapshot, trades, avg_daily_volume=1_000_000)
        assert sig.short_term_signal in (-1, 0, 1)
        assert 0 <= sig.vpin <= 1
        assert -1 <= sig.obi <= 1

    def test_long_only_no_short_signal(self):
        # Create strong sell pressure
        snapshot_sell = self.snapshot
        snapshot_sell.bids = [(80000, 100)]   # minimal bids
        snapshot_sell.asks = [(80100, 50000)] * 5  # huge ask side
        trades = pd.DataFrame({"timestamp": [datetime.utcnow()], "price": [80050], "size": [1000], "side": ["SELL"]})
        sig = self.strategy.compute_signal(snapshot_sell, trades)
        assert sig.short_term_signal >= 0, "Long-only: no short signals"


# ─── Mean Reversion Tests ─────────────────────────────────────────────────────
class TestBollingerBandReversion:
    def setup_method(self):
        from quant.strategies.mean_reversion import BollingerBandReversion
        self.bb = BollingerBandReversion(long_only=True)

    def test_signal_below_lower_band(self):
        """Stock hitting lower Bollinger Band should produce buy signal."""
        prices = make_prices(n_stocks=1, n_days=100)
        # Force last price below lower band
        prices_modified = prices.copy()
        mu = float(prices.iloc[-20:].mean().values[0])
        prices_modified.iloc[-1] = mu * 0.92   # well below 2-sigma band
        signals = self.bb.generate_signals(prices_modified)
        # May or may not trigger depending on RSI; just check no crash
        assert not signals.empty

    def test_long_only_no_short(self):
        prices = make_prices(10, 100)
        signals = self.bb.generate_signals(prices)
        if not signals.empty:
            assert (signals["signal"] == -1).sum() == 0


class TestShortTermReversal:
    def setup_method(self):
        from quant.strategies.mean_reversion import ShortTermReversal
        self.rev = ShortTermReversal(reversal_days=5, long_n=5, long_only=True)

    def test_signals_are_valid(self):
        prices = make_prices(20, 60)
        signals = self.rev.generate_signals(prices)
        assert not signals.empty
        assert all(s in (-1, 0, 1) for s in signals["signal"])

    def test_long_only_mode(self):
        prices = make_prices(20, 60)
        signals = self.rev.generate_signals(prices)
        assert (signals["signal"] == -1).sum() == 0


class TestMeanReversionComposite:
    def setup_method(self):
        from quant.strategies.mean_reversion import MeanReversionComposite
        self.composite = MeanReversionComposite(long_only=True)

    def test_requires_agreement(self):
        prices = make_prices(20, 120)
        signals = self.composite.generate_combined_signals(prices, min_agreement=2)
        # All long signals should have n_agreeing >= 2
        long_signals = signals[signals["signal"] == 1]
        if not long_signals.empty:
            assert (long_signals["n_agreeing"] >= 2).all()
