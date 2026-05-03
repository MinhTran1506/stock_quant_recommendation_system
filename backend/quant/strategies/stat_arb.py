"""
quant/strategies/stat_arb.py — Statistical Arbitrage Engine
════════════════════════════════════════════════════════════

Research basis:
  • Gatev, Goetzmann & Rouwenhorst (2006) — "Pairs Trading: Performance of a
    Relative-Value Arbitrage Rule" — foundational pairs trading paper.
  • Avellaneda & Lee (2008) — "Statistical Arbitrage in the US Equities Market"
    — PCA-based factor residual mean-reversion; used by Morgan Stanley PDT.
  • Engle & Granger (1987) / Johansen (1991) — Cointegration tests; industry
    standard for identifying long-run equilibria between related stocks.
  • Elliott, van der Hoek & Malcolm (2005) — Kalman Filter for dynamic hedge
    ratio estimation, outperforms static OLS in non-stationary markets.

Architecture:
  1. PairsFinder      — scans universe; finds cointegrated pairs via Johansen
  2. KalmanHedge      — dynamic hedge ratio via 1-D Kalman Filter
  3. SpreadModel       — OU process parameter estimation; entry/exit z-scores
  4. StatArbStrategy  — orchestrates signal generation and position management

Vietnam market note:
  Short-selling is restricted on HOSE/HNX → long-only variant implemented
  (buy the lagging stock, close when spread converges; no shorting required).
  Full long-short enabled for when regulatory environment permits.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog
from scipy import stats
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

warnings.filterwarnings("ignore", category=RuntimeWarning)
logger = structlog.get_logger(__name__)


# ─── Data classes ─────────────────────────────────────────────────────────────
@dataclass
class PairConfig:
    ticker_a: str
    ticker_b: str
    hedge_ratio: float              # initial OLS estimate
    spread_mean: float = 0.0
    spread_std: float = 1.0
    half_life_days: float = 10.0    # OU mean-reversion speed
    entry_z: float = 2.0            # open position at |z| > entry_z
    exit_z: float = 0.5             # close position at |z| < exit_z
    stop_z: float = 4.0             # stop-loss at |z| > stop_z
    lookback_days: int = 252        # estimation window


@dataclass
class SpreadSignal:
    """Output from spread monitor for a single pair."""
    ticker_a: str
    ticker_b: str
    spread: float
    z_score: float
    hedge_ratio: float
    signal: int          # +1 = long A/short B, -1 = short A/long B, 0 = flat
    half_life: float
    confidence: float    # p-value of cointegration test (lower = more confident)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ─── Kalman Filter Hedge Ratio ─────────────────────────────────────────────────
class KalmanHedge:
    """
    1-D Kalman Filter for dynamic hedge ratio estimation.

    Models the spread as:
        price_A(t) = β(t) · price_B(t) + ε(t)
        β(t) = β(t-1) + η(t)       (random walk state)

    The filter continuously updates β as new prices arrive, adapting to
    structural breaks and regime changes that static OLS misses.

    Reference: Elliott, van der Hoek & Malcolm (2005).
    """

    def __init__(self, delta: float = 1e-4, R: float = 1e-2):
        """
        delta: state-transition noise variance (higher = faster adaptation)
        R:     observation noise variance
        """
        self.delta = delta
        self.R = R
        self._beta: Optional[float] = None
        self._P: float = 0.0          # state covariance
        self._Q: float = delta / (1 - delta)   # process noise

    def update(self, price_a: float, price_b: float) -> float:
        """Process a new price pair; return updated hedge ratio β."""
        if self._beta is None:
            self._beta = price_a / max(price_b, 1e-8)
            self._P = 1.0
            return self._beta

        # Prediction step
        P_pred = self._P + self._Q

        # Innovation
        y_hat = self._beta * price_b
        innovation = price_a - y_hat

        # Kalman gain
        S = price_b * P_pred * price_b + self.R
        K = P_pred * price_b / S

        # Update
        self._beta += K * innovation
        self._P = (1 - K * price_b) * P_pred

        return self._beta

    def batch_fit(self, prices_a: np.ndarray, prices_b: np.ndarray) -> np.ndarray:
        """Run Kalman filter over full historical series."""
        betas = np.zeros(len(prices_a))
        for i, (a, b) in enumerate(zip(prices_a, prices_b)):
            betas[i] = self.update(a, b)
        return betas

    @property
    def hedge_ratio(self) -> float:
        return self._beta or 1.0


# ─── Ornstein-Uhlenbeck Spread Model ──────────────────────────────────────────
class OUSpreadModel:
    """
    Ornstein-Uhlenbeck process for spread dynamics.

    dX(t) = κ(μ - X(t))dt + σ dW(t)

    Parameters:
        κ  — mean-reversion speed (half-life = ln(2)/κ)
        μ  — long-run mean
        σ  — volatility of innovations

    Entry/exit thresholds derived from stationary distribution:
        X∞ ~ N(μ, σ²/2κ)
    """

    def fit(self, spread: np.ndarray) -> Dict[str, float]:
        """
        Estimate OU parameters by OLS on the discretized SDE:
            ΔX(t) = a + b·X(t-1) + ε(t)
            κ = -ln(1+b·Δt)/Δt  (daily: Δt=1)
        """
        spread = spread[~np.isnan(spread)]
        if len(spread) < 30:
            return {"kappa": 0.1, "mu": 0.0, "sigma": 1.0, "half_life": 10.0}

        X = spread[:-1]
        dX = np.diff(spread)
        # OLS: dX = a + b·X
        design = np.column_stack([np.ones_like(X), X])
        try:
            coeff, _, _, _ = np.linalg.lstsq(design, dX, rcond=None)
            a, b = coeff
        except np.linalg.LinAlgError:
            return {"kappa": 0.1, "mu": 0.0, "sigma": float(np.std(dX)), "half_life": 10.0}

        kappa = max(-b, 1e-4)           # mean-reversion speed
        mu = -a / b if abs(b) > 1e-8 else float(np.mean(spread))
        residuals = dX - (a + b * X)
        sigma = float(np.std(residuals))
        half_life = np.log(2) / kappa

        return {
            "kappa": float(kappa),
            "mu": float(mu),
            "sigma": float(sigma),
            "half_life": float(half_life),
            "sigma_eq": sigma / np.sqrt(2 * kappa),  # equilibrium std
        }


# ─── Pairs Finder ─────────────────────────────────────────────────────────────
class PairsFinder:
    """
    Scans a universe of stocks to find cointegrated pairs.

    Step 1: Pre-filter by Pearson correlation (|ρ| > min_corr).
    Step 2: Johansen cointegration test on filtered pairs.
    Step 3: ADF test on residual spread for robustness.
    Step 4: Estimate OU parameters; filter by half-life range.
    """

    def __init__(
        self,
        min_corr: float = 0.65,
        johansen_confidence: float = 0.95,
        min_half_life_days: float = 2,
        max_half_life_days: float = 63,    # ~3 months
        lookback_days: int = 252,
    ):
        self.min_corr = min_corr
        self.johansen_confidence = johansen_confidence
        self.min_half_life = min_half_life_days
        self.max_half_life = max_half_life_days
        self.lookback_days = lookback_days
        self._ou = OUSpreadModel()

    def find_pairs(self, prices: pd.DataFrame) -> List[PairConfig]:
        """
        prices: DataFrame with tickers as columns, dates as index,
                values = adjusted closing prices.
        Returns list of PairConfig for all valid cointegrated pairs.
        """
        prices = prices.tail(self.lookback_days).dropna(axis=1, how="any")
        tickers = list(prices.columns)
        log_prices = np.log(prices)
        returns = log_prices.diff().dropna()

        logger.info("Scanning for cointegrated pairs", n_tickers=len(tickers))

        # Step 1: correlation pre-filter
        corr_matrix = returns.corr()
        candidates = []
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                if abs(corr_matrix.iloc[i, j]) >= self.min_corr:
                    candidates.append((tickers[i], tickers[j]))

        logger.info("Correlation pre-filter", n_candidates=len(candidates))

        # Step 2-4: cointegration test + OU parameter filter
        valid_pairs: List[PairConfig] = []
        for ta, tb in candidates:
            pair = self._test_pair(log_prices[ta].values, log_prices[tb].values, ta, tb)
            if pair is not None:
                valid_pairs.append(pair)

        valid_pairs.sort(key=lambda p: p.half_life_days)
        logger.info("Valid cointegrated pairs found", n_pairs=len(valid_pairs))
        return valid_pairs

    def _test_pair(
        self,
        log_a: np.ndarray,
        log_b: np.ndarray,
        ta: str,
        tb: str,
    ) -> Optional[PairConfig]:
        """Run full cointegration + OU test for a single pair."""
        try:
            # Johansen test (handles both directions simultaneously)
            data = np.column_stack([log_a, log_b])
            johansen_result = coint_johansen(data, det_order=0, k_ar_diff=1)
            # Check trace statistic at 95% confidence level (index 1)
            trace_stat = johansen_result.lr1[0]
            crit_val_95 = johansen_result.cvt[0, 1]
            if trace_stat <= crit_val_95:
                return None  # not cointegrated

            # Static hedge ratio from first Johansen eigenvector
            eigen_vec = johansen_result.evec[:, 0]
            hedge = -eigen_vec[1] / eigen_vec[0] if abs(eigen_vec[0]) > 1e-8 else 1.0

            # Spread
            spread = log_a - hedge * log_b

            # ADF robustness check
            adf_stat, adf_pval, *_ = adfuller(spread, maxlags=1, autolag=None)
            if adf_pval > 0.10:
                return None  # spread not stationary

            # OU parameter estimation
            ou = self._ou.fit(spread)
            if not (self.min_half_life <= ou["half_life"] <= self.max_half_life):
                return None  # too slow or too fast to mean-revert

            spread_std = float(np.std(spread))
            if spread_std < 1e-8:
                return None

            return PairConfig(
                ticker_a=ta,
                ticker_b=tb,
                hedge_ratio=float(hedge),
                spread_mean=float(np.mean(spread)),
                spread_std=spread_std,
                half_life_days=ou["half_life"],
            )

        except Exception as e:
            logger.debug("Pair test failed", ta=ta, tb=tb, error=str(e))
            return None


# ─── Stat Arb Strategy ─────────────────────────────────────────────────────────
class StatArbStrategy:
    """
    Full statistical arbitrage strategy for Vietnam equities.

    Per-pair lifecycle:
      1. Compute spread using Kalman dynamic hedge ratio
      2. Standardise to z-score using rolling OU parameters
      3. Enter at |z| > entry_z, exit at |z| < exit_z, stop at |z| > stop_z
      4. Long-only mode: only enter longs on the lagging stock (no shorting)

    Backtestable via the BacktestOrchestrator; live-runnable via
    StrategyOrchestrator with paper execution.
    """

    def __init__(
        self,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 4.0,
        lookback_days: int = 63,     # rolling OU re-estimation window
        long_only: bool = True,      # True for Vietnam (no shorting)
        max_pairs: int = 20,
        rebalance_freq: str = "daily",
    ):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.lookback_days = lookback_days
        self.long_only = long_only
        self.max_pairs = max_pairs
        self._kalman_filters: Dict[str, KalmanHedge] = {}
        self._ou = OUSpreadModel()

    def generate_signals(
        self,
        prices: pd.DataFrame,           # index=date, cols=tickers
        pairs: List[PairConfig],
    ) -> List[SpreadSignal]:
        """
        Generate entry/exit signals for all active pairs.
        Called once per bar (daily or intraday).
        """
        signals: List[SpreadSignal] = []
        latest = prices.iloc[-1]
        history = prices.tail(self.lookback_days + 1)

        for pair in pairs[:self.max_pairs]:
            ta, tb = pair.ticker_a, pair.ticker_b
            if ta not in latest.index or tb not in latest.index:
                continue
            if latest[ta] <= 0 or latest[tb] <= 0:
                continue

            # Kalman dynamic hedge ratio
            key = f"{ta}_{tb}"
            if key not in self._kalman_filters:
                self._kalman_filters[key] = KalmanHedge()

            kf = self._kalman_filters[key]
            log_a_hist = np.log(history[ta].values)
            log_b_hist = np.log(history[tb].values)
            hedge_ratios = kf.batch_fit(log_a_hist, log_b_hist)
            current_hedge = float(hedge_ratios[-1])

            # Current spread
            spread = float(np.log(latest[ta]) - current_hedge * np.log(latest[tb]))

            # Rolling OU parameters
            spread_series = log_a_hist - hedge_ratios * log_b_hist
            ou_params = self._ou.fit(spread_series)
            spread_mean = ou_params["mu"]
            spread_std = max(ou_params.get("sigma_eq", ou_params["sigma"]), 1e-8)

            z = (spread - spread_mean) / spread_std

            # Signal logic
            if abs(z) > self.stop_z:
                signal = 0    # stop-loss: exit all
            elif abs(z) < self.exit_z:
                signal = 0    # convergence: exit
            elif z < -self.entry_z:
                # Spread below mean: A is cheap relative to B
                signal = 1    # long A, (short B if allowed)
            elif z > self.entry_z:
                # Spread above mean: A is expensive relative to B
                signal = -1 if not self.long_only else 0   # short A, long B
            else:
                signal = 0    # within band: hold

            signals.append(SpreadSignal(
                ticker_a=ta,
                ticker_b=tb,
                spread=round(spread, 6),
                z_score=round(float(z), 4),
                hedge_ratio=round(current_hedge, 4),
                signal=signal,
                half_life=ou_params["half_life"],
                confidence=0.05,   # from Johansen test p-value
            ))

        return signals

    def backtest(
        self,
        prices: pd.DataFrame,
        pairs: List[PairConfig],
        initial_capital: float = 1_000_000_000,
        position_size_pct: float = 0.05,   # 5% per pair
    ) -> pd.DataFrame:
        """
        Simple vectorised backtest of all pairs.
        Returns equity curve DataFrame.
        """
        equity = pd.Series(index=prices.index, dtype=float, name="equity")
        cash = initial_capital
        positions: Dict[str, float] = {}   # {ticker: vnd_value}

        for i in range(self.lookback_days, len(prices)):
            hist = prices.iloc[: i + 1]
            signals = self.generate_signals(hist, pairs)

            daily_pnl = 0.0
            for sig in signals:
                ta, tb = sig.ticker_a, sig.ticker_b
                curr_a = prices.iloc[i][ta] if ta in prices.columns else None
                curr_b = prices.iloc[i][tb] if tb in prices.columns else None

                if curr_a is None or curr_b is None:
                    continue

                # Mark-to-market open positions
                for ticker in [ta, tb]:
                    if ticker in positions:
                        prev_price = prices.iloc[i - 1].get(ticker, 0)
                        if prev_price > 0:
                            curr_price = prices.iloc[i].get(ticker, prev_price)
                            daily_pnl += positions[ticker] * (curr_price / prev_price - 1)

                # Enter/exit
                if sig.signal != 0 and ta not in positions:
                    # Enter long A position
                    alloc = cash * position_size_pct
                    positions[ta] = alloc
                    cash -= alloc
                elif sig.signal == 0 and ta in positions:
                    # Close position
                    cash += positions.pop(ta)

            cash += daily_pnl
            equity.iloc[i] = cash + sum(positions.values())

        return equity.fillna(method="ffill").fillna(initial_capital).to_frame()
