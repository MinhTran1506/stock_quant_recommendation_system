"""
quant/strategies/momentum_regime.py — Momentum + Regime Detection
══════════════════════════════════════════════════════════════════

Research basis:
  • Jegadeesh & Titman (1993) — Cross-sectional momentum (12-1 month).
  • Moskowitz, Ooi & Pedersen (2012) — "Time Series Momentum" (TSMOM);
    trend-following across asset classes; core strategy at AHL/Man Group.
  • Daniel & Moskowitz (2016) — Momentum crashes during bear markets;
    motivates regime-conditional momentum scaling.
  • Hamilton (1989) — Hidden Markov Model (HMM) for regime detection;
    industry standard for identifying bull/bear/sideways market states.
  • Bloch (2025) — Relative Moving Average (RMA) for adaptive regime
    deployment (SSRN 5278107).

Architecture:
  ┌─────────────────────────────────────────────────────────────────┐
  │  MarketRegimeDetector (HMM)                                     │
  │    → detects Bull / Bear / Sideways using returns + volatility  │
  ├─────────────────────────────────────────────────────────────────┤
  │  CrossSectionalMomentum                                         │
  │    → ranks stocks by 12-1m return; buy top decile, sell bottom  │
  ├─────────────────────────────────────────────────────────────────┤
  │  TimeSeriesMomentum (TSMOM)                                     │
  │    → trades each stock by its own trend vs 12m average          │
  ├─────────────────────────────────────────────────────────────────┤
  │  RegimeAdaptiveMomentum                                         │
  │    → scales momentum exposure DOWN in Bear/crash regimes        │
  │    → full exposure in Bull, zero in Bear, half in Sideways      │
  └─────────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog
from hmmlearn import hmm

logger = structlog.get_logger(__name__)


class MarketRegime(str, Enum):
    BULL     = "BULL"       # trending up, low volatility
    BEAR     = "BEAR"       # trending down, high volatility
    SIDEWAYS = "SIDEWAYS"   # range-bound, medium volatility


@dataclass
class RegimeState:
    regime: MarketRegime
    bull_prob: float
    bear_prob: float
    sideways_prob: float
    vol_30d: float
    trend_12m: float
    momentum_scalar: float   # 0.0 – 1.0: how much to scale momentum signals


# ─── Market Regime Detector ───────────────────────────────────────────────────
class MarketRegimeDetector:
    """
    Hidden Markov Model (2- or 3-state) trained on:
      Feature 1: 21-day rolling return (trend signal)
      Feature 2: 21-day realised volatility (risk signal)

    States are labelled post-hoc by their mean return:
      Highest mean return  → BULL
      Lowest mean return   → BEAR
      Middle               → SIDEWAYS

    Hamilton (1989) pioneered Markov-switching regime models;
    this HMM variant is standard practice at quant hedge funds.
    """

    def __init__(self, n_states: int = 3, covariance_type: str = "full"):
        self.n_states = n_states
        self._model: Optional[hmm.GaussianHMM] = None
        self._state_map: Dict[int, MarketRegime] = {}

    def fit(self, market_returns: pd.Series, min_obs: int = 100) -> "MarketRegimeDetector":
        """
        Fit HMM on market-level (e.g., VN-Index) returns.
        market_returns: daily log returns of the index.
        """
        returns = market_returns.dropna().values
        if len(returns) < min_obs:
            logger.warning("Insufficient data for HMM", n=len(returns))
            return self

        # Feature matrix: [rolling_return, rolling_vol]
        series = pd.Series(returns)
        roll_ret = series.rolling(21).mean().fillna(0)
        roll_vol = series.rolling(21).std().fillna(series.std())
        X = np.column_stack([roll_ret.values, roll_vol.values])

        self._model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        self._model.fit(X)

        # Label states by mean return of each component
        means = self._model.means_[:, 0]   # first feature = return
        sorted_states = np.argsort(means)   # ascending
        if self.n_states == 3:
            self._state_map = {
                int(sorted_states[0]): MarketRegime.BEAR,
                int(sorted_states[1]): MarketRegime.SIDEWAYS,
                int(sorted_states[2]): MarketRegime.BULL,
            }
        else:
            self._state_map = {
                int(sorted_states[0]): MarketRegime.BEAR,
                int(sorted_states[1]): MarketRegime.BULL,
            }

        logger.info("HMM regime model fitted", n_states=self.n_states)
        return self

    def predict(self, market_returns: pd.Series) -> RegimeState:
        """Predict current market regime from recent returns."""
        if self._model is None:
            return RegimeState(
                regime=MarketRegime.BULL,
                bull_prob=1.0, bear_prob=0.0, sideways_prob=0.0,
                vol_30d=0.15, trend_12m=0.0, momentum_scalar=1.0,
            )

        returns = market_returns.dropna().values[-63:]   # use last 63 days
        series = pd.Series(returns)
        roll_ret = series.rolling(21).mean().fillna(0)
        roll_vol = series.rolling(21).std().fillna(series.std())
        X = np.column_stack([roll_ret.values, roll_vol.values])

        # Posterior state probabilities
        _, posteriors = self._model.score_samples(X)
        probs = posteriors[-1]   # latest bar

        # Map to named regimes
        regime_probs: Dict[MarketRegime, float] = {}
        for state_idx, regime in self._state_map.items():
            regime_probs[regime] = float(probs[state_idx])

        # Fill missing states with 0
        bull_p  = regime_probs.get(MarketRegime.BULL, 0.0)
        bear_p  = regime_probs.get(MarketRegime.BEAR, 0.0)
        side_p  = regime_probs.get(MarketRegime.SIDEWAYS, 0.0)
        dominant = max(regime_probs, key=regime_probs.get)

        # Momentum scalar: scale down in BEAR (Daniel & Moskowitz 2016)
        scalar = {
            MarketRegime.BULL:     1.0,
            MarketRegime.SIDEWAYS: 0.5,
            MarketRegime.BEAR:     0.0,   # avoid momentum crashes
        }[dominant]

        # Continuous scalar using bull probability, capped by dominant-regime ceiling
        # Also dampen by actual return level: HMM labels are relative, so when all
        # returns are negative the "BULL" state is just the least-bad state.
        # return_weight → 0 when window mean ≈ -2% daily, → 1 when ≈ +2% daily
        window_mean = float(pd.Series(returns).mean())
        return_weight = max(0.0, min(1.0, (window_mean + 0.02) / 0.04))
        dominant_cap = scalar  # 0.0 for BEAR, 0.5 for SIDEWAYS, 1.0 for BULL
        continuous_scalar = min(
            dominant_cap,
            (bull_p * 1.0 + side_p * 0.5 + bear_p * 0.0) * return_weight,
        )

        vol_30d = float(pd.Series(returns).rolling(21).std().iloc[-1] * np.sqrt(252))
        trend_12m = float(pd.Series(returns).sum()) if len(returns) >= 252 else 0.0

        return RegimeState(
            regime=dominant,
            bull_prob=round(bull_p, 4),
            bear_prob=round(bear_p, 4),
            sideways_prob=round(side_p, 4),
            vol_30d=round(vol_30d, 4),
            trend_12m=round(trend_12m, 4),
            momentum_scalar=round(continuous_scalar, 4),
        )


# ─── Cross-Sectional Momentum ─────────────────────────────────────────────────
class CrossSectionalMomentum:
    """
    Jegadeesh & Titman (1993) — Buy past winners, sell past losers.

    Formation period: 12 months minus 1 month (skip month to avoid reversal)
    Holding period:   configurable (1–3 months typical)
    Long-only:        buy top quintile only (Vietnam: no shorting)
    """

    def __init__(
        self,
        formation_months: int = 12,
        skip_months: int = 1,
        long_n: int = 20,
        long_only: bool = True,
    ):
        self.formation_months = formation_months
        self.skip_months = skip_months
        self.long_n = long_n
        self.long_only = long_only

    def generate_signals(
        self,
        prices: pd.DataFrame,       # index=date, cols=tickers
        regime: Optional[RegimeState] = None,
    ) -> pd.DataFrame:
        """
        Returns DataFrame {ticker, momentum_return, signal, weight}.
        """
        days_formation = self.formation_months * 21
        days_skip      = self.skip_months * 21

        if len(prices) < days_formation + days_skip:
            return pd.DataFrame()

        # Return from t-12m to t-1m
        end_price   = prices.iloc[-(days_skip + 1)]
        start_price = prices.iloc[-(days_formation + days_skip)]
        mom_returns = (end_price / start_price.replace(0, np.nan) - 1).fillna(0)

        # Cross-sectional rank
        ranked = mom_returns.rank(ascending=False)
        n = len(ranked)

        rows = []
        for ticker, ret in mom_returns.items():
            rank = int(ranked[ticker])
            # Long signal: top quintile
            is_long = rank <= max(1, n // 5)
            # Short signal: bottom quintile (disabled for Vietnam long-only)
            is_short = (rank > n - max(1, n // 5)) and not self.long_only

            signal = 1 if is_long else (-1 if is_short else 0)

            # Regime scaling
            scalar = regime.momentum_scalar if regime else 1.0
            if regime and regime.regime == MarketRegime.BEAR:
                signal = 0   # no momentum trades in bear market

            rows.append({
                "ticker": ticker,
                "momentum_return": round(float(ret), 6),
                "rank": rank,
                "signal": signal * (1 if signal == 0 else 1),
                "weight": (1.0 / self.long_n * scalar) if signal == 1 else 0.0,
            })

        df = pd.DataFrame(rows).set_index("ticker")
        # Keep only top-N longs
        longs = df[df["signal"] == 1].nsmallest(self.long_n, "rank")
        df.loc[df["signal"] == 1, "signal"] = 0     # reset first
        df.loc[longs.index, "signal"] = 1
        return df


# ─── Time-Series Momentum ─────────────────────────────────────────────────────
class TimeSeriesMomentum:
    """
    Moskowitz, Ooi & Pedersen (2012) — each stock trades its own trend.

    Signal: sign of 12-month return (positive → long; negative → avoid/short)
    Position size: inversely proportional to ex-ante volatility (vol-scaling)
    This is the core of CTA / trend-following strategies.
    """

    def __init__(
        self,
        lookback_months: int = 12,
        vol_target: float = 0.15,     # annualised vol target per position
        long_only: bool = True,
    ):
        self.lookback_days = lookback_months * 21
        self.vol_target = vol_target
        self.long_only = long_only

    def generate_signals(
        self,
        prices: pd.DataFrame,
        regime: Optional[RegimeState] = None,
    ) -> pd.DataFrame:
        """Returns {ticker, signal, weight, vol_scale}."""
        if len(prices) < self.lookback_days + 21:
            return pd.DataFrame()

        current = prices.iloc[-1]
        past    = prices.iloc[-self.lookback_days]
        returns_12m = (current / past.replace(0, np.nan) - 1).fillna(0)

        # Realised vol (60-day)
        daily_ret = prices.pct_change().tail(60)
        vol_60d   = daily_ret.std() * np.sqrt(252)

        rows = []
        for ticker in prices.columns:
            ret = float(returns_12m.get(ticker, 0))
            vol = float(vol_60d.get(ticker, 0.20))
            if vol < 1e-8:
                vol = 0.20

            # TSMOM signal: positive trend → long
            signal = 1 if ret > 0 else (-1 if not self.long_only else 0)

            # Regime gate
            if regime and regime.regime == MarketRegime.BEAR:
                signal = 0

            # Vol-scaled weight: w = vol_target / realised_vol
            vol_scale = min(self.vol_target / vol, 3.0)   # cap at 3×
            weight    = vol_scale / len(prices.columns) if signal == 1 else 0.0
            weight   *= (regime.momentum_scalar if regime else 1.0)

            rows.append({
                "ticker":    ticker,
                "return_12m": round(ret, 4),
                "vol_60d":   round(vol, 4),
                "signal":    signal,
                "vol_scale": round(vol_scale, 4),
                "weight":    round(weight, 6),
            })

        return pd.DataFrame(rows).set_index("ticker")


# ─── Regime-Adaptive Momentum ─────────────────────────────────────────────────
class RegimeAdaptiveMomentum:
    """
    Combines HMM regime detection with both CS and TS momentum.

    In Bull regime:     full momentum exposure (CS + TSMOM)
    In Sideways:        half exposure (CS only)
    In Bear regime:     zero momentum; switch to mean-reversion signals

    This addresses the Daniel & Moskowitz (2016) momentum crash problem.
    """

    def __init__(
        self,
        long_only: bool = True,
        cs_long_n: int = 20,
        ts_vol_target: float = 0.15,
        n_hmm_states: int = 3,
    ):
        self.regime_detector = MarketRegimeDetector(n_states=n_hmm_states)
        self.cs_momentum = CrossSectionalMomentum(long_n=cs_long_n, long_only=long_only)
        self.ts_momentum = TimeSeriesMomentum(vol_target=ts_vol_target, long_only=long_only)
        self._fitted = False

    def fit(self, index_returns: pd.Series) -> "RegimeAdaptiveMomentum":
        """Fit HMM on market index returns (e.g., VN-Index)."""
        self.regime_detector.fit(index_returns)
        self._fitted = True
        return self

    def generate_signals(
        self,
        prices: pd.DataFrame,
        index_returns: pd.Series,
    ) -> Tuple[pd.DataFrame, RegimeState]:
        """
        Generate combined signals and current regime state.
        Returns (signals_df, regime_state).
        """
        # Current regime
        regime = self.regime_detector.predict(index_returns) if self._fitted else None

        # CS momentum signals
        cs_signals = self.cs_momentum.generate_signals(prices, regime)
        # TS momentum signals
        ts_signals = self.ts_momentum.generate_signals(prices, regime)

        if cs_signals.empty and ts_signals.empty:
            return pd.DataFrame(), regime or RegimeState(
                regime=MarketRegime.BULL, bull_prob=1.0, bear_prob=0.0,
                sideways_prob=0.0, vol_30d=0.15, trend_12m=0.0, momentum_scalar=1.0
            )

        # Combine: average signal when both agree; CS takes priority
        all_tickers = set(
            list(cs_signals.index) + list(ts_signals.index)
        )
        combined_rows = []
        for ticker in all_tickers:
            cs_sig = int(cs_signals.loc[ticker, "signal"]) if ticker in cs_signals.index else 0
            ts_sig = int(ts_signals.loc[ticker, "signal"]) if ticker in ts_signals.index else 0
            cs_wt  = float(cs_signals.loc[ticker, "weight"]) if ticker in cs_signals.index else 0
            ts_wt  = float(ts_signals.loc[ticker, "weight"]) if ticker in ts_signals.index else 0

            # Combined signal: require CS and TS to agree for full weight
            combined_sig = 1 if cs_sig == 1 and ts_sig == 1 else (
                1 if cs_sig == 1 else 0
            )
            combined_wt = (cs_wt + ts_wt) / 2 if combined_sig == 1 else 0.0

            combined_rows.append({
                "ticker":     ticker,
                "signal":     combined_sig,
                "weight":     round(combined_wt, 6),
                "cs_signal":  cs_sig,
                "ts_signal":  ts_sig,
                "regime":     regime.regime.value if regime else "BULL",
            })

        return pd.DataFrame(combined_rows).set_index("ticker"), regime
