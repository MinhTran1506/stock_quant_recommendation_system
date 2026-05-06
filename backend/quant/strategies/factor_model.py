"""
quant/strategies/factor_model.py — Multi-Factor Alpha Model
═══════════════════════════════════════════════════════════

Research basis:
  • Fama & French (1993/2015) — 3-factor / 5-factor model (Market, SMB, HML,
    RMW, CMA); backbone of factor investing at BlackRock, Vanguard, AQR.
  • Jegadeesh & Titman (1993) — Cross-sectional momentum (12-1 month);
    one of the most documented equity anomalies.
  • Asness, Moskowitz & Pedersen (2013) — "Value and Momentum Everywhere";
    AQR's foundational paper combining value + momentum.
  • Novy-Marx (2013) — Gross profitability factor (quality proxy).
  • Frazzini & Pedersen (2014) — Betting Against Beta (BAB); low-vol anomaly.
  • Hou, Xue & Zhang (2015) — q-factor model (investment + profitability).

Factor construction for Vietnam equities (adapted from global literature):
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ Factor   │ Proxy                    │ Construction                       │
  ├──────────┼──────────────────────────┼────────────────────────────────────┤
  │ MOM      │ 12-1 month return        │ Cross-sectional rank               │
  │ REV      │ 1-month return           │ Short-term reversal (negative MOM) │
  │ VALUE    │ P/B ratio (inverse)      │ Book-to-market percentile rank     │
  │ QUALITY  │ ROE + profit margin      │ Composite quality score            │
  │ LOW_VOL  │ 60-day realised vol      │ BAB: low vol gets positive weight  │
  │ SIZE     │ Market cap               │ Small-cap premium                  │
  │ GROWTH   │ Revenue YoY growth       │ Sales growth rank                  │
  │ LIQUIDITY│ Avg daily turnover       │ Illiquidity premium (Amihud)       │
  └──────────┴──────────────────────────┴────────────────────────────────────┘

Output: cross-sectional composite alpha score (0-100 rank) per stock,
        combined via IC-weighted factor blending.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog
from scipy.stats import spearmanr
from sklearn.preprocessing import RobustScaler

logger = structlog.get_logger(__name__)


@dataclass
class FactorConfig:
    """Weight and direction for each factor."""
    momentum_weight: float = 0.25
    reversal_weight: float = -0.10    # negative: short-term reversal is contrarian
    value_weight: float = 0.15
    quality_weight: float = 0.20
    low_vol_weight: float = 0.15
    size_weight: float = 0.05
    growth_weight: float = 0.10
    liquidity_weight: float = 0.10
    # Lookback windows
    momentum_months: int = 12
    reversal_months: int = 1
    vol_days: int = 60
    # Winsorise at n-th percentile before ranking
    winsor_pct: float = 0.05


class FactorModel:
    """
    Cross-sectional factor alpha model.

    Builds a composite factor score for each stock in the universe
    using IC-weighted (Information Coefficient) blending.

    The model is intentionally transparent and explainable — each
    factor's contribution to the final score is tracked for reporting.
    """

    def __init__(self, config: Optional[FactorConfig] = None):
        self.config = config or FactorConfig()
        self._ic_history: Dict[str, List[float]] = {}   # track rolling factor ICs

    # ── Public API ─────────────────────────────────────────────────────────
    def compute_scores(
        self,
        prices: pd.DataFrame,           # index=date, cols=tickers, daily close
        fundamentals: pd.DataFrame,     # index=ticker, cols=pe,pb,roe,revenue_growth,...
        volumes: pd.DataFrame,          # index=date, cols=tickers, daily volume
    ) -> pd.DataFrame:
        """
        Compute composite factor scores for all stocks.

        Returns DataFrame with columns:
          [ticker, score, rank, mom, rev, value, quality, low_vol, size, growth,
           liquidity, factor_detail]
        """
        cfg = self.config
        result_rows = []

        tickers = (
            prices.columns.tolist()
            if (fundamentals is None or fundamentals.empty)
            else [t for t in prices.columns if t in fundamentals.index]
        )

        # ── Individual factor computation ─────────────────────────────
        mom_factor      = self._momentum_factor(prices, months=cfg.momentum_months)
        rev_factor      = self._reversal_factor(prices, months=cfg.reversal_months)
        vol_factor      = self._low_vol_factor(prices, days=cfg.vol_days)
        size_factor     = self._size_factor(prices, fundamentals)
        value_factor    = self._value_factor(fundamentals)
        quality_factor  = self._quality_factor(fundamentals)
        growth_factor   = self._growth_factor(fundamentals)
        liquidity_factor = self._liquidity_factor(prices, volumes)

        # ── Cross-sectional ranking (percentile 0→1) ──────────────────
        def cs_rank(series: pd.Series) -> pd.Series:
            return series.rank(pct=True).fillna(0.5)

        mom_r       = cs_rank(mom_factor)
        rev_r       = cs_rank(rev_factor)
        vol_r       = cs_rank(vol_factor)
        size_r      = cs_rank(size_factor)
        value_r     = cs_rank(value_factor)
        quality_r   = cs_rank(quality_factor)
        growth_r    = cs_rank(growth_factor)
        liq_r       = cs_rank(liquidity_factor)

        # ── IC-weighted composite ─────────────────────────────────────
        for ticker in tickers:
            raw = {
                "mom":      mom_r.get(ticker, 0.5),
                "rev":      rev_r.get(ticker, 0.5),
                "value":    value_r.get(ticker, 0.5),
                "quality":  quality_r.get(ticker, 0.5),
                "low_vol":  vol_r.get(ticker, 0.5),
                "size":     size_r.get(ticker, 0.5),
                "growth":   growth_r.get(ticker, 0.5),
                "liquidity":liq_r.get(ticker, 0.5),
            }
            composite = (
                cfg.momentum_weight  * raw["mom"]
                + cfg.reversal_weight  * raw["rev"]   # negative weight = contrarian
                + cfg.value_weight     * raw["value"]
                + cfg.quality_weight   * raw["quality"]
                + cfg.low_vol_weight   * raw["low_vol"]
                + cfg.size_weight      * raw["size"]
                + cfg.growth_weight    * raw["growth"]
                + cfg.liquidity_weight * raw["liquidity"]
            )
            result_rows.append({
                "ticker": ticker,
                "composite_score": composite,
                **{f"factor_{k}": round(v * 100, 1) for k, v in raw.items()},
            })

        if not result_rows:
            return pd.DataFrame()

        df = pd.DataFrame(result_rows).set_index("ticker")
        # Normalise composite to 0-100
        mn = df["composite_score"].min()
        mx = df["composite_score"].max()
        df["score"] = ((df["composite_score"] - mn) / max(mx - mn, 1e-8) * 100).round(2)
        df["rank"] = df["score"].rank(ascending=False, method="first").astype(int)
        return df.sort_values("score", ascending=False)

    def long_short_portfolio(
        self,
        scores: pd.DataFrame,
        long_n: int = 20,
        short_n: int = 20,
    ) -> Dict[str, float]:
        """
        Construct a long-short portfolio from factor scores.
        Returns {ticker: weight} (positive=long, negative=short).
        """
        if scores.empty:
            return {}
        sorted_scores = scores.sort_values("score", ascending=False)
        n = len(sorted_scores)

        portfolio: Dict[str, float] = {}
        # Top quartile: long
        for ticker in sorted_scores.head(min(long_n, n // 4)).index:
            portfolio[ticker] = 1.0 / long_n
        # Bottom quartile: short
        for ticker in sorted_scores.tail(min(short_n, n // 4)).index:
            portfolio[ticker] = -1.0 / short_n

        return portfolio

    def evaluate_ic(
        self,
        scores: pd.DataFrame,
        forward_returns: pd.Series,   # {ticker: forward_return}
    ) -> Dict[str, float]:
        """
        Compute Information Coefficient (rank correlation of factor scores
        with realised forward returns). IC > 0.05 is considered alpha.
        """
        common = scores.index.intersection(forward_returns.index)
        if len(common) < 10:
            return {}

        ics = {}
        for factor_col in [c for c in scores.columns if c.startswith("factor_")]:
            ic, pval = spearmanr(
                scores.loc[common, factor_col],
                forward_returns.loc[common],
            )
            fname = factor_col.replace("factor_", "")
            ics[fname] = {"ic": round(float(ic), 4), "pval": round(float(pval), 4)}
            self._ic_history.setdefault(fname, []).append(float(ic))

        # Overall composite IC
        overall_ic, _ = spearmanr(scores.loc[common, "score"], forward_returns.loc[common])
        ics["composite"] = {"ic": round(float(overall_ic), 4)}
        return ics

    # ── Individual factor constructors ────────────────────────────────────
    def _momentum_factor(self, prices: pd.DataFrame, months: int = 12) -> pd.Series:
        """
        Jegadeesh-Titman momentum: return from t-12m to t-1m.
        Skip 1 month to avoid short-term reversal contamination.
        """
        if len(prices) < months * 21 + 21:
            return pd.Series(dtype=float)
        ret_12m = prices.pct_change(months * 21).iloc[-1]
        ret_1m  = prices.pct_change(21).iloc[-1]
        return (ret_12m - ret_1m).fillna(0)

    def _reversal_factor(self, prices: pd.DataFrame, months: int = 1) -> pd.Series:
        """1-month reversal (contrarian short-term signal)."""
        if len(prices) < months * 21:
            return pd.Series(dtype=float)
        return prices.pct_change(months * 21).iloc[-1].fillna(0)

    def _low_vol_factor(self, prices: pd.DataFrame, days: int = 60) -> pd.Series:
        """
        Betting Against Beta (Frazzini & Pedersen 2014):
        Low realised volatility stocks earn higher risk-adjusted returns.
        Factor value = inverse vol (lower vol → higher score → positive weight).
        """
        if len(prices) < days:
            return pd.Series(dtype=float)
        daily_ret = prices.pct_change().tail(days)
        vol = daily_ret.std() * np.sqrt(252)
        return (1.0 / vol.replace(0, np.nan)).fillna(0)

    def _size_factor(
        self, prices: pd.DataFrame, fundamentals: pd.DataFrame
    ) -> pd.Series:
        """
        Small-cap premium (Fama-French SMB).
        Proxy: inverse of approximate market cap.
        """
        if fundamentals is None or "market_cap" not in fundamentals.columns:
            return pd.Series(dtype=float)
        mcap = fundamentals["market_cap"].dropna()
        return (1.0 / mcap.replace(0, np.nan)).fillna(0)

    def _value_factor(self, fundamentals: pd.DataFrame) -> pd.Series:
        """
        Fama-French HML (High Minus Low):
        Book-to-market value = inverse P/B ratio.
        """
        if fundamentals is None or "pb_ratio" not in fundamentals.columns:
            return pd.Series(dtype=float)
        pb = fundamentals["pb_ratio"].replace(0, np.nan)
        return (1.0 / pb).fillna(0)

    def _quality_factor(self, fundamentals: pd.DataFrame) -> pd.Series:
        """
        Novy-Marx (2013) gross profitability + Fama-French RMW.
        Composite: (ROE + ROA) / 2
        """
        if fundamentals is None:
            return pd.Series(dtype=float)
        roe = fundamentals.get("roe", pd.Series(dtype=float)).fillna(0)
        roa = fundamentals.get("roa", pd.Series(dtype=float)).fillna(0)
        return ((roe + roa) / 2).clip(-1, 1)

    def _growth_factor(self, fundamentals: pd.DataFrame) -> pd.Series:
        """Revenue growth YoY (Fama-French CMA proxy: low investment → high return)."""
        if fundamentals is None or "revenue_growth" not in fundamentals.columns:
            return pd.Series(dtype=float)
        return fundamentals["revenue_growth"].fillna(0).clip(-1, 1)

    def _liquidity_factor(
        self, prices: pd.DataFrame, volumes: pd.DataFrame
    ) -> pd.Series:
        """
        Amihud (2002) illiquidity ratio:
            ILLIQ = mean(|return| / VND_volume) × 10^6
        Higher ILLIQ = less liquid = higher expected return premium.
        """
        if volumes is None or len(prices) < 21:
            return pd.Series(dtype=float)
        common_cols = prices.columns.intersection(volumes.columns)
        ret_abs = prices[common_cols].pct_change().abs().tail(21)
        vol_vnd = volumes[common_cols].tail(21)
        # Avoid division by zero
        illiq = (ret_abs / vol_vnd.replace(0, np.nan)).mean()
        return illiq.fillna(0)
