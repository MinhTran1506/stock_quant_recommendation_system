"""
quant/strategies/mean_reversion.py — Mean Reversion Strategy Suite
═══════════════════════════════════════════════════════════════════

Research basis:
  • Poterba & Summers (1988) — Evidence of mean reversion in stock prices
    at 3–5 year horizons; foundational for contrarian investing.
  • DeBondt & Thaler (1985) — "Does the Stock Market Overreact?" —
    loser stocks outperform over 3–5 years (long-run reversal).
  • Jegadeesh (1990) — Short-term reversal: past 1-week losers outperform
    over the next week (microstructure liquidity provision effect).
  • Lehmann (1990) — Contrarian profits from short-term price reversals.
  • Lo & MacKinlay (1990) — Contrarian strategy profitability.

Mean reversion types implemented:
  ┌──────────────────────────────────────────────────────────────────┐
  │ Strategy            │ Horizon │ Mechanism                        │
  ├─────────────────────┼─────────┼──────────────────────────────────┤
  │ Bollinger Band Rev. │ Days    │ Price > 2σ band → fade move      │
  │ RSI Extremes        │ Days    │ RSI < 30 or > 70 → fade          │
  │ Short-term Reversal │ 1-week  │ Buy last-week losers             │
  │ Long-run Contrarian │ 3-5yr   │ Buy 5-year losers, sell winners  │
  │ Intraday Reversal   │ 1-60min │ Overextended intraday moves fade │
  └─────────────────────┴─────────┴──────────────────────────────────┘
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MeanReversionSignal:
    ticker: str
    strategy: str
    signal: int           # +1 buy | -1 sell | 0 neutral
    z_score: float        # how far from mean (normalised)
    indicator_value: float
    threshold: float
    confidence: float
    holding_days: int     # expected holding period


# ─── Bollinger Band Reversion ─────────────────────────────────────────────────
class BollingerBandReversion:
    """
    Classic Bollinger Band mean reversion.
    Buy when price closes below lower band; sell when above upper band.
    Add confirmation: RSI or volume surge filter.
    """

    def __init__(
        self,
        window: int = 20,
        n_std: float = 2.0,
        rsi_confirm_window: int = 14,
        long_only: bool = True,
    ):
        self.window = window
        self.n_std = n_std
        self.rsi_window = rsi_confirm_window
        self.long_only = long_only

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Returns DataFrame {ticker: signal} where signal ∈ {-1, 0, 1}.
        Positive = buy (price touched lower band), Negative = sell.
        """
        rows = []
        for ticker in prices.columns:
            close = prices[ticker].dropna()
            if len(close) < self.window + self.rsi_window:
                continue

            ma   = close.rolling(self.window).mean()
            std  = close.rolling(self.window).std()
            upper = ma + self.n_std * std
            lower = ma - self.n_std * std

            last_close = float(close.iloc[-1])
            last_lower = float(lower.iloc[-1])
            last_upper = float(upper.iloc[-1])
            last_ma    = float(ma.iloc[-1])
            last_std   = max(float(std.iloc[-1]), 1e-8)

            z_score    = (last_close - last_ma) / last_std

            # RSI confirmation
            delta  = close.diff()
            gain   = delta.clip(lower=0).rolling(self.rsi_window).mean()
            loss   = (-delta.clip(upper=0)).rolling(self.rsi_window).mean()
            rs     = gain / (loss + 1e-8)
            rsi    = float(100 - 100 / (1 + rs.iloc[-1]))

            if last_close <= last_lower and rsi < 35:
                signal, conf = 1, min(abs(z_score) / self.n_std, 1.0)
            elif last_close >= last_upper and rsi > 65 and not self.long_only:
                signal, conf = -1, min(abs(z_score) / self.n_std, 1.0)
            else:
                signal, conf = 0, 0.0

            rows.append(MeanReversionSignal(
                ticker=ticker,
                strategy="bollinger_reversion",
                signal=signal,
                z_score=round(z_score, 3),
                indicator_value=round(last_close, 2),
                threshold=round(last_lower if signal == 1 else last_upper, 2),
                confidence=round(conf, 3),
                holding_days=self.window // 2,
            ))

        return pd.DataFrame([vars(r) for r in rows]).set_index("ticker")


# ─── RSI Extreme Reversion ────────────────────────────────────────────────────
class RSIExtremesReversion:
    """
    Buy when RSI < oversold_threshold (default 30).
    Sell (or close long) when RSI > overbought_threshold (default 70).
    """

    def __init__(
        self,
        window: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        long_only: bool = True,
    ):
        self.window = window
        self.oversold = oversold
        self.overbought = overbought
        self.long_only = long_only

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for ticker in prices.columns:
            close = prices[ticker].dropna()
            if len(close) < self.window + 5:
                continue

            delta  = close.diff()
            gain   = delta.clip(lower=0).rolling(self.window).mean()
            loss   = (-delta.clip(upper=0)).rolling(self.window).mean()
            rs     = gain / (loss + 1e-8)
            rsi_series = 100 - 100 / (1 + rs)
            rsi    = float(rsi_series.iloc[-1])
            prev_rsi = float(rsi_series.iloc[-2]) if len(rsi_series) >= 2 else rsi

            # Require RSI to be turning (momentum exhaustion)
            if rsi < self.oversold and rsi > prev_rsi:  # RSI starting to recover
                signal = 1
                conf   = (self.oversold - rsi) / self.oversold
            elif rsi > self.overbought and rsi < prev_rsi and not self.long_only:
                signal = -1
                conf   = (rsi - self.overbought) / (100 - self.overbought)
            else:
                signal, conf = 0, 0.0

            rows.append(MeanReversionSignal(
                ticker=ticker,
                strategy="rsi_reversion",
                signal=signal,
                z_score=round((rsi - 50) / 25, 3),
                indicator_value=round(rsi, 2),
                threshold=self.oversold if signal == 1 else self.overbought,
                confidence=round(min(conf, 1.0), 3),
                holding_days=self.window // 2,
            ))

        return pd.DataFrame([vars(r) for r in rows]).set_index("ticker")


# ─── Short-term Reversal ──────────────────────────────────────────────────────
class ShortTermReversal:
    """
    Jegadeesh (1990) / Lehmann (1990):
    Buy stocks with lowest return over past week; sell highest.
    Pure liquidity provision / market-making alpha (contrarian).

    Academic consensus: ~0.5–1.0% weekly excess return, but decays
    quickly with capacity. Best for smaller portfolios.
    """

    def __init__(
        self,
        reversal_days: int = 5,     # 1-week reversal
        long_n: int = 20,
        short_n: int = 20,
        long_only: bool = True,
    ):
        self.reversal_days = reversal_days
        self.long_n = long_n
        self.short_n = short_n
        self.long_only = long_only

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        if len(prices) <= self.reversal_days:
            return pd.DataFrame()

        weekly_returns = prices.pct_change(self.reversal_days).iloc[-1].dropna()
        ranked = weekly_returns.rank(ascending=True)   # rank 1 = worst performer
        n = len(ranked)

        rows = []
        for ticker, ret in weekly_returns.items():
            rank = int(ranked[ticker])
            # Buy worst performers (reversal)
            if rank <= min(self.long_n, n // 4):
                signal, conf = 1, float(1 - ret)  # more negative → more confident
            # Sell best performers if short allowed
            elif rank > n - min(self.short_n, n // 4) and not self.long_only:
                signal, conf = -1, float(ret)
            else:
                signal, conf = 0, 0.0

            rows.append(MeanReversionSignal(
                ticker=ticker,
                strategy="short_term_reversal",
                signal=signal,
                z_score=round(float((ret - weekly_returns.mean()) / (weekly_returns.std() + 1e-8)), 3),
                indicator_value=round(float(ret), 4),
                threshold=0.0,
                confidence=round(min(abs(conf), 1.0), 3),
                holding_days=self.reversal_days,
            ))

        return pd.DataFrame([vars(r) for r in rows]).set_index("ticker")


# ─── Intraday Reversal ────────────────────────────────────────────────────────
class IntradayReversion:
    """
    Intraday mean reversion: stocks that gap-up or have large intraday
    moves tend to partially revert within the same session.

    Signal: if open-to-current > 2σ of historical daily range,
            fade the move (expect partial reversal by close).
    """

    def __init__(
        self,
        z_threshold: float = 1.5,
        lookback_days: int = 20,
        long_only: bool = True,
    ):
        self.z_threshold = z_threshold
        self.lookback_days = lookback_days
        self.long_only = long_only

    def generate_signals(
        self,
        intraday_prices: pd.DataFrame,   # today's 1-min bars, cols=tickers
        eod_prices: pd.DataFrame,        # historical daily OHLCV
    ) -> pd.DataFrame:
        rows = []
        for ticker in intraday_prices.columns:
            today = intraday_prices[ticker].dropna()
            hist  = eod_prices[ticker].dropna() if ticker in eod_prices.columns else pd.Series()

            if today.empty or len(hist) < self.lookback_days:
                continue

            open_price  = float(today.iloc[0])
            curr_price  = float(today.iloc[-1])

            if open_price <= 0:
                continue

            intraday_ret = (curr_price - open_price) / open_price

            # Historical intraday volatility
            hist_close = hist.tail(self.lookback_days)
            hist_returns = hist_close.pct_change().dropna()
            hist_vol = float(hist_returns.std())

            z = intraday_ret / max(hist_vol, 1e-8)

            if z < -self.z_threshold:
                # Stock has fallen too far intraday → buy reversal
                signal = 1
                conf   = min(abs(z) / (self.z_threshold * 2), 1.0)
            elif z > self.z_threshold and not self.long_only:
                # Overextended to upside → sell reversal
                signal = -1
                conf   = min(abs(z) / (self.z_threshold * 2), 1.0)
            else:
                signal, conf = 0, 0.0

            rows.append(MeanReversionSignal(
                ticker=ticker,
                strategy="intraday_reversion",
                signal=signal,
                z_score=round(z, 3),
                indicator_value=round(intraday_ret, 4),
                threshold=self.z_threshold,
                confidence=round(conf, 3),
                holding_days=1,   # close before end of day
            ))

        return pd.DataFrame([vars(r) for r in rows]).set_index("ticker")


# ─── Mean Reversion Composite ─────────────────────────────────────────────────
class MeanReversionComposite:
    """
    Combines all mean-reversion signals with configurable weights.
    Cross-validates signals across models before generating final output.
    """

    def __init__(self, long_only: bool = True):
        self.bb    = BollingerBandReversion(long_only=long_only)
        self.rsi   = RSIExtremesReversion(long_only=long_only)
        self.rev   = ShortTermReversal(long_only=long_only)

    def generate_combined_signals(
        self,
        prices: pd.DataFrame,
        min_agreement: int = 2,   # require at least N strategies to agree
    ) -> pd.DataFrame:
        """
        Generate signals that at least `min_agreement` strategies agree on.
        Returns DataFrame with columns: [signal, confidence, strategies_agreeing]
        """
        bb_sigs  = self.bb.generate_signals(prices)
        rsi_sigs = self.rsi.generate_signals(prices)
        rev_sigs = self.rev.generate_signals(prices)

        all_tickers = set(
            list(bb_sigs.index) + list(rsi_sigs.index) + list(rev_sigs.index)
        )

        rows = []
        for ticker in all_tickers:
            bb_s  = int(bb_sigs.loc[ticker, "signal"])  if ticker in bb_sigs.index  else 0
            rsi_s = int(rsi_sigs.loc[ticker, "signal"]) if ticker in rsi_sigs.index else 0
            rev_s = int(rev_sigs.loc[ticker, "signal"]) if ticker in rev_sigs.index else 0

            signals = [bb_s, rsi_s, rev_s]
            n_buy  = signals.count(1)
            n_sell = signals.count(-1)

            if n_buy >= min_agreement:
                final_signal = 1
                conf = (n_buy / 3) * max(
                    bb_sigs.loc[ticker, "confidence"]  if ticker in bb_sigs.index  else 0,
                    rsi_sigs.loc[ticker, "confidence"] if ticker in rsi_sigs.index else 0,
                    rev_sigs.loc[ticker, "confidence"] if ticker in rev_sigs.index else 0,
                )
            elif n_sell >= min_agreement:
                final_signal = -1
                conf = n_sell / 3
            else:
                final_signal = 0
                conf = 0.0

            rows.append({
                "ticker": ticker,
                "signal": final_signal,
                "confidence": round(conf, 4),
                "bb_signal": bb_s,
                "rsi_signal": rsi_s,
                "rev_signal": rev_s,
                "n_agreeing": max(n_buy, n_sell),
            })

        return pd.DataFrame(rows).set_index("ticker")
