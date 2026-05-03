"""
quant/strategies/order_flow.py — Order Flow & Microstructure Alpha
══════════════════════════════════════════════════════════════════

Research basis:
  • Glosten & Milgrom (1985) — Bid-ask spread decomposition: adverse
    selection component reveals informed trading activity.
  • Kyle (1985) — Market depth model; lambda (price impact) as signal.
  • Easley et al. (2012) — VPIN (Volume-Synchronized Probability of
    Informed Trading); widely used by HFT firms and market makers.
  • Cont, Kukanov & Stoikov (2014) — Order book imbalance as short-term
    price predictor (1–10 second horizon).
  • Almgren & Chriss (2001) — Optimal execution; minimise market impact.
  • Hendershott, Jones & Menkveld (2011) — Algorithmic trading and
    market quality; HFT provides liquidity in normal conditions.
  • Hagströmer & Nordén (2013) — HFT taxonomy: market-making vs
    directional strategies.

Microstructure signals implemented:
  ┌──────────────────────────────────────────────────────────────────┐
  │ Signal              │ Horizon  │ Formula                         │
  ├─────────────────────┼──────────┼─────────────────────────────────┤
  │ OBI (Order Book     │ 1-5 min  │ (BidVol-AskVol)/(BidVol+AskVol)│
  │  Imbalance)         │          │                                 │
  │ VPIN                │ 30-60min │ |BuyVol-SellVol|/TotalVol       │
  │ Trade Imbalance     │ 5-15 min │ Buy%-Sell% of trade count       │
  │ Spread Alpha        │ 1-30 min │ Spread vs rolling mean spread   │
  │ Price Impact (λ)    │ 1-5 min  │ ΔPrice / ΔVolume (Kyle lambda)  │
  │ Toxic Flow          │ 15-60min │ High VPIN + momentum divergence │
  └─────────────────────┴──────────┴─────────────────────────────────┘

Vietnam market note:
  HOSE processes ~300-500K trades/day with ATS (Automated Trading System).
  Intraday data available from Vietstock/FiinGroup at 1-min resolution.
  Full tick data requires direct exchange connection (Phase 3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


# ─── Data containers ──────────────────────────────────────────────────────────
@dataclass
class OrderBookSnapshot:
    ticker: str
    timestamp: datetime
    bids: List[Tuple[float, int]]   # [(price, qty), ...] best first
    asks: List[Tuple[float, int]]
    last_trade_price: float = 0.0
    last_trade_size: int = 0
    last_trade_side: str = "UNKNOWN"  # BUY | SELL | UNKNOWN


@dataclass
class MicrostructureSignal:
    ticker: str
    timestamp: str
    obi: float              # Order Book Imbalance [-1, 1]
    vpin: float             # VPIN [0, 1] — higher = more informed flow
    trade_imbalance: float  # Buy fraction of trades [0, 1]
    kyle_lambda: float      # Price impact coefficient
    spread_bps: float       # Current spread in basis points
    spread_ratio: float     # Current spread / rolling mean spread
    toxic_flow: bool        # True if VPIN is high and trend is fading
    short_term_signal: int  # +1 buy pressure | -1 sell pressure | 0 neutral
    confidence: float       # Signal strength [0, 1]


# ─── Order Book Imbalance ─────────────────────────────────────────────────────
class OrderBookImbalanceModel:
    """
    Cont, Kukanov & Stoikov (2014):
    OBI = (V_bid - V_ask) / (V_bid + V_ask)

    Strong positive OBI → buying pressure → short-term price increase.
    Uses multiple depth levels (weighted by distance from mid).
    """

    def __init__(self, n_levels: int = 5, depth_decay: float = 0.5):
        """
        n_levels:    number of order book levels to use
        depth_decay: exponential weight decay per level (closer levels weighted more)
        """
        self.n_levels = n_levels
        self.depth_decay = depth_decay

    def compute(
        self, bids: List[Tuple[float, int]], asks: List[Tuple[float, int]]
    ) -> float:
        """
        Returns OBI ∈ [-1, 1].
        Positive = buy pressure, Negative = sell pressure.
        """
        if not bids or not asks:
            return 0.0

        bid_vol = 0.0
        ask_vol = 0.0

        for i in range(min(self.n_levels, len(bids))):
            weight = self.depth_decay ** i
            bid_vol += bids[i][1] * weight

        for i in range(min(self.n_levels, len(asks))):
            weight = self.depth_decay ** i
            ask_vol += asks[i][1] * weight

        total = bid_vol + ask_vol
        if total < 1e-8:
            return 0.0

        return float((bid_vol - ask_vol) / total)

    def compute_series(
        self,
        book_snapshots: List[OrderBookSnapshot],
    ) -> pd.Series:
        """Compute OBI time series from a list of snapshots."""
        timestamps = [s.timestamp for s in book_snapshots]
        obis = [self.compute(s.bids, s.asks) for s in book_snapshots]
        return pd.Series(obis, index=pd.DatetimeIndex(timestamps), name="obi")


# ─── VPIN Calculator ──────────────────────────────────────────────────────────
class VPINCalculator:
    """
    Easley, López de Prado & O'Hara (2012) — Volume-Synchronized
    Probability of Informed Trading.

    VPIN estimates the probability that any given trade comes from an
    informed trader. High VPIN → expect increased volatility and spreads.

    Simplified (bucket-based):
        VPIN = |V_buy - V_sell| / V_total
    averaged over the last N volume buckets.

    Production note: Use the full Bayesian VPIN from Easley et al.
    for live deployment; this is the simplified fast approximation.
    """

    def __init__(self, n_buckets: int = 50, bucket_size_pct: float = 0.01):
        """
        n_buckets:       rolling window of buckets for VPIN average
        bucket_size_pct: each bucket = this fraction of daily volume
        """
        self.n_buckets = n_buckets
        self.bucket_size_pct = bucket_size_pct
        self._bucket_imbalances: List[float] = []

    def update(self, trades: pd.DataFrame, avg_daily_volume: float) -> float:
        """
        trades: DataFrame with columns [timestamp, price, size, side]
                side: 'BUY' | 'SELL' (or inferred from tick rule)
        Returns current VPIN estimate [0, 1].
        """
        if trades.empty or avg_daily_volume <= 0:
            return 0.0

        bucket_size = max(1, int(avg_daily_volume * self.bucket_size_pct))
        trades = trades.sort_values("timestamp")

        # Infer trade direction using tick rule if side not provided
        if "side" not in trades.columns or trades["side"].isna().all():
            price_change = trades["price"].diff().fillna(0)
            trades = trades.copy()
            trades["side"] = np.where(price_change >= 0, "BUY", "SELL")

        # Fill volume buckets
        cumulative_vol = 0
        buy_vol = 0.0
        sell_vol = 0.0

        for _, row in trades.iterrows():
            cumulative_vol += row["size"]
            if row["side"] == "BUY":
                buy_vol += row["size"]
            else:
                sell_vol += row["size"]

            if cumulative_vol >= bucket_size:
                imbalance = abs(buy_vol - sell_vol) / max(cumulative_vol, 1)
                self._bucket_imbalances.append(imbalance)
                if len(self._bucket_imbalances) > self.n_buckets:
                    self._bucket_imbalances.pop(0)
                cumulative_vol = 0
                buy_vol = 0.0
                sell_vol = 0.0

        if not self._bucket_imbalances:
            return 0.0

        return float(np.mean(self._bucket_imbalances[-self.n_buckets:]))


# ─── Kyle Lambda (Price Impact) ───────────────────────────────────────────────
class KyleLambdaEstimator:
    """
    Kyle (1985) price impact coefficient λ.
    Estimates how much price moves per unit of signed volume:
        ΔP_t = λ · OrderFlow_t + ε_t

    Higher λ = less liquid, larger impact per trade.
    Used for: position sizing, execution cost estimation.
    """

    def estimate(
        self,
        prices: np.ndarray,
        signed_volumes: np.ndarray,  # positive=buy, negative=sell
        window: int = 100,
    ) -> float:
        """OLS estimate of Kyle lambda over the given window."""
        if len(prices) < window or len(signed_volumes) < window:
            return 0.0

        dp = np.diff(prices[-window:])
        sv = signed_volumes[-window + 1:]

        if len(dp) < 10 or np.std(sv) < 1e-8:
            return 0.0

        # Weighted OLS (recent observations weighted more)
        weights = np.exp(np.linspace(-1, 0, len(dp)))
        try:
            lam = float(np.cov(dp * weights, sv * weights)[0, 1] /
                        max(np.var(sv * weights), 1e-8))
        except Exception:
            lam = 0.0

        return max(0.0, lam)


# ─── Microstructure Strategy ──────────────────────────────────────────────────
class MicrostructureStrategy:
    """
    Combined short-term microstructure alpha strategy.

    Signal generation pipeline (per ticker, per bar):
      1. OBI from current order book (level 1-5)
      2. VPIN from recent trades (last 30 min)
      3. Kyle λ from intraday price-volume
      4. Spread regime (narrow vs. wide)
      5. Composite signal with confidence weighting

    Typical holding period: 1–30 minutes (intraday)
    Not suitable for overnight positions.

    Vietnam market:
      - T session: 09:15–11:30 and 13:00–14:45 (ATC at 14:45)
      - Pre-open: 09:00–09:15 (order matching)
      - Avoid first/last 15 minutes (elevated adverse selection)
    """

    def __init__(
        self,
        obi_threshold: float = 0.3,      # |OBI| > this → signal
        vpin_high_threshold: float = 0.7, # VPIN > this → toxic flow warning
        spread_wide_threshold: float = 2.0,  # spread > 2× mean → avoid
        min_confidence: float = 0.4,
        long_only: bool = True,
    ):
        self.obi_threshold = obi_threshold
        self.vpin_high_threshold = vpin_high_threshold
        self.spread_wide_threshold = spread_wide_threshold
        self.min_confidence = min_confidence
        self.long_only = long_only

        self._obi_model   = OrderBookImbalanceModel()
        self._vpin_calc   = VPINCalculator()
        self._kyle_est    = KyleLambdaEstimator()
        self._spread_history: Dict[str, List[float]] = {}

    def compute_signal(
        self,
        snapshot: OrderBookSnapshot,
        trades: pd.DataFrame,           # recent trades (last 30 min)
        avg_daily_volume: float = 0,
        price_history: Optional[np.ndarray] = None,
    ) -> MicrostructureSignal:
        """
        Compute microstructure signal for a single ticker.
        """
        ticker = snapshot.ticker
        ts = snapshot.timestamp.isoformat()

        # 1. Order Book Imbalance
        obi = self._obi_model.compute(snapshot.bids, snapshot.asks)

        # 2. VPIN
        vpin = self._vpin_calc.update(trades, avg_daily_volume) if not trades.empty else 0.0

        # 3. Trade imbalance (fraction of trades that are buys)
        if not trades.empty and "side" in trades.columns:
            n_buy  = (trades["side"] == "BUY").sum()
            n_total = len(trades)
            trade_imbalance = float(n_buy / max(n_total, 1))
        else:
            trade_imbalance = 0.5

        # 4. Bid-ask spread
        best_bid = snapshot.bids[0][0] if snapshot.bids else 0
        best_ask = snapshot.asks[0][0] if snapshot.asks else 0
        mid = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else 1
        spread_bps = float((best_ask - best_bid) / mid * 10_000) if mid > 0 else 0

        # Spread relative to rolling mean
        hist = self._spread_history.setdefault(ticker, [])
        hist.append(spread_bps)
        if len(hist) > 200:
            hist.pop(0)
        mean_spread = float(np.mean(hist)) if hist else spread_bps
        spread_ratio = spread_bps / max(mean_spread, 1e-8)

        # 5. Kyle lambda
        kyle_lam = 0.0
        if price_history is not None and len(price_history) > 20:
            signed_vol = np.sign(np.diff(price_history)) * avg_daily_volume / len(price_history)
            kyle_lam = self._kyle_est.estimate(price_history, signed_vol)

        # 6. Toxic flow detection: high VPIN + OBI diverges from recent trend
        is_toxic = (vpin > self.vpin_high_threshold and
                    spread_ratio > self.spread_wide_threshold)

        # 7. Composite signal
        # Buy signal: strong positive OBI + low VPIN + narrow spread + buy-side pressure
        buy_score = (
            max(obi, 0) * 0.40 +
            (trade_imbalance - 0.5) * 2 * 0.35 +   # normalise to [-1, 1]
            (1 - vpin) * 0.15 +
            max(1 - spread_ratio / 2, 0) * 0.10
        )
        sell_score = (
            max(-obi, 0) * 0.40 +
            (0.5 - trade_imbalance) * 2 * 0.35 +
            (1 - vpin) * 0.15 +
            max(1 - spread_ratio / 2, 0) * 0.10
        )

        if is_toxic:
            # When toxic flow is detected, fade the signal (contrarian)
            signal = 0
            confidence = 0.0
        elif buy_score > self.obi_threshold and buy_score > sell_score:
            signal = 1
            confidence = min(buy_score, 1.0)
        elif sell_score > self.obi_threshold and not self.long_only:
            signal = -1
            confidence = min(sell_score, 1.0)
        else:
            signal = 0
            confidence = 0.0

        if confidence < self.min_confidence:
            signal = 0

        return MicrostructureSignal(
            ticker=ticker,
            timestamp=ts,
            obi=round(obi, 4),
            vpin=round(vpin, 4),
            trade_imbalance=round(trade_imbalance, 4),
            kyle_lambda=round(kyle_lam, 8),
            spread_bps=round(spread_bps, 2),
            spread_ratio=round(spread_ratio, 3),
            toxic_flow=is_toxic,
            short_term_signal=signal,
            confidence=round(confidence, 4),
        )

    def batch_signals(
        self,
        snapshots: Dict[str, OrderBookSnapshot],
        trades_by_ticker: Dict[str, pd.DataFrame],
        avg_volumes: Dict[str, float],
    ) -> List[MicrostructureSignal]:
        """Generate signals for all tickers simultaneously."""
        signals = []
        for ticker, snapshot in snapshots.items():
            trades = trades_by_ticker.get(ticker, pd.DataFrame())
            avg_vol = avg_volumes.get(ticker, 0)
            sig = self.compute_signal(snapshot, trades, avg_vol)
            signals.append(sig)
        return sorted(signals, key=lambda s: abs(s.obi), reverse=True)

    # ── Optimal execution sizing ────────────────────────────────────────────
    def almgren_chriss_size(
        self,
        target_value: float,        # VND value to trade
        avg_daily_volume: float,    # ADV in shares
        price: float,               # current price
        kyle_lambda: float,         # estimated price impact
        risk_aversion: float = 1e-6,
        trading_horizon_minutes: int = 30,
    ) -> Dict[str, float]:
        """
        Almgren & Chriss (2001) optimal execution.
        Returns recommended trade schedule (fraction per minute).

        Minimises: Expected Cost + risk_aversion × Variance of Cost
        """
        n_steps = trading_horizon_minutes
        total_shares = int(target_value / max(price, 1))

        if total_shares <= 0 or avg_daily_volume <= 0:
            return {"immediate": 1.0, "schedule": [1.0]}

        # Temporary impact: η = Kyle λ
        eta = kyle_lambda * price / max(avg_daily_volume, 1)
        # Permanent impact: γ ≈ η/2 (simplified)
        gamma = eta / 2

        # Almgren-Chriss optimal schedule
        try:
            kappa = np.sqrt(risk_aversion / (2 * eta)) if eta > 0 else 0.1
            j = np.arange(1, n_steps + 1)
            # Sinh-based schedule
            fractions = (np.sinh(kappa * (n_steps - j + 1)) -
                         np.sinh(kappa * (n_steps - j))) / np.sinh(kappa * n_steps)
            fractions = np.clip(fractions, 0, 1)
            fractions /= fractions.sum()
        except Exception:
            # Fallback: uniform schedule (TWAP)
            fractions = np.ones(n_steps) / n_steps

        return {
            "total_shares": total_shares,
            "schedule_fractions": fractions.tolist(),
            "estimated_impact_bps": float(gamma * total_shares / avg_daily_volume * 10000),
        }


# ─── Market Making Signal ──────────────────────────────────────────────────────
class MarketMakingSignal:
    """
    Avellaneda & Stoikov (2008) market making model.
    Computes optimal bid-ask quotes around reservation price.

    Reservation price: r = S - q·γ·σ²·T
    where:
      S = mid price, q = current inventory, γ = risk aversion,
      σ = volatility, T = remaining time

    Optimal spread: δ* = γ·σ²·T + (2/γ)·ln(1 + γ/k)
    where k = order arrival rate.

    Vietnam application: only valid for highly liquid stocks
    (VNM, VIC, TCB, BID, VCB) with tight spreads and high turnover.
    """

    def __init__(
        self,
        risk_aversion: float = 0.1,
        order_arrival_rate: float = 10.0,   # orders per minute
        dt: float = 1 / 252 / 390,          # 1 minute as fraction of year
    ):
        self.gamma = risk_aversion
        self.k = order_arrival_rate
        self.dt = dt

    def compute_quotes(
        self,
        mid_price: float,
        volatility: float,      # annualised
        inventory: int,         # current position in shares
        remaining_time_min: float = 30,
    ) -> Dict[str, float]:
        """
        Returns optimal bid and ask quotes.
        """
        T = remaining_time_min / (252 * 390)  # convert to years
        sigma2 = volatility**2 / 252 / 390    # per-minute variance

        # Reservation price (adjusted for inventory risk)
        r = mid_price - inventory * self.gamma * sigma2 * T * mid_price

        # Optimal half-spread
        try:
            half_spread = (self.gamma * sigma2 * T * mid_price +
                           (2 / self.gamma) * np.log(1 + self.gamma / self.k))
        except (ValueError, ZeroDivisionError):
            half_spread = mid_price * 0.001

        optimal_bid = max(r - half_spread / 2, mid_price * 0.99)
        optimal_ask = min(r + half_spread / 2, mid_price * 1.01)

        return {
            "reservation_price": round(r, 2),
            "optimal_bid": round(optimal_bid, 2),
            "optimal_ask": round(optimal_ask, 2),
            "half_spread": round(half_spread, 4),
            "spread_bps": round(half_spread / mid_price * 10_000, 2),
            "inventory_skew": round(mid_price - r, 4),
        }
