"""
data/feature_store/features.py — Feature computation pipeline.

Computes a rich set of technical, microstructure, and fundamental features
for each stock, caches hot features in Redis, and archives feature snapshots
to S3 / Feast for training.

Feature groups:
  - Price/Return features (momentum, volatility, trend)
  - Volume features (OBV, VWAP, volume ratios)
  - Technical indicators (RSI, MACD, Bollinger Bands, ATR, etc.)
  - Microstructure features (bid-ask spread, order imbalance, trade intensity)
  - Fundamental ratios (P/E, P/B, ROE, debt-to-equity)
  - Sentiment features (from NLP pipeline)
  - Cross-sectional features (sector rank, market-relative momentum)
"""
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import redis.asyncio as aioredis
import structlog
import ta  # Technical analysis library

from config import get_settings

settings = get_settings()
logger = structlog.get_logger(__name__)


class FeatureStore:
    """
    Central feature computation and serving layer.

    Hot path (serving): Redis cache → compute on miss
    Cold path (training): PostgreSQL / S3 feature snapshots
    """

    FEATURE_VERSION = "v1"

    def __init__(self):
        self._redis: Optional[aioredis.Redis] = None

    async def _get_redis(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = await aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._redis

    def _cache_key(self, ticker: str, feature_set: str) -> str:
        return f"features:{self.FEATURE_VERSION}:{ticker}:{feature_set}"

    # ── Public API ─────────────────────────────────────────────────────────
    async def get_features(
        self,
        ticker: str,
        df: Optional[pd.DataFrame] = None,
        use_cache: bool = True,
    ) -> Dict[str, float]:
        """
        Return the full feature vector for a ticker, suitable for model inference.
        If df (OHLCV DataFrame) is provided, compute from it; else load from cache.
        """
        cache_key = self._cache_key(ticker, "full")

        # 1. Try Redis cache first
        if use_cache:
            redis = await self._get_redis()
            cached = await redis.get(cache_key)
            if cached:
                return json.loads(cached)

        # 2. Compute features
        if df is None:
            raise ValueError("DataFrame required to compute features on cache miss")

        features = self.compute_all_features(df, ticker=ticker)

        # 3. Cache result
        redis = await self._get_redis()
        await redis.setex(
            cache_key,
            settings.redis_feature_ttl_seconds,
            json.dumps({k: float(v) if v is not None else None
                        for k, v in features.items()}),
        )
        return features

    async def set_features(self, ticker: str, features: Dict[str, float]) -> None:
        """Write pre-computed features to cache (called by Airflow pipeline)."""
        redis = await self._get_redis()
        cache_key = self._cache_key(ticker, "full")
        await redis.setex(
            cache_key,
            settings.redis_feature_ttl_seconds,
            json.dumps(features),
        )

    # ── Feature computation ────────────────────────────────────────────────
    def compute_all_features(
        self,
        df: pd.DataFrame,
        ticker: str = "",
    ) -> Dict[str, Any]:
        """
        Compute all feature groups from an OHLCV DataFrame.

        DataFrame must have columns: [date, open, high, low, close, volume]
        sorted ascending by date, with no gaps.
        """
        df = df.copy().sort_values("date").reset_index(drop=True)
        df = self._validate_and_clean(df)

        features: Dict[str, Any] = {"ticker": ticker, "computed_at": datetime.utcnow().isoformat()}

        features.update(self._price_return_features(df))
        features.update(self._volatility_features(df))
        features.update(self._volume_features(df))
        features.update(self._technical_indicator_features(df))
        features.update(self._trend_features(df))
        features.update(self._seasonality_features(df))

        return features

    def compute_microstructure_features(
        self,
        bids: List[List[float]],
        asks: List[List[float]],
        recent_trades: Optional[List[Dict]] = None,
    ) -> Dict[str, float]:
        """
        Compute order-book microstructure features for HFT/intraday models.
        Inputs: bids/asks are [[price, qty], ...] sorted best-first.
        """
        features = {}

        if bids and asks:
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            mid_price = (best_bid + best_ask) / 2.0

            features["bid_ask_spread"] = best_ask - best_bid
            features["relative_spread"] = (best_ask - best_bid) / mid_price
            features["mid_price"] = mid_price

            # Order book imbalance (top 5 levels)
            bid_vol = sum(q for _, q in bids[:5])
            ask_vol = sum(q for _, q in asks[:5])
            total_vol = bid_vol + ask_vol
            features["order_imbalance"] = (bid_vol - ask_vol) / total_vol if total_vol else 0.0
            features["bid_depth_top5"] = bid_vol
            features["ask_depth_top5"] = ask_vol

            # Weighted mid price
            features["weighted_mid_price"] = (
                (best_bid * asks[0][1] + best_ask * bids[0][1])
                / (bids[0][1] + asks[0][1])
                if bids[0][1] + asks[0][1] > 0 else mid_price
            )

        # Trade intensity
        if recent_trades:
            trades_df = pd.DataFrame(recent_trades)
            features["trade_count_1min"] = len(trades_df)
            features["trade_volume_1min"] = float(trades_df.get("volume", pd.Series([0])).sum())
            buy_vol = float(trades_df[trades_df.get("side", "") == "BUY"].get("volume", pd.Series([0])).sum())
            total = features["trade_volume_1min"]
            features["buy_trade_ratio"] = buy_vol / total if total else 0.5

        return features

    # ── Private helpers ────────────────────────────────────────────────────
    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {"date", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        df = df.dropna(subset=["close"])
        return df

    def _price_return_features(self, df: pd.DataFrame) -> Dict[str, float]:
        close = df["close"]
        feats = {}

        for window in [1, 3, 5, 10, 20, 60, 120, 252]:
            if len(close) > window:
                feats[f"return_{window}d"] = float(close.pct_change(window).iloc[-1])

        # Momentum: 12-month - 1-month (Jegadeesh-Titman)
        if len(close) > 252:
            feats["momentum_12m1m"] = float(
                close.pct_change(252).iloc[-1] - close.pct_change(21).iloc[-1]
            )

        # Distance from 52-week high/low
        if len(close) >= 252:
            high_52w = close.rolling(252).max().iloc[-1]
            low_52w = close.rolling(252).min().iloc[-1]
            feats["pct_from_52w_high"] = float((close.iloc[-1] - high_52w) / high_52w)
            feats["pct_from_52w_low"] = float((close.iloc[-1] - low_52w) / low_52w)

        return feats

    def _volatility_features(self, df: pd.DataFrame) -> Dict[str, float]:
        close = df["close"]
        log_ret = np.log(close / close.shift(1)).dropna()
        feats = {}

        for window in [5, 10, 20, 60]:
            if len(log_ret) >= window:
                feats[f"realised_vol_{window}d"] = float(
                    log_ret.rolling(window).std().iloc[-1] * np.sqrt(252)
                )

        # Parkinson high-low volatility estimator
        if len(df) >= 20:
            hl_ratio = np.log(df["high"] / df["low"])
            feats["parkinson_vol_20d"] = float(
                np.sqrt(hl_ratio.rolling(20).apply(
                    lambda x: (1 / (4 * np.log(2))) * np.mean(x**2)
                ).iloc[-1] * 252)
            )

        # Volatility ratio (short/long) — measures vol regime
        if "realised_vol_5d" in feats and "realised_vol_60d" in feats:
            if feats["realised_vol_60d"] > 0:
                feats["vol_ratio_5_60"] = feats["realised_vol_5d"] / feats["realised_vol_60d"]

        return feats

    def _volume_features(self, df: pd.DataFrame) -> Dict[str, float]:
        feats = {}
        close = df["close"]
        volume = df["volume"]

        # Volume ratios
        for window in [5, 20]:
            if len(volume) > window:
                vol_ma = volume.rolling(window).mean()
                feats[f"volume_ratio_{window}d"] = float(
                    volume.iloc[-1] / vol_ma.iloc[-1] if vol_ma.iloc[-1] > 0 else 1.0
                )

        # OBV (On-Balance Volume) trend
        if len(df) >= 20:
            obv = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
            obv_change = obv.pct_change(20).iloc[-1]
            feats["obv_trend_20d"] = float(obv_change) if pd.notna(obv_change) else 0.0

        # VWAP deviation
        if "high" in df.columns and "low" in df.columns:
            typical_price = (df["high"] + df["low"] + close) / 3
            vwap = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
            feats["vwap_deviation"] = float(
                (close.iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1]
                if pd.notna(vwap.iloc[-1]) and vwap.iloc[-1] > 0 else 0.0
            )

        return feats

    def _technical_indicator_features(self, df: pd.DataFrame) -> Dict[str, float]:
        feats = {}
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # RSI
        if len(close) >= 14:
            rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
            feats["rsi_14"] = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else 50.0

        # MACD
        if len(close) >= 26:
            macd_ind = ta.trend.MACD(close)
            macd_val = macd_ind.macd().iloc[-1]
            macd_sig = macd_ind.macd_signal().iloc[-1]
            feats["macd"] = float(macd_val) if pd.notna(macd_val) else 0.0
            feats["macd_signal"] = float(macd_sig) if pd.notna(macd_sig) else 0.0
            feats["macd_hist"] = feats["macd"] - feats["macd_signal"]

        # Bollinger Bands
        if len(close) >= 20:
            bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
            bb_upper = bb.bollinger_hband().iloc[-1]
            bb_lower = bb.bollinger_lband().iloc[-1]
            bb_mid = bb.bollinger_mavg().iloc[-1]
            if pd.notna(bb_upper) and bb_upper > bb_lower:
                feats["bb_pct_b"] = float((close.iloc[-1] - bb_lower) / (bb_upper - bb_lower))
                feats["bb_bandwidth"] = float((bb_upper - bb_lower) / bb_mid)

        # ATR (Average True Range)
        if len(df) >= 14:
            atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
            feats["atr_14"] = float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else 0.0
            feats["atr_pct"] = feats["atr_14"] / float(close.iloc[-1]) if close.iloc[-1] > 0 else 0.0

        # Stochastic
        if len(df) >= 14:
            stoch = ta.momentum.StochasticOscillator(high, low, close)
            k = stoch.stoch().iloc[-1]
            d = stoch.stoch_signal().iloc[-1]
            feats["stoch_k"] = float(k) if pd.notna(k) else 50.0
            feats["stoch_d"] = float(d) if pd.notna(d) else 50.0

        return feats

    def _trend_features(self, df: pd.DataFrame) -> Dict[str, float]:
        feats = {}
        close = df["close"]

        # Moving average cross signals
        for short, long in [(5, 20), (10, 50), (20, 200)]:
            if len(close) > long:
                ma_short = close.rolling(short).mean().iloc[-1]
                ma_long = close.rolling(long).mean().iloc[-1]
                if pd.notna(ma_short) and pd.notna(ma_long) and ma_long > 0:
                    feats[f"ma_cross_{short}_{long}"] = float((ma_short - ma_long) / ma_long)
                    feats[f"price_vs_ma{long}"] = float((close.iloc[-1] - ma_long) / ma_long)

        # ADX (trend strength)
        if len(df) >= 14 and "high" in df.columns and "low" in df.columns:
            adx_ind = ta.trend.ADXIndicator(df["high"], df["low"], close, window=14)
            adx = adx_ind.adx().iloc[-1]
            feats["adx_14"] = float(adx) if pd.notna(adx) else 25.0

        return feats

    def _seasonality_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calendar / seasonality features."""
        feats = {}
        if "date" in df.columns:
            last_date = pd.to_datetime(df["date"].iloc[-1])
            feats["day_of_week"] = float(last_date.dayofweek)
            feats["month"] = float(last_date.month)
            feats["quarter"] = float(last_date.quarter)
            feats["is_month_end"] = float(last_date.is_month_end)
            feats["is_quarter_end"] = float(last_date.is_quarter_end)
        return feats
