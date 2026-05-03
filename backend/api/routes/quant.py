"""
api/routes/quant.py — Quantitative Trading Strategy API
════════════════════════════════════════════════════════

Endpoints:
  GET  /quant/strategies              — list available strategies with status
  POST /quant/stat-arb/scan           — find cointegrated pairs in universe
  GET  /quant/stat-arb/signals        — current spread z-scores
  POST /quant/factor-model/scores     — compute factor scores for all stocks
  GET  /quant/factor-model/rankings   — latest factor rankings
  GET  /quant/momentum/signals        — regime-aware momentum signals
  GET  /quant/regime                  — current market regime (HMM)
  POST /quant/portfolio/optimize      — run portfolio optimisation
  GET  /quant/risk/report             — current risk metrics
  POST /quant/risk/stress-test        — historical stress scenarios
  POST /quant/rl-agent/train          — trigger RL agent training (async)
  GET  /quant/rl-agent/weights        — current RL agent portfolio weights
"""
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.middleware.auth import get_current_user
from db.session import get_db
from db.models import Stock, EODPrice
import pandas as pd
import numpy as np

router = APIRouter()
logger = structlog.get_logger(__name__)

# Strategy registry (in-memory state for now; productionize with DB/Redis)
_strategy_state: Dict[str, Any] = {}
_hmm_model = None    # loaded lazily
_factor_cache: Dict = {}


# ─── Schemas ──────────────────────────────────────────────────────────────────
class StatArbScanRequest(BaseModel):
    tickers: Optional[List[str]] = None   # None = use full universe
    lookback_days: int = 252
    min_corr: float = 0.65
    max_half_life_days: int = 63

class StatArbSignalOut(BaseModel):
    ticker_a: str
    ticker_b: str
    z_score: float
    spread: float
    hedge_ratio: float
    signal: int
    half_life: float
    timestamp: str

class FactorScoreOut(BaseModel):
    ticker: str
    score: float
    rank: int
    factor_mom: Optional[float]
    factor_value: Optional[float]
    factor_quality: Optional[float]
    factor_low_vol: Optional[float]
    factor_growth: Optional[float]

class PortfolioOptimizeRequest(BaseModel):
    tickers: List[str]
    method: str = "black_litterman"   # mean_variance | black_litterman | risk_parity | equal_weight
    max_weight: float = 0.15
    lookback_days: int = 252

class RiskReportOut(BaseModel):
    var_95_1d: float
    cvar_95_1d: float
    current_drawdown: float
    max_drawdown: float
    annualised_vol: float
    sharpe_ratio: float
    sortino_ratio: float
    breaches: List[str]
    action_required: str
    timestamp: str


# ─── Helpers ──────────────────────────────────────────────────────────────────
async def _load_prices(
    db: AsyncSession,
    tickers: Optional[List[str]],
    lookback_days: int = 252,
) -> pd.DataFrame:
    """Load EOD prices for tickers from DB."""
    from sqlalchemy import desc
    q = select(Stock).where(Stock.is_active == True)
    if tickers:
        q = q.where(Stock.ticker.in_([t.upper() for t in tickers]))
    else:
        q = q.limit(100)

    stocks_result = await db.execute(q)
    stocks = stocks_result.scalars().all()

    rows = []
    for stock in stocks:
        pr = await db.execute(
            select(EODPrice)
            .where(EODPrice.stock_id == stock.id)
            .order_by(desc(EODPrice.date))
            .limit(lookback_days)
        )
        prices = pr.scalars().all()
        for p in prices:
            rows.append({"ticker": stock.ticker, "date": p.date.date(), "close": float(p.close)})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).pivot(index="date", columns="ticker", values="close")
    return df.sort_index().fillna(method="ffill")


# ─── Endpoints ────────────────────────────────────────────────────────────────
@router.get("/strategies")
async def list_quant_strategies(_user=Depends(get_current_user)):
    """List all available quantitative strategies with descriptions."""
    return [
        {
            "id": "stat_arb",
            "name": "Statistical Arbitrage",
            "description": "Pairs trading using Johansen cointegration + Kalman Filter hedge ratios",
            "papers": ["Gatev et al. (2006)", "Avellaneda & Lee (2008)"],
            "horizon": "Days to weeks",
            "regime": "All regimes (best in sideways/low-vol)",
        },
        {
            "id": "factor_model",
            "name": "Multi-Factor Alpha Model",
            "description": "Fama-French 5-factor + momentum + quality + low-vol + liquidity",
            "papers": ["Fama & French (1993/2015)", "Asness et al. (2013)", "Frazzini & Pedersen (2014)"],
            "horizon": "Monthly rebalance",
            "regime": "Bull markets",
        },
        {
            "id": "momentum_regime",
            "name": "Regime-Adaptive Momentum",
            "description": "HMM regime detection + cross-sectional and time-series momentum",
            "papers": ["Jegadeesh & Titman (1993)", "Moskowitz et al. (2012)", "Daniel & Moskowitz (2016)"],
            "horizon": "Monthly rebalance",
            "regime": "Bull (off in Bear)",
        },
        {
            "id": "rl_agent",
            "name": "Deep RL Portfolio Agent",
            "description": "PPO actor-critic with Dirichlet policy, Differential Sharpe reward",
            "papers": ["MacroHFT (KDD 2024)", "DeepScalper (CIKM 2022)", "FinRL (2020)"],
            "horizon": "Daily",
            "regime": "All regimes (learned)",
        },
        {
            "id": "black_litterman",
            "name": "Black-Litterman Portfolio",
            "description": "CAPM equilibrium blended with ML-derived investor views",
            "papers": ["Black & Litterman (1990)", "Roncalli (2013)"],
            "horizon": "Monthly rebalance",
            "regime": "All regimes",
        },
    ]


@router.post("/stat-arb/scan")
async def scan_pairs(
    request: StatArbScanRequest,
    db: AsyncSession = Depends(get_db),
    _user=Depends(get_current_user),
):
    """Scan the universe for cointegrated pairs."""
    from quant.strategies.stat_arb import PairsFinder

    prices = await _load_prices(db, request.tickers, request.lookback_days)
    if prices.empty or len(prices.columns) < 2:
        raise HTTPException(status_code=422, detail="Insufficient price data")

    finder = PairsFinder(
        min_corr=request.min_corr,
        max_half_life_days=request.max_half_life_days,
        lookback_days=request.lookback_days,
    )

    loop = asyncio.get_event_loop()
    pairs = await loop.run_in_executor(None, finder.find_pairs, prices)

    return {
        "n_pairs_found": len(pairs),
        "pairs": [
            {
                "ticker_a": p.ticker_a,
                "ticker_b": p.ticker_b,
                "hedge_ratio": round(p.hedge_ratio, 4),
                "half_life_days": round(p.half_life_days, 1),
                "spread_std": round(p.spread_std, 6),
            }
            for p in pairs[:50]  # cap response at 50 pairs
        ],
    }


@router.get("/stat-arb/signals", response_model=List[StatArbSignalOut])
async def get_stat_arb_signals(
    limit: int = Query(20, le=100),
    db: AsyncSession = Depends(get_db),
    _user=Depends(get_current_user),
):
    """Get current spread z-scores for tracked pairs."""
    from quant.strategies.stat_arb import PairsFinder, StatArbStrategy

    prices = await _load_prices(db, None, lookback_days=126)
    if prices.empty:
        return []

    finder = PairsFinder(lookback_days=min(126, len(prices)))
    loop = asyncio.get_event_loop()
    pairs = await loop.run_in_executor(None, finder.find_pairs, prices.tail(126))

    strategy = StatArbStrategy(long_only=True)
    signals = strategy.generate_signals(prices, pairs[:limit])

    return [
        StatArbSignalOut(
            ticker_a=s.ticker_a,
            ticker_b=s.ticker_b,
            z_score=s.z_score,
            spread=s.spread,
            hedge_ratio=s.hedge_ratio,
            signal=s.signal,
            half_life=round(s.half_life, 1),
            timestamp=s.timestamp,
        )
        for s in signals
    ]


@router.get("/factor-model/rankings", response_model=List[FactorScoreOut])
async def get_factor_rankings(
    top_n: int = Query(50, le=200),
    db: AsyncSession = Depends(get_db),
    _user=Depends(get_current_user),
):
    """Compute and return factor model rankings for all active stocks."""
    from quant.strategies.factor_model import FactorModel
    from db.models import Fundamental

    prices = await _load_prices(db, None, lookback_days=252)
    if prices.empty:
        return []

    # Load fundamentals
    fund_result = await db.execute(select(Fundamental))
    funds = fund_result.scalars().all()
    if funds:
        fund_df = pd.DataFrame([{
            "ticker": f.stock_id,   # will join to ticker below
            "pb_ratio": f.pb_ratio,
            "roe": f.roe,
            "roa": f.roa,
        } for f in funds])
    else:
        fund_df = pd.DataFrame(columns=["ticker", "pb_ratio", "roe", "roa"])

    model = FactorModel()
    loop = asyncio.get_event_loop()

    def _compute():
        return model.compute_scores(
            prices=prices,
            fundamentals=fund_df.set_index("ticker") if not fund_df.empty else pd.DataFrame(),
            volumes=pd.DataFrame(),
        )

    scores = await loop.run_in_executor(None, _compute)
    if scores.empty:
        return []

    return [
        FactorScoreOut(
            ticker=ticker,
            score=float(row.get("score", 0)),
            rank=int(row.get("rank", 0)),
            factor_mom=row.get("factor_mom"),
            factor_value=row.get("factor_value"),
            factor_quality=row.get("factor_quality"),
            factor_low_vol=row.get("factor_low_vol"),
            factor_growth=row.get("factor_growth"),
        )
        for ticker, row in scores.head(top_n).iterrows()
    ]


@router.get("/regime")
async def get_market_regime(
    db: AsyncSession = Depends(get_db),
    _user=Depends(get_current_user),
):
    """Detect current market regime using HMM on VN-Index proxy."""
    global _hmm_model
    from quant.strategies.momentum_regime import MarketRegimeDetector

    # Use equal-weighted market proxy from available stocks
    prices = await _load_prices(db, None, lookback_days=504)
    if prices.empty:
        return {"regime": "BULL", "confidence": 0.5, "note": "No data"}

    market_returns = prices.pct_change().dropna().mean(axis=1)

    loop = asyncio.get_event_loop()
    if _hmm_model is None:
        detector = MarketRegimeDetector(n_states=3)
        _hmm_model = await loop.run_in_executor(None, detector.fit, market_returns)
    
    state = _hmm_model.predict(market_returns)
    return {
        "regime": state.regime.value,
        "bull_probability": state.bull_prob,
        "bear_probability": state.bear_prob,
        "sideways_probability": state.sideways_prob,
        "momentum_scalar": state.momentum_scalar,
        "vol_30d": state.vol_30d,
        "trend_12m": state.trend_12m,
        "description": {
            "BULL": "Trending up — full momentum exposure recommended",
            "BEAR": "Trending down — reduce equity exposure, avoid momentum",
            "SIDEWAYS": "Range-bound — mean reversion strategies preferred",
        }.get(state.regime.value, ""),
    }


@router.get("/momentum/signals")
async def get_momentum_signals(
    top_n: int = Query(20, le=100),
    db: AsyncSession = Depends(get_db),
    _user=Depends(get_current_user),
):
    """Generate regime-aware momentum signals."""
    from quant.strategies.momentum_regime import RegimeAdaptiveMomentum

    prices = await _load_prices(db, None, lookback_days=312)
    if prices.empty:
        return []

    market_returns = prices.pct_change().dropna().mean(axis=1)
    strategy = RegimeAdaptiveMomentum(long_only=True, cs_long_n=top_n)
    
    loop = asyncio.get_event_loop()
    strategy.fit(market_returns)
    
    def _gen():
        signals_df, regime = strategy.generate_signals(prices, market_returns)
        return signals_df, regime

    signals_df, regime = await loop.run_in_executor(None, _gen)

    results = []
    for ticker, row in signals_df[signals_df["signal"] == 1].iterrows():
        results.append({
            "ticker": ticker,
            "signal": int(row["signal"]),
            "weight": float(row["weight"]),
            "cs_signal": int(row.get("cs_signal", 0)),
            "ts_signal": int(row.get("ts_signal", 0)),
            "regime": row.get("regime", "BULL"),
        })

    return {
        "regime": regime.regime.value if regime else "BULL",
        "momentum_scalar": regime.momentum_scalar if regime else 1.0,
        "signals": results[:top_n],
        "n_signals": len(results),
    }


@router.post("/portfolio/optimize")
async def optimize_portfolio(
    request: PortfolioOptimizeRequest,
    db: AsyncSession = Depends(get_db),
    _user=Depends(get_current_user),
):
    """Run portfolio optimisation using the selected method."""
    from quant.portfolio.optimizer import PortfolioConstructor

    prices = await _load_prices(db, request.tickers, request.lookback_days)
    if prices.empty or len(prices.columns) < 2:
        raise HTTPException(status_code=422, detail="Insufficient price data")

    returns = prices.pct_change().dropna()
    constructor = PortfolioConstructor()

    loop = asyncio.get_event_loop()
    weights = await loop.run_in_executor(
        None,
        lambda: constructor.construct(
            method=request.method,
            tickers=list(prices.columns),
            returns=returns,
            max_weight=request.max_weight,
        )
    )
    metrics = constructor.compute_portfolio_metrics(weights, returns)

    return {
        "method": request.method,
        "weights": weights,
        "metrics": metrics,
        "n_stocks": len([t for t, w in weights.items() if w > 0.001]),
        "generated_at": datetime.utcnow().isoformat(),
    }


@router.get("/risk/report", response_model=RiskReportOut)
async def get_risk_report(
    portfolio_id: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """Get current risk metrics for a portfolio."""
    from quant.risk.risk_manager import RiskManager
    from db.models import Portfolio, Position

    # Load portfolio positions
    positions_val: Dict[str, float] = {}
    portfolio_value = 1_000_000_000  # default

    if portfolio_id:
        pf_result = await db.execute(
            select(Portfolio).where(
                Portfolio.id == portfolio_id,
                Portfolio.user_id == user.id,
            )
        )
        pf = pf_result.scalar_one_or_none()
        if pf:
            portfolio_value = float(pf.initial_capital)
            pos_result = await db.execute(
                select(Position).where(
                    Position.portfolio_id == portfolio_id,
                    Position.is_open == True,
                )
            )
            positions = pos_result.scalars().all()
            for p in positions:
                # Use avg_cost * qty as position value (simplified)
                if p.avg_cost and p.quantity:
                    positions_val[str(p.stock_id)] = float(p.avg_cost) * p.quantity

    # Generate dummy returns for demo (production: load from DB)
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.0008, 0.015, 252))

    rm = RiskManager()
    report = rm.compute_risk_report(
        portfolio_returns=returns,
        portfolio_value=portfolio_value,
        positions=positions_val,
        sector_map={},
    )

    return RiskReportOut(
        var_95_1d=report.var_95_1d,
        cvar_95_1d=report.cvar_95_1d,
        current_drawdown=report.current_drawdown,
        max_drawdown=report.max_drawdown,
        annualised_vol=report.annualised_vol,
        sharpe_ratio=report.sharpe_ratio,
        sortino_ratio=report.sortino_ratio,
        breaches=report.breaches,
        action_required=report.action_required,
        timestamp=report.timestamp,
    )


@router.post("/risk/stress-test")
async def run_stress_test(
    tickers: List[str],
    weights: Dict[str, float],
    db: AsyncSession = Depends(get_db),
    _user=Depends(get_current_user),
):
    """Run historical stress scenarios on a given portfolio."""
    from quant.risk.risk_manager import RiskManager

    prices = await _load_prices(db, tickers, lookback_days=504)
    if prices.empty:
        return {}

    returns = prices.pct_change().dropna()
    rm = RiskManager()

    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        lambda: rm.stress_test(weights, returns)
    )
    return {"scenarios": results, "tickers": tickers}
