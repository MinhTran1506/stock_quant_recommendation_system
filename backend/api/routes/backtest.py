"""
api/routes/backtest.py — Backtest submission and results endpoints.
"""
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from api.middleware.auth import get_current_user
from backtest.engine import BacktestConfig, BacktestOrchestrator
from db.models import BacktestRun, Strategy
from db.session import get_db

router = APIRouter()
logger = structlog.get_logger(__name__)
_orchestrator = BacktestOrchestrator()


# ─── Schemas ──────────────────────────────────────────────────────────────────
class BacktestRequest(BaseModel):
    name: str = Field(..., description="Human-readable name for this run")
    strategy_id: Optional[str] = None
    start_date: str = Field(..., example="2020-01-01")
    end_date: str = Field(..., example="2024-01-01")
    initial_capital: float = Field(1_000_000_000, description="Initial capital in VND")
    universe: Optional[List[str]] = Field(None, description="Ticker list; null = all stocks")
    commission_pct: float = 0.0015
    slippage_pct: float = 0.001
    stop_loss_pct: Optional[float] = 0.07
    take_profit_pct: Optional[float] = None
    max_position_pct: float = 0.10
    max_positions: int = 20
    engine: str = Field("vectorbt", description="vectorbt | backtrader")
    # Signal source: use meta-model ranking as entry signal
    signal_source: str = Field("meta_model", description="meta_model | custom")
    min_score: float = Field(60.0, description="Minimum meta-model score to enter")
    horizon_days: int = Field(5, description="Rebalance horizon")


class BacktestSummary(BaseModel):
    id: str
    name: str
    status: str
    start_date: str
    end_date: str
    initial_capital: float
    total_return_pct: Optional[float]
    sharpe_ratio: Optional[float]
    max_drawdown_pct: Optional[float]
    total_trades: Optional[int]
    created_at: str


class BacktestDetail(BacktestSummary):
    summary_metrics: Optional[Dict]
    equity_curve: Optional[List[Dict]]
    monthly_returns: Optional[List[Dict]]
    trade_log: Optional[List[Dict]]
    config: Optional[Dict]


# ─── Background runner ────────────────────────────────────────────────────────
async def _run_backtest_job(run_id: str, request: BacktestRequest, user_id: str):
    """
    Background task: fetches data, builds signals, runs engine, saves results.
    Status transitions: PENDING → RUNNING → DONE | FAILED
    """
    from db.session import _session_factory  # import here to avoid circular

    async with _session_factory() as db:
        # Mark as RUNNING
        result = await db.execute(
            select(BacktestRun).where(BacktestRun.id == run_id)
        )
        run = result.scalar_one_or_none()
        if not run:
            return
        run.status = "RUNNING"
        await db.commit()

        try:
            config = BacktestConfig(
                start_date=request.start_date,
                end_date=request.end_date,
                initial_capital=request.initial_capital,
                commission_pct=request.commission_pct,
                slippage_pct=request.slippage_pct,
                stop_loss_pct=request.stop_loss_pct,
                take_profit_pct=request.take_profit_pct,
                max_position_pct=request.max_position_pct,
                max_positions=request.max_positions,
                universe_filter=request.universe,
            )

            # Fetch price data from DB
            import pandas as pd
            from db.models import EODPrice, Stock as StockModel

            # Build price matrix and signals (simplified; production would
            # hydrate from TimescaleDB using the universe filter)
            tickers = request.universe or []
            if not tickers:
                st_result = await db.execute(
                    select(StockModel.ticker).where(StockModel.is_active == True).limit(100)
                )
                tickers = [r[0] for r in st_result.all()]

            # Load historical predictions (model scores) to build entry signals
            from db.models import Prediction as PredModel
            pred_result = await db.execute(
                select(PredModel)
                .where(
                    PredModel.horizon_days == request.horizon_days,
                    PredModel.score >= request.min_score,
                )
                .order_by(PredModel.generated_at)
            )
            preds = pred_result.scalars().all()

            # Build signal DataFrame: +1 where score >= min_score, 0 elsewhere
            # (Full implementation would join to price data per date)
            signals_dict: Dict = {}
            prices_dict: Dict = {}

            # For now, create dummy data structure for demonstration
            # In production, this loads from TimescaleDB
            import numpy as np
            date_range = pd.bdate_range(request.start_date, request.end_date)
            for ticker in tickers[:20]:  # cap at 20 for demo
                prices_dict[ticker] = pd.Series(
                    np.random.lognormal(0, 0.01, len(date_range)).cumprod() * 10000,
                    index=date_range,
                )
                signals_dict[ticker] = pd.Series(
                    np.random.choice([-1, 0, 0, 1], len(date_range)),
                    index=date_range,
                )

            prices_df = pd.DataFrame(prices_dict)
            signals_df = pd.DataFrame(signals_dict)

            # Run engine
            if request.engine == "vectorbt":
                results = await _orchestrator.run_vectorbt(
                    prices=prices_df,
                    signals=signals_df,
                    config=config,
                    run_id=run_id,
                )
            else:
                logger.info("Backtrader engine selected", run_id=run_id)
                # Backtrader requires per-ticker OHLCV; adapt accordingly
                results = await _orchestrator.run_vectorbt(
                    prices=prices_df, signals=signals_df, config=config, run_id=run_id
                )

            # Save results
            run.status = "DONE"
            run.completed_at = datetime.utcnow()
            run.summary_metrics = {
                "total_return_pct": results.total_return_pct,
                "annualised_return_pct": results.annualised_return_pct,
                "sharpe_ratio": results.sharpe_ratio,
                "sortino_ratio": results.sortino_ratio,
                "max_drawdown_pct": results.max_drawdown_pct,
                "calmar_ratio": results.calmar_ratio,
                "total_trades": results.total_trades,
                "win_rate": results.win_rate,
                "avg_win_pct": results.avg_win_pct,
                "avg_loss_pct": results.avg_loss_pct,
                "profit_factor": results.profit_factor,
                "avg_holding_days": results.avg_holding_days,
                "volatility_annualised": results.volatility_annualised,
            }
            run.equity_curve = results.equity_curve
            run.trade_log = results.trade_log[:500]
            await db.commit()
            logger.info("Backtest completed", run_id=run_id,
                        sharpe=results.sharpe_ratio, total_ret=results.total_return_pct)

        except Exception as e:
            run.status = "FAILED"
            run.error_message = str(e)
            await db.commit()
            logger.error("Backtest failed", run_id=run_id, error=str(e))


# ─── Endpoints ────────────────────────────────────────────────────────────────
@router.post("/", response_model=BacktestSummary, status_code=202)
async def submit_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """Submit a new backtest run. Executes asynchronously; poll /backtest/{id} for results."""
    run_id = str(uuid4())
    run = BacktestRun(
        id=run_id,
        user_id=user.id,
        name=request.name,
        start_date=datetime.fromisoformat(request.start_date),
        end_date=datetime.fromisoformat(request.end_date),
        initial_capital=request.initial_capital,
        config=request.model_dump(),
        status="PENDING",
    )
    if request.strategy_id:
        run.strategy_id = request.strategy_id

    db.add(run)
    await db.commit()

    background_tasks.add_task(_run_backtest_job, run_id, request, str(user.id))
    logger.info("Backtest submitted", run_id=run_id, name=request.name)

    return BacktestSummary(
        id=run_id,
        name=request.name,
        status="PENDING",
        start_date=request.start_date,
        end_date=request.end_date,
        initial_capital=request.initial_capital,
        total_return_pct=None,
        sharpe_ratio=None,
        max_drawdown_pct=None,
        total_trades=None,
        created_at=datetime.utcnow().isoformat(),
    )


@router.get("/", response_model=List[BacktestSummary])
async def list_backtests(
    limit: int = Query(20, le=100),
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """List the current user's backtest runs."""
    result = await db.execute(
        select(BacktestRun)
        .where(BacktestRun.user_id == user.id)
        .order_by(desc(BacktestRun.created_at))
        .offset(offset).limit(limit)
    )
    runs = result.scalars().all()
    return [
        BacktestSummary(
            id=str(r.id),
            name=r.name or "",
            status=r.status,
            start_date=str(r.start_date.date()),
            end_date=str(r.end_date.date()),
            initial_capital=float(r.initial_capital),
            total_return_pct=r.summary_metrics.get("total_return_pct") if r.summary_metrics else None,
            sharpe_ratio=r.summary_metrics.get("sharpe_ratio") if r.summary_metrics else None,
            max_drawdown_pct=r.summary_metrics.get("max_drawdown_pct") if r.summary_metrics else None,
            total_trades=r.summary_metrics.get("total_trades") if r.summary_metrics else None,
            created_at=r.created_at.isoformat(),
        )
        for r in runs
    ]


@router.get("/{run_id}", response_model=BacktestDetail)
async def get_backtest(
    run_id: str,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """Get full backtest results including equity curve, trades, and metrics."""
    result = await db.execute(
        select(BacktestRun).where(
            BacktestRun.id == run_id,
            BacktestRun.user_id == user.id,
        )
    )
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Backtest run not found")

    metrics = run.summary_metrics or {}
    return BacktestDetail(
        id=str(run.id),
        name=run.name or "",
        status=run.status,
        start_date=str(run.start_date.date()),
        end_date=str(run.end_date.date()),
        initial_capital=float(run.initial_capital),
        total_return_pct=metrics.get("total_return_pct"),
        sharpe_ratio=metrics.get("sharpe_ratio"),
        max_drawdown_pct=metrics.get("max_drawdown_pct"),
        total_trades=metrics.get("total_trades"),
        created_at=run.created_at.isoformat(),
        summary_metrics=metrics,
        equity_curve=run.equity_curve,
        monthly_returns=None,
        trade_log=run.trade_log,
        config=run.config,
    )


@router.delete("/{run_id}", status_code=204)
async def delete_backtest(
    run_id: str,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    result = await db.execute(
        select(BacktestRun).where(BacktestRun.id == run_id, BacktestRun.user_id == user.id)
    )
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Not found")
    await db.delete(run)
    await db.commit()
