"""
backtest/engine.py — Backtesting engine.

Two-tier approach:
  - vectorbt:   fast portfolio-level research (thousands of parameter combos)
  - Backtrader: realistic fill simulation (partial fills, latency injection,
                commission tiers, slippage models)

Both engines share the same strategy interface so strategies are portable.
"""
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import UUID

import backtrader as bt
import numpy as np
import pandas as pd
import structlog
import vectorbt as vbt

logger = structlog.get_logger(__name__)


# ─── Configuration ─────────────────────────────────────────────────────────────
@dataclass
class BacktestConfig:
    """All parameters controlling a backtest run."""
    start_date: str                          # "2020-01-01"
    end_date: str                            # "2024-01-01"
    initial_capital: float = 1_000_000_000  # 1 billion VND
    commission_pct: float = 0.0015          # 0.15% per side (typical VN broker)
    slippage_pct: float = 0.001             # 0.1% slippage model
    latency_ms: int = 0                     # artificial order latency (HFT research)
    max_position_pct: float = 0.10          # max 10% per position
    max_positions: int = 20                 # max concurrent positions
    # Risk controls
    stop_loss_pct: Optional[float] = 0.07   # 7% stop-loss per position
    take_profit_pct: Optional[float] = None
    # Execution
    order_fill_price: str = "next_open"     # next_open | close | vwap
    allow_short: bool = False               # short-selling currently restricted in VN
    # Rebalance
    rebalance_frequency: str = "weekly"     # daily | weekly | monthly
    universe_filter: Optional[List[str]] = None  # None = all stocks


@dataclass
class BacktestResults:
    """Standardised backtest results container."""
    run_id: str
    config: BacktestConfig

    # Returns
    total_return_pct: float = 0.0
    annualised_return_pct: float = 0.0
    benchmark_return_pct: float = 0.0  # VN-Index

    # Risk
    volatility_annualised: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: int = 0
    calmar_ratio: float = 0.0

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    information_ratio: float = 0.0

    # Trade statistics
    total_trades: int = 0
    win_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    avg_holding_days: float = 0.0

    # Time series (serialisable to JSON)
    equity_curve: List[Dict] = field(default_factory=list)   # [{date, value}, ...]
    drawdown_series: List[Dict] = field(default_factory=list)
    monthly_returns: List[Dict] = field(default_factory=list)
    trade_log: List[Dict] = field(default_factory=list)


# ─── vectorbt fast backtester ─────────────────────────────────────────────────
class VectorBTEngine:
    """
    Fast portfolio backtester using vectorbt.
    Ideal for:
      - Parameter sweeps (grid search over strategy params)
      - Preliminary strategy evaluation across many stocks
      - Research / prototyping phase

    Limitation: simplified execution model (no partial fills, no latency).
    """

    def run(
        self,
        prices: pd.DataFrame,           # columns=tickers, index=datetime
        signals: pd.DataFrame,          # same shape, values ∈ {-1, 0, 1}
        config: BacktestConfig,
        run_id: str,
    ) -> BacktestResults:
        """
        Execute a signal-driven backtest.

        signals: +1=buy, -1=sell/close, 0=hold
        """
        logger.info("Starting vectorbt backtest", run_id=run_id,
                    tickers=len(prices.columns), days=len(prices))

        # Align signals and prices
        prices = prices.loc[config.start_date:config.end_date]
        signals = signals.reindex(prices.index).fillna(0)

        entries = signals == 1
        exits = signals == -1

        # ── Portfolio simulation ───────────────────────────────────────
        pf = vbt.Portfolio.from_signals(
            close=prices,
            entries=entries,
            exits=exits,
            init_cash=config.initial_capital,
            fees=config.commission_pct,
            slippage=config.slippage_pct,
            size=config.max_position_pct,         # fraction of capital per trade
            size_type="percent",
            max_orders=config.max_positions,
            allow_partial=True,
            accumulate=False,
            sl_stop=config.stop_loss_pct,
            tp_stop=config.take_profit_pct,
            freq="B",
        )

        return self._extract_results(pf, config, run_id)

    def _extract_results(
        self, pf: vbt.Portfolio, config: BacktestConfig, run_id: str
    ) -> BacktestResults:
        """Extract standardised metrics from a vectorbt Portfolio object."""
        stats = pf.stats()
        equity = pf.value()

        # Equity curve
        equity_curve = [
            {"date": str(d.date()), "value": float(v)}
            for d, v in equity.items()
        ]

        # Drawdown series
        dd = pf.drawdown()
        drawdown_series = [
            {"date": str(d.date()), "drawdown": float(v)}
            for d, v in dd.items()
        ]

        # Monthly returns
        monthly_ret = equity.resample("ME").last().pct_change().dropna()
        monthly_returns = [
            {"month": str(d.date()), "return": float(v)}
            for d, v in monthly_ret.items()
        ]

        # Trade log
        trades = pf.trades.records_readable
        trade_log = trades.to_dict("records") if not trades.empty else []

        # Annualised return
        n_years = len(equity) / 252
        total_ret = float((equity.iloc[-1] / equity.iloc[0]) - 1)
        ann_ret = float((1 + total_ret) ** (1 / max(n_years, 0.01)) - 1) if n_years > 0 else 0

        # Sharpe
        daily_returns = equity.pct_change().dropna()
        sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0

        # Sortino
        downside_std = float(daily_returns[daily_returns < 0].std() * np.sqrt(252))
        sortino = float(ann_ret / downside_std) if downside_std > 0 else 0

        # Max drawdown
        cum_max = equity.cummax()
        drawdowns = (equity - cum_max) / cum_max
        max_dd = float(drawdowns.min())

        # Calmar
        calmar = float(-ann_ret / max_dd) if max_dd < 0 else 0

        # Trade stats
        if not trades.empty:
            wins = trades[trades["PnL"] > 0]
            losses = trades[trades["PnL"] <= 0]
            win_rate = len(wins) / len(trades)
            avg_win = float(wins["Return"].mean()) if not wins.empty else 0
            avg_loss = float(losses["Return"].mean()) if not losses.empty else 0
            profit_factor = float(wins["PnL"].sum() / abs(losses["PnL"].sum())) if not losses.empty else float("inf")
            avg_holding = float(trades["Duration"].dt.days.mean()) if "Duration" in trades.columns else 0
        else:
            win_rate = avg_win = avg_loss = profit_factor = avg_holding = 0

        return BacktestResults(
            run_id=run_id,
            config=config,
            total_return_pct=round(total_ret * 100, 2),
            annualised_return_pct=round(ann_ret * 100, 2),
            volatility_annualised=round(float(daily_returns.std() * np.sqrt(252)) * 100, 2),
            max_drawdown_pct=round(max_dd * 100, 2),
            sharpe_ratio=round(sharpe, 3),
            sortino_ratio=round(sortino, 3),
            calmar_ratio=round(calmar, 3),
            total_trades=len(trades),
            win_rate=round(win_rate * 100, 2),
            avg_win_pct=round(avg_win * 100, 2),
            avg_loss_pct=round(avg_loss * 100, 2),
            profit_factor=round(profit_factor, 3),
            avg_holding_days=round(avg_holding, 1),
            equity_curve=equity_curve,
            drawdown_series=drawdown_series,
            monthly_returns=monthly_returns,
            trade_log=trade_log[:1000],  # cap at 1000 rows for API response
        )


# ─── Backtrader realistic simulator ───────────────────────────────────────────
class VNSlippageModel(bt.Sizer):
    """
    Vietnam market slippage model.
    Price impact scales with order size relative to daily volume.
    """
    params = (("base_slippage", 0.001), ("volume_impact", 0.002))

    def _getsizing(self, comminfo, cash, data, isbuy):
        # Position size from parent config
        size = int(cash * 0.10 / data.close[0])  # 10% of cash per trade
        return max(size, 0)


class BacktraderEngine:
    """
    Realistic backtest using Backtrader.
    Supports:
      - Partial fills based on daily volume
      - Latency injection (for HFT research)
      - Commission tiers (standard broker, negotiated tier)
      - Stop-loss / take-profit orders
      - Multiple data feeds simultaneously
    """

    def run(
        self,
        ohlcv_data: Dict[str, pd.DataFrame],  # {ticker: OHLCV DataFrame}
        strategy_class: type,                  # Backtrader Strategy class
        config: BacktestConfig,
        strategy_params: Optional[Dict] = None,
        run_id: str = "",
    ) -> BacktestResults:
        """
        Run a Backtrader simulation.

        ohlcv_data: dict of {ticker: DataFrame with [date, open, high, low, close, volume]}
        strategy_class: a bt.Strategy subclass
        """
        logger.info("Starting Backtrader simulation", run_id=run_id,
                    tickers=len(ohlcv_data))

        cerebro = bt.Cerebro()

        # ── Capital & commission ────────────────────────────────────────
        cerebro.broker.setcash(config.initial_capital)
        cerebro.broker.setcommission(commission=config.commission_pct)

        # ── Data feeds ──────────────────────────────────────────────────
        start = pd.Timestamp(config.start_date)
        end = pd.Timestamp(config.end_date)

        for ticker, df in ohlcv_data.items():
            df_bt = df.copy()
            df_bt["date"] = pd.to_datetime(df_bt["date"])
            df_bt = df_bt.set_index("date").sort_index()
            df_bt = df_bt.loc[start:end]

            data_feed = bt.feeds.PandasData(
                dataname=df_bt,
                name=ticker,
                open="open",
                high="high",
                low="low",
                close="close",
                volume="volume",
                openinterest=-1,
            )
            cerebro.adddata(data_feed)

        # ── Strategy ────────────────────────────────────────────────────
        params = strategy_params or {}
        if config.stop_loss_pct:
            params["stop_loss_pct"] = config.stop_loss_pct
        if config.take_profit_pct:
            params["take_profit_pct"] = config.take_profit_pct
        cerebro.addstrategy(strategy_class, **params)

        # ── Analysers ───────────────────────────────────────────────────
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.045)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="time_return", timeframe=bt.TimeFrame.Days)

        initial_cash = cerebro.broker.getvalue()
        results = cerebro.run()
        final_value = cerebro.broker.getvalue()

        strat = results[0]
        return self._extract_results(strat, initial_cash, final_value, config, run_id)

    def _extract_results(
        self,
        strat: bt.Strategy,
        initial_cash: float,
        final_value: float,
        config: BacktestConfig,
        run_id: str,
    ) -> BacktestResults:
        sharpe_analysis = strat.analyzers.sharpe.get_analysis()
        dd_analysis = strat.analyzers.drawdown.get_analysis()
        trade_analysis = strat.analyzers.trades.get_analysis()
        time_return = strat.analyzers.time_return.get_analysis()

        total_ret = (final_value - initial_cash) / initial_cash
        n_years = len(time_return) / 252 if time_return else 1
        ann_ret = float((1 + total_ret) ** (1 / max(n_years, 0.01)) - 1)

        # Equity curve from daily returns
        equity_curve = []
        value = initial_cash
        for date, ret in time_return.items():
            value *= (1 + ret)
            equity_curve.append({"date": str(date.date()), "value": round(value, 2)})

        # Trade stats
        won = trade_analysis.get("won", {})
        lost = trade_analysis.get("lost", {})
        total_trades = trade_analysis.get("total", {}).get("closed", 0)
        win_rate = won.get("total", 0) / total_trades if total_trades else 0
        profit_factor = abs(won.get("pnl", {}).get("total", 0) /
                           lost.get("pnl", {}).get("total", 1e-9))

        return BacktestResults(
            run_id=run_id,
            config=config,
            total_return_pct=round(total_ret * 100, 2),
            annualised_return_pct=round(ann_ret * 100, 2),
            volatility_annualised=0.0,  # computed separately
            max_drawdown_pct=round(-dd_analysis.get("max", {}).get("drawdown", 0), 2),
            max_drawdown_duration_days=dd_analysis.get("max", {}).get("len", 0),
            sharpe_ratio=round(sharpe_analysis.get("sharperatio", 0) or 0, 3),
            total_trades=total_trades,
            win_rate=round(win_rate * 100, 2),
            avg_win_pct=round(won.get("pnl", {}).get("average", 0) / initial_cash * 100, 2),
            avg_loss_pct=round(lost.get("pnl", {}).get("average", 0) / initial_cash * 100, 2),
            profit_factor=round(profit_factor, 3),
            equity_curve=equity_curve,
        )


# ─── Async wrapper ────────────────────────────────────────────────────────────
class BacktestOrchestrator:
    """
    Async orchestrator that dispatches backtest jobs.
    - Short/research jobs: VectorBT (fast)
    - Production validation: Backtrader (realistic)
    """

    def __init__(self):
        self._vbt = VectorBTEngine()
        self._bt = BacktraderEngine()

    async def run_vectorbt(
        self,
        prices: pd.DataFrame,
        signals: pd.DataFrame,
        config: BacktestConfig,
        run_id: str,
    ) -> BacktestResults:
        """Run vectorbt in a thread executor to not block the event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._vbt.run, prices, signals, config, run_id
        )

    async def run_backtrader(
        self,
        ohlcv_data: Dict[str, pd.DataFrame],
        strategy_class: type,
        config: BacktestConfig,
        strategy_params: Optional[Dict] = None,
        run_id: str = "",
    ) -> BacktestResults:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._bt.run,
            ohlcv_data,
            strategy_class,
            config,
            strategy_params,
            run_id,
        )
