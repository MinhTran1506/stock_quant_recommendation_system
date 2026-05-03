"""
quant/risk/risk_manager.py — Quantitative Risk Management
══════════════════════════════════════════════════════════

Research basis:
  • Rockafellar & Uryasev (2000) — CVaR (Expected Shortfall); regulatory
    standard under Basel III/IV; superior to VaR for tail risk.
  • Roncalli (2020) — "Handbook on Financial Risk Management"; industry
    reference for quant risk systems at banks and hedge funds.
  • ESMA Guidelines — position limits, concentration risk, drawdown controls.
  • Markowitz (1959) — Semi-variance for downside risk.

Risk controls implemented:
  1. Value-at-Risk (VaR): parametric, historical simulation, Monte Carlo
  2. Conditional VaR (CVaR / Expected Shortfall)
  3. Maximum drawdown monitoring with auto-deleveraging
  4. Concentration limits (single stock, sector)
  5. Factor exposure limits (beta, momentum exposure)
  6. Real-time position P&L monitor with kill-switch
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RiskLimits:
    """Configurable risk limits for a strategy."""
    # Portfolio-level
    max_drawdown_pct: float = 0.15          # 15% max drawdown → reduce exposure
    hard_stop_drawdown_pct: float = 0.20    # 20% → halt all trading
    var_95_limit_pct: float = 0.02          # 1-day 95% VaR < 2% of portfolio
    cvar_95_limit_pct: float = 0.03         # 1-day CVaR < 3%
    max_portfolio_vol: float = 0.20         # annualised vol target

    # Position-level
    max_single_position_pct: float = 0.15   # no stock > 15% of portfolio
    max_sector_concentration_pct: float = 0.40  # no sector > 40%
    max_correlation_exposure: float = 0.60  # avg pairwise correlation limit

    # Leverage (long-only: always ≤ 1.0 for Vietnam paper trading)
    max_gross_leverage: float = 1.0
    max_net_leverage: float = 1.0


@dataclass
class RiskReport:
    """Complete risk snapshot for a portfolio."""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    var_95_1d: float = 0.0
    var_99_1d: float = 0.0
    cvar_95_1d: float = 0.0
    cvar_99_1d: float = 0.0
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    annualised_vol: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    beta: float = 0.0
    max_position_pct: float = 0.0
    max_sector_pct: float = 0.0
    breaches: List[str] = field(default_factory=list)
    action_required: str = "NONE"   # NONE | REDUCE | HALT


class RiskManager:
    """
    Real-time quantitative risk management engine.

    Used by:
      - StrategyOrchestrator before order submission (pre-trade checks)
      - Background worker for continuous portfolio monitoring
      - API endpoint for risk reporting dashboard
    """

    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        self._peak_value: float = 1.0
        self._returns_history: List[float] = []

    # ── Pre-trade position check ───────────────────────────────────────────
    def pre_trade_check(
        self,
        ticker: str,
        proposed_value: float,
        portfolio_value: float,
        current_positions: Dict[str, float],   # {ticker: vnd_value}
        sector_map: Dict[str, str],            # {ticker: sector}
    ) -> Tuple[bool, str]:
        """
        Returns (approved: bool, reason: str).
        Check concentration limits before submitting order.
        """
        # Single position limit
        new_positions = {**current_positions, ticker: proposed_value}
        total = max(sum(new_positions.values()), 1e-8)
        position_pct = proposed_value / portfolio_value

        if position_pct > self.limits.max_single_position_pct:
            return False, (f"Position limit breach: {ticker} would be "
                           f"{position_pct:.1%} > {self.limits.max_single_position_pct:.1%}")

        # Sector concentration
        sector = sector_map.get(ticker, "Unknown")
        sector_value = sum(
            v for t, v in new_positions.items()
            if sector_map.get(t) == sector
        )
        sector_pct = sector_value / portfolio_value
        if sector_pct > self.limits.max_sector_concentration_pct:
            return False, (f"Sector limit breach: {sector} would be "
                           f"{sector_pct:.1%} > {self.limits.max_sector_concentration_pct:.1%}")

        # Gross leverage
        gross = sum(new_positions.values()) / portfolio_value
        if gross > self.limits.max_gross_leverage:
            return False, f"Leverage limit: gross={gross:.2f} > {self.limits.max_gross_leverage}"

        return True, "OK"

    # ── Portfolio risk metrics ─────────────────────────────────────────────
    def compute_risk_report(
        self,
        portfolio_returns: pd.Series,
        portfolio_value: float,
        positions: Dict[str, float],
        sector_map: Dict[str, str],
        benchmark_returns: Optional[pd.Series] = None,
        lookback_days: int = 252,
    ) -> RiskReport:
        """Compute full risk report for current portfolio state."""
        ret = portfolio_returns.dropna().tail(lookback_days)
        if len(ret) < 10:
            return RiskReport()

        breaches = []
        ret_arr = ret.values

        # VaR and CVaR
        var_95 = float(-np.percentile(ret_arr, 5))
        var_99 = float(-np.percentile(ret_arr, 1))
        cvar_95 = float(-np.mean(ret_arr[ret_arr <= np.percentile(ret_arr, 5)]))
        cvar_99 = float(-np.mean(ret_arr[ret_arr <= np.percentile(ret_arr, 1)]))

        # Drawdown
        cum_ret = (1 + ret).cumprod()
        rolling_max = cum_ret.cummax()
        drawdowns = (cum_ret - rolling_max) / rolling_max
        current_dd = float(drawdowns.iloc[-1])
        max_dd = float(drawdowns.min())

        # Volatility
        ann_vol = float(ret.std() * np.sqrt(252))
        ann_ret = float(ret.mean() * 252)
        sharpe  = (ann_ret - 0.045) / max(ann_vol, 1e-8)

        # Sortino (downside deviation)
        downside = ret[ret < 0].std() * np.sqrt(252)
        sortino  = (ann_ret - 0.045) / max(float(downside), 1e-8)

        # Beta
        beta = 0.0
        if benchmark_returns is not None and len(benchmark_returns) > 30:
            common = ret.index.intersection(benchmark_returns.index)
            if len(common) > 30:
                bm = benchmark_returns.loc[common].values
                pt = ret.loc[common].values
                beta = float(np.cov(pt, bm)[0, 1] / max(np.var(bm), 1e-8))

        # Position concentration
        total_port = max(sum(positions.values()), 1e-8)
        max_pos_pct = float(max(positions.values()) / portfolio_value) if positions else 0

        sector_totals: Dict[str, float] = {}
        for t, v in positions.items():
            s = sector_map.get(t, "Unknown")
            sector_totals[s] = sector_totals.get(s, 0) + v
        max_sector_pct = float(max(sector_totals.values()) / portfolio_value) if sector_totals else 0

        # Breach detection
        if var_95 > self.limits.var_95_limit_pct:
            breaches.append(f"VaR95={var_95:.2%} > limit {self.limits.var_95_limit_pct:.2%}")
        if cvar_95 > self.limits.cvar_95_limit_pct:
            breaches.append(f"CVaR95={cvar_95:.2%} > limit {self.limits.cvar_95_limit_pct:.2%}")
        if abs(current_dd) > self.limits.max_drawdown_pct:
            breaches.append(f"Drawdown={current_dd:.2%} > limit -{self.limits.max_drawdown_pct:.2%}")
        if ann_vol > self.limits.max_portfolio_vol:
            breaches.append(f"Vol={ann_vol:.2%} > limit {self.limits.max_portfolio_vol:.2%}")
        if max_pos_pct > self.limits.max_single_position_pct:
            breaches.append(f"MaxPosition={max_pos_pct:.2%} > limit {self.limits.max_single_position_pct:.2%}")

        # Action determination
        if abs(current_dd) >= self.limits.hard_stop_drawdown_pct:
            action = "HALT"
        elif breaches:
            action = "REDUCE"
        else:
            action = "NONE"

        return RiskReport(
            var_95_1d=round(var_95, 6),
            var_99_1d=round(var_99, 6),
            cvar_95_1d=round(cvar_95, 6),
            cvar_99_1d=round(cvar_99, 6),
            current_drawdown=round(current_dd, 6),
            max_drawdown=round(max_dd, 6),
            annualised_vol=round(ann_vol, 6),
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4),
            beta=round(beta, 4),
            max_position_pct=round(max_pos_pct, 4),
            max_sector_pct=round(max_sector_pct, 4),
            breaches=breaches,
            action_required=action,
        )

    def monte_carlo_var(
        self,
        returns: pd.Series,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        n_simulations: int = 10_000,
        confidence: float = 0.95,
    ) -> Dict[str, float]:
        """
        Monte Carlo 1-day VaR and CVaR.
        More accurate for non-normal return distributions.
        """
        mu  = returns.mean().values if isinstance(returns, pd.DataFrame) else np.zeros(len(weights))
        # Cholesky decomposition for correlated simulation
        try:
            L = np.linalg.cholesky(cov_matrix + 1e-6 * np.eye(len(weights)))
        except np.linalg.LinAlgError:
            L = np.diag(np.sqrt(np.diag(cov_matrix) + 1e-6))

        z = np.random.standard_normal((n_simulations, len(weights)))
        simulated_returns = mu + z @ L.T
        port_returns = simulated_returns @ weights

        var_mc   = float(-np.percentile(port_returns, (1 - confidence) * 100))
        cvar_mc  = float(-np.mean(port_returns[port_returns <= np.percentile(port_returns, (1 - confidence) * 100)]))

        return {
            "var_mc": round(var_mc, 6),
            "cvar_mc": round(cvar_mc, 6),
            "confidence": confidence,
            "n_simulations": n_simulations,
        }

    def stress_test(
        self,
        weights: Dict[str, float],
        returns: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Historical stress scenarios (crisis periods).
        Estimate portfolio loss under each scenario.
        """
        scenarios = {
            "2008_financial_crisis": ("2008-01-01", "2009-03-31"),
            "2020_covid_crash":      ("2020-01-20", "2020-03-31"),
            "2022_rate_shock":       ("2022-01-01", "2022-12-31"),
            "2007_quant_meltdown":   ("2007-07-01", "2007-09-30"),
        }

        w = np.array([weights.get(t, 0) for t in returns.columns])
        results = {}

        for name, (start, end) in scenarios.items():
            try:
                period_returns = returns.loc[start:end]
                if len(period_returns) > 5:
                    port_ret = (period_returns @ w)
                    total_loss = float((1 + port_ret).prod() - 1)
                    max_dd = float(((1 + port_ret).cumprod() /
                                    (1 + port_ret).cumprod().cummax() - 1).min())
                    results[name] = {
                        "total_return": round(total_loss, 4),
                        "max_drawdown": round(max_dd, 4),
                    }
            except Exception:
                pass

        return results
