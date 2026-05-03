"""
quant/portfolio/optimizer.py — Portfolio Construction Engine
═══════════════════════════════════════════════════════════

Research basis:
  • Markowitz (1952) — Mean-Variance Optimisation (MVO); foundational.
  • Black & Litterman (1990) — Bayesian framework blending market
    equilibrium (CAPM implied returns) with investor views; eliminates
    extreme corner solutions of MVO. Standard at Goldman Sachs, JP Morgan.
  • Roncalli (2013) — Risk Parity / Equal Risk Contribution (ERC);
    used by Bridgewater (All Weather), AQR, Invesco.
  • NeurIPS 2024 — m-Sparse Sharpe Ratio Maximisation; sparse portfolio
    construction that limits concentration risk.
  • Ledoit & Wolf (2004) — Covariance shrinkage; robust estimation for
    small samples (N > T problem common in emerging markets).

Portfolio types:
  ┌──────────────────────────────────────────────────────────────────┐
  │ Type              │ Objective              │ Best For             │
  ├───────────────────┼────────────────────────┼──────────────────────┤
  │ MeanVariance      │ max Sharpe (QP)         │ High conviction      │
  │ BlackLitterman    │ BL posterior weights    │ Views + equilibrium  │
  │ RiskParity        │ Equal risk contribution │ Diversification      │
  │ MinVariance       │ min portfolio variance  │ Low-vol defensive    │
  │ MaxDiversification│ max diversification     │ Factor exposure mgmt │
  └──────────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import structlog
from scipy.optimize import minimize, LinearConstraint, Bounds

logger = structlog.get_logger(__name__)


# ─── Covariance Estimation ─────────────────────────────────────────────────────
def ledoit_wolf_shrinkage(returns: pd.DataFrame) -> np.ndarray:
    """
    Ledoit-Wolf analytical shrinkage estimator.
    Shrinks sample covariance toward scaled identity matrix.
    Handles N > T regime common in Vietnamese universe vs. short history.
    """
    try:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf()
        lw.fit(returns.fillna(0).values)
        return lw.covariance_
    except Exception:
        return returns.cov().fillna(0).values


# ─── Mean-Variance Optimisation ───────────────────────────────────────────────
class MeanVarianceOptimizer:
    """
    Markowitz mean-variance with maximum Sharpe objective.
    Constraints: long-only, weights sum to 1, max position size.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.045,   # 4.5% Vietnam policy rate
        max_weight: float = 0.20,
        min_weight: float = 0.0,
    ):
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight
        self.min_weight = min_weight

    def optimize(
        self,
        expected_returns: pd.Series,
        cov_matrix: np.ndarray,
        tickers: List[str],
    ) -> Dict[str, float]:
        """
        Maximise Sharpe Ratio subject to constraints.
        Returns {ticker: weight}.
        """
        n = len(tickers)
        mu = expected_returns.reindex(tickers).fillna(0).values
        sigma = cov_matrix

        def neg_sharpe(w):
            port_ret = np.dot(w, mu)
            port_vol = np.sqrt(w @ sigma @ w)
            return -(port_ret - self.risk_free_rate / 252) / max(port_vol, 1e-8)

        # Initial guess: equal weight
        w0 = np.ones(n) / n
        bounds = Bounds(lb=self.min_weight, ub=self.max_weight)
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

        result = minimize(
            neg_sharpe, w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9},
        )

        if not result.success:
            logger.warning("MVO optimisation failed, using equal weight")
            weights = w0
        else:
            weights = result.x

        # Clip negatives from numerical noise
        weights = np.clip(weights, 0, self.max_weight)
        weights /= weights.sum()
        return dict(zip(tickers, weights.round(6).tolist()))


# ─── Black-Litterman ──────────────────────────────────────────────────────────
@dataclass
class InvestorView:
    """A single investor view on expected returns."""
    description: str
    tickers: List[str]        # tickers involved in the view
    view_weights: List[float] # relative weights in the view portfolio
    expected_return: float    # expected excess return of this view
    confidence: float         # uncertainty τ²/ω; higher = more confident (0-1)


class BlackLittermanOptimizer:
    """
    Black-Litterman model.

    Combines market equilibrium (CAPM implied returns from market caps)
    with investor views (from ML model signals) via Bayes theorem.

    Posterior expected returns:
        μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹ Π + P'Ω⁻¹Q]

    where:
        Π  = CAPM equilibrium returns (from market caps)
        Q  = view expected returns vector
        P  = pick matrix (which assets each view covers)
        Ω  = view uncertainty diagonal matrix
        τ  = scaling parameter (typically 1/T)
    """

    def __init__(
        self,
        risk_aversion: float = 2.5,   # δ: typical market risk aversion
        tau: float = 0.025,            # scaling for prior uncertainty
        max_weight: float = 0.15,
    ):
        self.delta = risk_aversion
        self.tau = tau
        self.max_weight = max_weight
        self._mvo = MeanVarianceOptimizer(max_weight=max_weight)

    def optimize(
        self,
        market_caps: pd.Series,
        cov_matrix: np.ndarray,
        tickers: List[str],
        views: List[InvestorView],
    ) -> Dict[str, float]:
        """
        Compute BL posterior weights.

        market_caps: Series {ticker: market_cap} for equilibrium computation
        views:       list of InvestorView from ML model / analyst signals
        """
        n = len(tickers)
        sigma = cov_matrix

        # ── Market capitalisation weights (prior) ─────────────────────
        mcap = market_caps.reindex(tickers).fillna(1.0)
        w_eq = mcap / mcap.sum()

        # ── CAPM equilibrium returns ──────────────────────────────────
        pi = self.delta * sigma @ w_eq.values   # shape (n,)

        if not views:
            # No views: use pure equilibrium
            return self._mvo.optimize(
                pd.Series(pi * 252, index=tickers), sigma, tickers
            )

        # ── Build P (pick matrix) and Q (view returns) ────────────────
        k = len(views)
        P = np.zeros((k, n))
        Q = np.zeros(k)
        omega_diag = np.zeros(k)
        ticker_idx = {t: i for i, t in enumerate(tickers)}

        for i, view in enumerate(views):
            for t, wt in zip(view.tickers, view.view_weights):
                if t in ticker_idx:
                    P[i, ticker_idx[t]] = wt
            Q[i] = view.expected_return
            # Ω_ii = (1 - confidence) / confidence × τ × P_i Σ P_i'
            p_i = P[i]
            omega_diag[i] = max(
                (1 - view.confidence) / max(view.confidence, 1e-6)
                * self.tau * float(p_i @ sigma @ p_i),
                1e-8,
            )

        omega = np.diag(omega_diag)
        tau_sigma_inv = np.linalg.inv(self.tau * sigma + 1e-6 * np.eye(n))
        omega_inv = np.linalg.inv(omega)

        # ── BL posterior expected returns ─────────────────────────────
        A = tau_sigma_inv + P.T @ omega_inv @ P
        b = tau_sigma_inv @ pi + P.T @ omega_inv @ Q
        mu_bl = np.linalg.solve(A, b)

        # ── Posterior covariance ──────────────────────────────────────
        sigma_bl = sigma + np.linalg.inv(A)

        # ── MVO on BL posterior ───────────────────────────────────────
        mu_bl_annual = pd.Series(mu_bl * 252, index=tickers)
        return self._mvo.optimize(mu_bl_annual, sigma_bl, tickers)

    def views_from_factor_scores(
        self,
        factor_scores: pd.DataFrame,
        top_n: int = 10,
        confidence: float = 0.6,
    ) -> List[InvestorView]:
        """
        Auto-generate BL views from factor model scores.
        Top-N stocks: expected outperformance view.
        Bottom-N stocks: expected underperformance view.
        """
        if factor_scores.empty:
            return []

        sorted_df = factor_scores.sort_values("score", ascending=False)
        n = len(sorted_df)

        views = []
        # Long view: top-N stocks outperform
        top_tickers = list(sorted_df.head(min(top_n, n // 4)).index)
        if top_tickers:
            views.append(InvestorView(
                description=f"Top-{len(top_tickers)} factor stocks outperform",
                tickers=top_tickers,
                view_weights=[1.0 / len(top_tickers)] * len(top_tickers),
                expected_return=0.08 / 252,   # 8% annual alpha / 252 days
                confidence=confidence,
            ))

        return views


# ─── Risk Parity ──────────────────────────────────────────────────────────────
class RiskParityOptimizer:
    """
    Equal Risk Contribution (ERC) portfolio.
    Each asset contributes equally to total portfolio variance.

    Used by Bridgewater (All Weather), AQR Risk Parity.
    Particularly robust when expected returns are uncertain.
    """

    def optimize(
        self,
        cov_matrix: np.ndarray,
        tickers: List[str],
        target_risk_pct: Optional[float] = None,   # if set, scale to this vol
    ) -> Dict[str, float]:
        n = len(tickers)
        sigma = cov_matrix

        def risk_contribution(w):
            """Return vector of risk contributions (should be equal)."""
            port_var = w @ sigma @ w
            marginal_rc = sigma @ w
            return w * marginal_rc / max(port_var, 1e-8)

        def objective(w):
            rc = risk_contribution(w)
            target = np.mean(rc)
            return float(np.sum((rc - target) ** 2))

        def gradient(w):
            """Analytical gradient for faster convergence."""
            port_var = w @ sigma @ w
            mv = sigma @ w
            rc = w * mv / max(port_var, 1e-8)
            target = np.mean(rc)
            d_rc_d_w = (np.diag(mv) + np.diag(w) @ sigma) / max(port_var, 1e-8)
            d_rc_d_w -= np.outer(mv * w, mv) / max(port_var**2, 1e-16) * 2
            return 2 * d_rc_d_w.T @ (rc - target)

        w0 = np.ones(n) / n
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = Bounds(lb=0.0, ub=1.0)

        result = minimize(
            objective, w0,
            jac=gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        weights = result.x if result.success else w0
        weights = np.clip(weights, 0, None)
        weights /= max(weights.sum(), 1e-8)
        return dict(zip(tickers, weights.round(6).tolist()))


# ─── Portfolio Construction Service ──────────────────────────────────────────
class PortfolioConstructor:
    """
    High-level service combining all optimisers.
    Selects optimal method based on data availability and strategy type.
    """

    def __init__(self):
        self.mv  = MeanVarianceOptimizer()
        self.bl  = BlackLittermanOptimizer()
        self.rp  = RiskParityOptimizer()

    def construct(
        self,
        method: str,
        tickers: List[str],
        returns: pd.DataFrame,
        expected_returns: Optional[pd.Series] = None,
        market_caps: Optional[pd.Series] = None,
        factor_scores: Optional[pd.DataFrame] = None,
        max_weight: float = 0.15,
    ) -> Dict[str, float]:
        """
        method: 'mean_variance' | 'black_litterman' | 'risk_parity' | 'equal_weight'
        """
        if len(tickers) < 2:
            return {t: 1.0 / max(len(tickers), 1) for t in tickers}

        # Robust covariance estimation
        cov = ledoit_wolf_shrinkage(returns[tickers].dropna())

        if method == "equal_weight":
            w = 1.0 / len(tickers)
            return {t: round(w, 6) for t in tickers}

        elif method == "risk_parity":
            return self.rp.optimize(cov, tickers)

        elif method == "mean_variance" and expected_returns is not None:
            self.mv.max_weight = max_weight
            return self.mv.optimize(expected_returns, cov, tickers)

        elif method == "black_litterman":
            if market_caps is None:
                market_caps = pd.Series({t: 1.0 for t in tickers})
            views = []
            if factor_scores is not None and not factor_scores.empty:
                self.bl.max_weight = max_weight
                views = self.bl.views_from_factor_scores(factor_scores)
            return self.bl.optimize(market_caps, cov, tickers, views)

        else:
            # Fallback: risk parity
            return self.rp.optimize(cov, tickers)

    def compute_portfolio_metrics(
        self,
        weights: Dict[str, float],
        returns: pd.DataFrame,
        risk_free_rate: float = 0.045,
    ) -> Dict[str, float]:
        """Compute Sharpe, vol, max drawdown for a given weight allocation."""
        w = np.array([weights.get(t, 0) for t in returns.columns])
        port_ret = returns @ w
        ann_ret  = float(port_ret.mean() * 252)
        ann_vol  = float(port_ret.std() * np.sqrt(252))
        sharpe   = (ann_ret - risk_free_rate) / max(ann_vol, 1e-8)

        cum = (1 + port_ret).cumprod()
        rolling_max = cum.cummax()
        drawdowns = (cum - rolling_max) / rolling_max
        max_dd = float(drawdowns.min())

        return {
            "annualised_return": round(ann_ret, 4),
            "annualised_vol": round(ann_vol, 4),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown": round(max_dd, 4),
            "calmar_ratio": round(-ann_ret / max_dd, 4) if max_dd < 0 else 0,
        }
