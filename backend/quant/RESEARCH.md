# Quantitative Trading — Research Foundation
## Papers & Algorithms Implemented

This document maps each implemented algorithm to its academic source,
explaining the connection between theory and code in the platform.

---

## 1. Statistical Arbitrage (`quant/strategies/stat_arb.py`)

| Paper | Year | Key Contribution | Used In |
|---|---|---|---|
| Gatev, Goetzmann & Rouwenhorst | 2006 | Pairs trading performance; distance-based pair selection | `PairsFinder` |
| Avellaneda & Lee | 2008 | PCA factor residual stat arb (Morgan Stanley PDT approach) | `StatArbStrategy` |
| Engle & Granger | 1987 | Cointegration (two-step test) | `_test_pair` ADF step |
| Johansen | 1991 | Multivariate cointegration test; handles both directions | `coint_johansen` |
| Elliott, van der Hoek & Malcolm | 2005 | Kalman Filter for dynamic hedge ratio | `KalmanHedge` |

**Key insight:** Static OLS hedge ratios break down as stock relationships evolve.
The Kalman Filter adapts the hedge ratio in real time, dramatically reducing
spread model error in non-stationary markets (Vietnam included).

**Vietnam adaptation:** Short-selling restrictions mean the strategy is
configured `long_only=True` by default — only the underperforming stock is
bought; no short position is opened on the outperforming stock.

---

## 2. Multi-Factor Alpha Model (`quant/strategies/factor_model.py`)

| Paper | Year | Factor | Code |
|---|---|---|---|
| Fama & French | 1993 | Market, SMB (size), HML (value) | `_value_factor`, `_size_factor` |
| Fama & French | 2015 | + RMW (profitability), CMA (investment) | `_quality_factor` |
| Jegadeesh & Titman | 1993 | Cross-sectional momentum (12-1m) | `_momentum_factor` |
| Asness, Moskowitz & Pedersen | 2013 | Value + Momentum everywhere (AQR) | Combined in `compute_scores` |
| Novy-Marx | 2013 | Gross profitability (quality factor) | `_quality_factor` |
| Frazzini & Pedersen | 2014 | Betting Against Beta (low-vol anomaly) | `_low_vol_factor` |
| Amihud | 2002 | Illiquidity ratio (liquidity premium) | `_liquidity_factor` |

**Factor weights** use Information Coefficient (IC) blending — factors with
higher historical IC receive proportionally higher weight. The composite
score replicates the core of AQR's equity factor strategy.

---

## 3. Regime Detection + Momentum (`quant/strategies/momentum_regime.py`)

| Paper | Year | Contribution | Code |
|---|---|---|---|
| Hamilton | 1989 | HMM regime-switching model | `MarketRegimeDetector` |
| Jegadeesh & Titman | 1993 | Cross-sectional momentum | `CrossSectionalMomentum` |
| Moskowitz, Ooi & Pedersen | 2012 | Time-series momentum (TSMOM) | `TimeSeriesMomentum` |
| Daniel & Moskowitz | 2016 | Momentum crashes in bear markets | Regime gate in `generate_signals` |
| Bloch | 2025 | RMA framework for adaptive deployment | Continuous scalar scaling |

**Critical insight from Daniel & Moskowitz (2016):** Momentum strategies
suffer severe drawdowns specifically during market reversals (bear-to-bull
transitions). The HMM regime detector gates momentum OFF during Bear states,
preventing these catastrophic crashes.

---

## 4. Deep RL Portfolio Agent (`quant/strategies/rl_agent.py`)

| Paper | Year | Contribution | Code |
|---|---|---|---|
| Moody & Saffell | 1998 | Differential Sharpe Ratio reward | `DifferentialSharpeReward` |
| Schulman et al. | 2017 | PPO (Proximal Policy Optimisation) | `PPOTrainer` |
| Li et al. / MacroHFT | KDD 2024 | Memory-augmented context-aware RL for HFT | `PortfolioActorCritic` GRU |
| DeepScalper | CIKM 2022 | Risk-aware RL with drawdown penalty | Drawdown penalty in reward |
| Xiong et al. | 2018 | DDPG on Chinese stock market | Architecture inspiration |
| Safe-FinRL | 2022 | Low bias/variance DRL for high-freq trading | Gradient clipping, entropy |

**Architecture novelties:**
- **Dirichlet policy** (vs. softmax): naturally enforces sum-to-1 simplex
  constraint for portfolio weights; no separate projection step needed.
- **Multi-head attention** across stocks: learns correlation structure
  dynamically (similar to MASTER architecture, AAAI 2024).
- **GRU temporal memory**: retains context across trading steps
  (inspired by MacroHFT's memory augmentation, KDD 2024).

---

## 5. Portfolio Optimisation (`quant/portfolio/optimizer.py`)

| Paper | Year | Method | Code |
|---|---|---|---|
| Markowitz | 1952 | Mean-Variance Optimisation | `MeanVarianceOptimizer` |
| Black & Litterman | 1990 | Bayesian equilibrium + views | `BlackLittermanOptimizer` |
| Roncalli | 2013 | Equal Risk Contribution (Risk Parity) | `RiskParityOptimizer` |
| Ledoit & Wolf | 2004 | Analytical covariance shrinkage | `ledoit_wolf_shrinkage` |
| NeurIPS 2024 | 2024 | m-Sparse Sharpe Ratio max | `max_weight` constraint |

**Black-Litterman advantage:** Standard Markowitz MVO produces extreme,
unintuitive weights from noisy return estimates. BL combines the market's
"wisdom" (CAPM equilibrium) with your ML model's views via Bayesian updating,
producing stable, well-diversified portfolios that trade only when the model
has genuine alpha.

---

## 6. Risk Management (`quant/risk/risk_manager.py`)

| Paper / Standard | Contribution | Code |
|---|---|---|
| Rockafellar & Uryasev (2000) | CVaR (Expected Shortfall) | `compute_risk_report` CVaR |
| Basel III/IV | Regulatory VaR / CVaR framework | `RiskLimits` defaults |
| Roncalli (2020) | Handbook on Financial Risk Management | Overall architecture |

---

## Strategy Selection Guide

```
Market Condition          Recommended Strategy           Notes
─────────────────────────────────────────────────────────────────────
Bull, low vol            Cross-Sectional Momentum       Full weight
Bull, trending           Time-Series Momentum (TSMOM)   Vol-scaled
Sideways/range-bound     Statistical Arbitrage          Mean-reversion
High vol / uncertain     Risk Parity Portfolio          Stable weights
High conviction views    Black-Litterman                Views + equil.
Unknown / all-weather    RL Agent (trained)             Adaptive
─────────────────────────────────────────────────────────────────────
```

## Vietnam-Specific Adaptations

1. **No short-selling** (long-only constraint) on HOSE/HNX → all strategies
   default to `long_only=True`; negative signals are simply excluded rather
   than shorted.

2. **Daily price limits** (±7% on HOSE) → stop-loss and position sizing
   must account for the possibility that limit-down stocks cannot be exited.

3. **T+2.5 settlement** → strategies with high turnover incur cash drag;
   rebalance frequency is limited to weekly/monthly for most strategies.

4. **Market depth** → position sizes are capped relative to average daily
   volume (ADV) to avoid market impact on smaller-cap stocks.

5. **Phase 0 regulatory check** still applies → all execution goes through
   paper trading until legal clearance is obtained.
