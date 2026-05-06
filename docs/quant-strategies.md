# Quant Strategies

Detailed reference for all quantitative strategies, portfolio optimizers, risk management controls, and the backtesting engine.

---

## Strategies Overview

| Strategy | Module | Frequency | Market Suitability |
|----------|--------|-----------|-------------------|
| [Factor Model](#1-factor-model) | `factor_model.py` | Daily | Trending / liquid markets |
| [Statistical Arbitrage](#2-statistical-arbitrage) | `stat_arb.py` | Daily–Intraday | Sideways, mean-reverting |
| [Momentum + Regime](#3-momentum--regime-detection) | `momentum_regime.py` | Daily | Trending markets |
| [Mean Reversion](#4-mean-reversion) | `mean_reversion.py` | Daily | Range-bound |
| [Order Flow](#5-order-flow--microstructure) | `order_flow.py` | Intraday | High-frequency |
| [RL Agent](#6-reinforcement-learning-agent) | `rl_agent.py` | Daily | All regimes (adaptive) |

---

## 1. Factor Model

**File:** `backend/quant/strategies/factor_model.py`

A classic multi-factor cross-sectional model. Stocks are ranked by composite factor score each day; the top N become the portfolio.

### Factors and Weights

| Factor | Weight | Signal | Look-back |
|--------|--------|--------|-----------|
| Momentum (12-1m) | 0.25 | 12-month return minus last month | 252d |
| Reversal | -0.10 | Prior 1-month return (negative weight = fade) | 21d |
| Value (P/B) | 0.15 | Low price-to-book | trailing |
| Quality (ROE) | 0.20 | High return-on-equity | trailing |
| Low Volatility | 0.15 | Low 252-day realised vol | 252d |
| Size | 0.05 | Small market cap (small-cap premium) | trailing |
| Growth | 0.10 | Revenue/earnings growth | trailing |
| Liquidity | 0.10 | High average daily turnover | 63d |

**Composite score** = weighted z-score sum across all factors.

### Ranking

- Stocks are ranked `ascending=False, method="first"` to eliminate ties.
- Top 20 stocks (configurable) are selected for the long-only portfolio.

### Output

```python
FactorSignal:
    ticker: str
    composite_score: float      # z-score weighted sum
    factor_scores: Dict[str, float]  # individual z-scores
    rank: int                   # 1 = best
    signal: Literal["BUY", "HOLD", "AVOID"]
```

---

## 2. Statistical Arbitrage

**File:** `backend/quant/strategies/stat_arb.py`

Pairs trading using cointegration — finds pairs of stocks whose prices move together long-term, trades when their spread diverges.

### Pair Discovery (`PairsFinder`)

Screening criteria:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `min_corr` | 0.65 | Minimum Pearson correlation |
| `max_half_life_days` | 63 | Max mean-reversion speed (~3 months) |
| `lookback_days` | 252 | Cointegration test window |

Cointegration test: **Johansen test** for long-run equilibrium.

### Dynamic Hedge Ratio (`KalmanHedge`)

The hedge ratio β (stock A vs stock B) is estimated in real time using a **Kalman Filter**, adapting to structural breaks in the relationship.

### Spread Model (`OUSpreadModel`)

Models spread as an **Ornstein-Uhlenbeck** process:

$$dS_t = \kappa(\theta - S_t)\,dt + \sigma\,dW_t$$

Parameters estimated: mean-reversion speed κ, long-run mean θ, volatility σ, and half-life = ln(2) / κ.

### Entry / Exit Rules

| Signal | Trigger |
|--------|---------|
| Enter Long spread | z-score < −2.0 |
| Enter Short spread | z-score > +2.0 |
| Exit | \|z-score\| < 0.5 |
| Stop-loss | \|z-score\| > 4.0 |

> **Note:** Vietnam restricts short-selling, so only the long-only variant is used in practice. The strategy goes long the underperformer and reduces/exits the overperformer.

### Output

```python
SpreadSignal:
    ticker_a: str
    ticker_b: str
    z_score: float
    signal: Literal["LONG_A", "LONG_B", "EXIT", "NONE"]
    half_life: float        # days
    confidence: float       # 0–1
```

---

## 3. Momentum + Regime Detection

**File:** `backend/quant/strategies/momentum_regime.py`

Combines cross-sectional momentum with Hidden Markov Model (HMM) market regime detection to scale exposure dynamically.

### Market Regime Detection (`MarketRegimeDetector`)

A 3-state HMM is fitted on market returns + realised volatility:

| State | Label | Characteristics | Momentum Scalar |
|-------|-------|-----------------|-----------------|
| High return, low vol | BULL | Trending up | 1.0× |
| Low return, high vol | BEAR | Trending down | 0.0× |
| Mixed | SIDEWAYS | Range-bound | 0.5× |

The `momentum_scalar` dampens position sizing based on:
1. Dominant regime probability (bull/sideways/bear weighted average)
2. A `return_weight` factor: if recent market returns are near −2%/day, exposure approaches 0

### Cross-Sectional Momentum (`CrossSectionalMomentum`)

- **Signal:** 12-month return minus last month (12-1m momentum)
- **Long:** Top decile of universe by momentum
- **Short:** Bottom decile (skipped in long-only Vietnam mode)
- **Rebalance:** Monthly

### Time Series Momentum (`TimeSeriesMomentum`)

Each stock trades by its own 12-month trend:
- If own 12m return > 0 → long
- If own 12m return < 0 → flat (no short)

### Output

```python
RegimeState:
    regime: Literal["BULL", "SIDEWAYS", "BEAR"]
    bull_prob: float
    sideways_prob: float
    bear_prob: float
    momentum_scalar: float   # 0.0–1.0, caps at 0.6
```

---

## 4. Mean Reversion

**File:** `backend/quant/strategies/mean_reversion.py`

### Bollinger Band Reversion (`BollingerBandReversion`)

| Parameter | Default |
|-----------|---------|
| Window | 20 days |
| Bands | ±2σ |
| Confirmation | RSI < 35 (oversold) or RSI > 65 (overbought) |
| Horizon | Days |

### RSI Extremes Reversion (`RSIExtremesReversion`)

| Parameter | Default |
|-----------|---------|
| RSI window | 14 days |
| Oversold threshold | 30 |
| Overbought threshold | 70 |

### Short-term Reversal

Fade prior-week losers; typical 1-week holding period.

### Long-run Contrarian

Fade prior 5-year cumulative losers; 3–5 year holding horizon.

---

## 5. Order Flow / Microstructure

**File:** `backend/quant/strategies/order_flow.py`

Intraday alpha signals derived from order book and trade flow data.

| Signal | Model | Horizon | Formula |
|--------|-------|---------|---------|
| Order Book Imbalance (OBI) | `OrderBookImbalanceModel` | 1–5 min | (BidVol − AskVol) / (BidVol + AskVol) |
| VPIN | `VPINCalculator` | 30–60 min | \|BuyVol − SellVol\| / TotalVol |
| Trade Imbalance | `TradeImbalanceModel` | 5–15 min | Buy% − Sell% |
| Kyle Lambda | `KyleLambdaModel` | 1–5 min | ΔPrice / ΔVolume |
| Toxic Flow | Composite | 15–60 min | High VPIN + buy/sell divergence |

### Output

```python
MicrostructureSignal:
    ticker: str
    obi: float              # order book imbalance −1 to +1
    vpin: float             # 0 to 1
    kyle_lambda: float      # price impact per unit volume
    trade_imbalance: float
    short_term_signal: Literal["BUY", "SELL", "NEUTRAL"]
    confidence: float
```

---

## 6. Reinforcement Learning Agent

**File:** `backend/quant/strategies/rl_agent.py`

A PPO (Proximal Policy Optimization) agent that learns a continuous portfolio allocation policy.

### State Space (per step)

| Component | Shape | Contents |
|-----------|-------|---------|
| Per-stock features | (N, 7) | Returns (1d/5d/20d), volatility, RSI, MACD, Bollinger %B |
| Portfolio weights | (N+1,) | Current allocations including cash |
| Regime features | (3,) | HMM probabilities [bull, sideways, bear] |

### Action Space

Continuous portfolio weights ∈ [0, 1]^N (long-only, sum-to-1 constraint).

### Reward Function

$$r_t = \Delta \text{Sharpe}_t - \lambda_{\text{dd}} \cdot \mathbb{1}[\text{drawdown} > \theta] - c \cdot \|\Delta w_t\|_1$$

Components:
- Differential Sharpe Ratio (risk-adjusted performance)
- Drawdown penalty when drawdown exceeds threshold
- Transaction cost deduction proportional to turnover

### Architecture

- Actor-Critic with shared CNN feature extractor + GRU memory
- Deterministic policy for live inference
- Retrained weekly via Airflow (`quant_daily_signals` DAG)

---

## Portfolio Optimizers

**File:** `backend/quant/portfolio/optimizer.py`

All optimizers return `Dict[str, float]` mapping ticker → weight, summing to 1.0.

### Mean-Variance Optimization (`MeanVarianceOptimizer`)

- **Objective:** Maximize Sharpe ratio
- **Solver:** SLSQP
- **Constraints:** Long-only, sum-to-1, max weight 20%, min weight 0%
- **Risk-free rate:** 4.5% (Vietnam policy rate)
- **Covariance:** Ledoit-Wolf shrinkage (robust for small N/T)

### Black-Litterman (`BlackLittermanOptimizer`)

- **Objective:** Combine CAPM equilibrium returns with ML model views
- **Formula:** Posterior weights blend market cap weights with signal-implied views
- **Parameters:** `risk_aversion=2.5`, `tau=0.025`
- **Benefit:** Reduces extreme corner solutions common in raw MVO

### Risk Parity / ERC (`RiskParityOptimizer`)

- **Objective:** Equal Risk Contribution — each stock contributes the same marginal portfolio volatility
- **Method:** Convex optimization (CVXPY)
- **Benefit:** Pure diversification, no return estimation needed

### Minimum Variance

- **Objective:** Minimize portfolio variance
- **Use case:** Defensive allocation during high-uncertainty regimes

### Maximum Diversification (`MaxDiversificationOptimizer`)

- **Objective:** Maximize diversification ratio = Σ(σᵢ wᵢ) / σ_portfolio
- **Use case:** Maximize exposure to independent return sources

---

## Risk Manager

**File:** `backend/quant/risk/risk_manager.py`

### Risk Limits (Defaults)

| Limit | Default | Action on Breach |
|-------|---------|------------------|
| `max_drawdown_pct` | 15% | Reduce exposure |
| `hard_stop_drawdown_pct` | 20% | Halt all trading |
| `var_95_limit_pct` | 2% (1-day) | Warning |
| `cvar_95_limit_pct` | 3% (1-day) | Warning |
| `max_portfolio_vol` | 20% annualized | Warning |
| `max_single_position_pct` | 15% | Reject order |
| `max_sector_concentration_pct` | 40% | Warning |
| `max_correlation_exposure` | 0.60 avg pairwise | Warning |
| `max_gross_leverage` | 1.0× | Reject order |

### Risk Metrics (`RiskReport`)

```
var_95_1d, var_99_1d         — Value-at-Risk (parametric Normal)
cvar_95_1d, cvar_99_1d       — Conditional VaR (Expected Shortfall)
current_drawdown             — Current peak-to-trough drawdown
max_drawdown                 — Historical maximum drawdown
annualised_vol               — Annualized portfolio volatility
sharpe_ratio                 — Annualized risk-adjusted return
sortino_ratio                — Downside risk-adjusted return
beta                         — Market beta vs VN-Index
max_position_pct             — Largest single position
breaches: List[str]          — Names of violated limits
action_required              — NONE | REDUCE | HALT
```

VaR computation supports three methods: Parametric (Normal), Historical Simulation, Monte Carlo.

### Pre-Trade Checks

Before each order, the risk manager verifies:
1. Single position will not exceed `max_single_position_pct`
2. Sector will not exceed `max_sector_concentration_pct`
3. Gross leverage will not exceed `max_gross_leverage`

Returns `(approved: bool, reason: str)`.

---

## Backtesting Engine

**File:** `backend/backtest/engine.py`

### BacktestConfig

```python
BacktestConfig(
    start_date: str             # "2020-01-01"
    end_date: str               # "2024-01-01"
    initial_capital: float      # 1,000,000,000 VND default
    commission_pct: float       # 0.15% per side
    slippage_pct: float         # 0.1%
    latency_ms: int             # 0 (HFT research use)
    max_position_pct: float     # 10% per position
    max_positions: int          # 20 concurrent
    stop_loss_pct: float        # 7%
    take_profit_pct: float      # None (optional)
    order_fill_price: str       # "next_open" | "close" | "vwap"
    allow_short: bool           # False (Vietnam restriction)
    rebalance_frequency: str    # "daily" | "weekly" | "monthly"
    universe_filter: List[str]  # None = all stocks
)
```

### Two Engines

**VectorBTEngine** — Vectorized fast backtester
- Built on vectorbt 0.26.1
- Processes entire history in one pass
- Ideal for parameter sweeps and research
- `freq="D"` chunked time series

**BacktraderEngine** — Event-driven realism
- Partial fills, latency simulation, tiered commissions
- Closer to live execution behavior

### BacktestResults Output

```
total_return_pct, annualised_return_pct
benchmark_return_pct          — VN-Index comparison
sharpe_ratio, sortino_ratio, calmar_ratio
information_ratio             — Active return / tracking error
max_drawdown_pct, max_drawdown_duration_days
volatility_annualised
total_trades, win_rate
avg_win_pct, avg_loss_pct, profit_factor
avg_holding_days
equity_curve                  — [{date, value}, ...]
drawdown_series               — [{date, drawdown_pct}, ...]
monthly_returns               — [{month, return_pct}, ...]
trade_log                     — [{ticker, entry_date, exit_date, ...}, ...]
```

---

## ML Models

### Forecasting Stack

| Model | File | Architecture | Output |
|-------|------|-------------|--------|
| TFT | `tft.py` | Temporal Fusion Transformer (NeuralForecast) | Quantile forecasts for horizons [1,3,5,10,20]d |
| N-BEATS | `nbeats.py` | N-BEATS + N-HiTS ensemble | Quantile forecasts for horizons [1..20]d |
| TCN | `tcn.py` | Dilated causal conv + residuals | Intraday (1-min to 1-hour) predictions |
| GNN | `gnn.py` | GraphSAGE (sector + correlation graph) | Stock relationship-aware embeddings |
| Meta-model | `meta_model.py` | LightGBM LambdaRank | Stock score 0–100 |

### Meta-Ranking Model (LightGBM)

Combines outputs from all base models to produce a final stock ranking:

- **Objective:** LambdaRank (cross-sectional ranking)
- **Target:** Forward 5-day return percentile rank
- **Features:** 48 total
  - 15 TFT quantile outputs (q10/q50/q90 × 5 horizons)
  - 5 N-BEATS/N-HiTS outputs
  - 19 technical features (returns, RSI, MACD, Bollinger %B, volume ratios, etc.)
  - 6 fundamental features (P/E, P/B, ROE, ROA, D/E, dividend yield)
  - 3 sentiment features (1d/7d sentiment score, 7d news count)

**Output:** SHAP feature importance + stock score 0–100

### Training Pipeline

Orchestrated by `backend/models/training_pipeline.py`:

1. Load & validate data from TimescaleDB
2. Train TFT
3. Train N-BEATS/N-HiTS ensemble
4. Train TCN (if minute data available)
5. Generate base-model predictions for meta-model features
6. Train LightGBM meta-ranker
7. Walk-forward validation
8. Register champion model in MLflow
9. Update `model_versions` table in DB

Default periods: 3-year train, 6-month validation, 3-month test.
