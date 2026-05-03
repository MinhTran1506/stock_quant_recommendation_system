"""
quant/strategies/rl_agent.py — Deep Reinforcement Learning Trading Agent
════════════════════════════════════════════════════════════════════════

Research basis:
  • MacroHFT (KDD 2024) — Memory-Augmented Context-aware RL for HFT;
    hierarchical agent with macro regime context + micro execution.
  • DeepScalper (CIKM 2022) — Risk-aware RL for intraday trading;
    hindsight-augmented reward with drawdown penalties.
  • FinRL (2020) — Open-source framework for financial RL;
    uses PPO/DDPG/SAC on portfolio management tasks.
  • Xiong et al. — DDPG on Chinese equity market (outperforms B&H).
  • Safe-FinRL (2022) — Low bias/variance DRL for high-freq trading.

Architecture: Proximal Policy Optimisation (PPO) agent
  State space:
    • Price features (returns, vol, RSI, MACD) per stock — shape (N, F)
    • Portfolio state (current weights, cash) — shape (N+1,)
    • Regime state (HMM probabilities) — shape (3,)
  Action space:
    • Continuous portfolio weights ∈ [0, 1]^N, sum-to-1 (long-only)
  Reward:
    • Differential Sharpe Ratio (Moody & Saffell 1998) — risk-adjusted
    • Drawdown penalty (DeepScalper style)
    • Transaction cost deduction

Training: Actor-Critic with shared CNN feature extractor + GRU memory
Inference: deterministic policy for live/paper trading
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet

from config import get_settings

settings = get_settings()
logger = structlog.get_logger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── State / Reward ────────────────────────────────────────────────────────────
@dataclass
class PortfolioState:
    """Complete state fed to the RL agent."""
    features: np.ndarray      # (N_stocks, N_features) price/technical features
    weights:  np.ndarray      # (N_stocks + 1,) current weights including cash
    regime:   np.ndarray      # (3,) HMM regime probabilities [bull, side, bear]
    step:     int             # current step in episode


def build_state_features(
    prices: pd.DataFrame,
    window: int = 20,
) -> np.ndarray:
    """
    Build per-stock feature matrix for the current timestep.
    Features: [ret_1d, ret_5d, ret_20d, vol_20d, rsi, macd_hist, bb_pct]
    Shape: (N_stocks, 7)
    """
    features = []
    for ticker in prices.columns:
        close = prices[ticker].dropna()
        if len(close) < window + 10:
            features.append(np.zeros(7))
            continue

        c = close.values
        ret_1d  = (c[-1] / c[-2] - 1) if len(c) > 1 else 0
        ret_5d  = (c[-1] / c[-5] - 1) if len(c) > 5 else 0
        ret_20d = (c[-1] / c[-20] - 1) if len(c) > 20 else 0
        vol_20d = float(np.std(np.diff(np.log(c[-21:]))) * np.sqrt(252)) if len(c) > 21 else 0

        # RSI
        delta = np.diff(c[-15:])
        gain = np.mean(delta[delta > 0]) if any(delta > 0) else 0
        loss = abs(np.mean(delta[delta < 0])) if any(delta < 0) else 1e-6
        rsi = 100 - 100 / (1 + gain / loss)

        # MACD histogram
        ema12 = float(pd.Series(c).ewm(span=12).mean().iloc[-1])
        ema26 = float(pd.Series(c).ewm(span=26).mean().iloc[-1])
        macd  = ema12 - ema26
        signal = float(pd.Series(c).ewm(span=9).mean().iloc[-1])
        macd_hist = (macd - signal) / (abs(signal) + 1e-8)

        # Bollinger %B
        ma20   = float(np.mean(c[-20:]))
        std20  = float(np.std(c[-20:]))
        bb_pct = (c[-1] - (ma20 - 2 * std20)) / (4 * std20 + 1e-8)

        features.append([ret_1d, ret_5d, ret_20d, vol_20d,
                          (rsi - 50) / 50, macd_hist, bb_pct])

    return np.array(features, dtype=np.float32)


# ─── Neural Network Architecture ──────────────────────────────────────────────
class PortfolioActorCritic(nn.Module):
    """
    Shared actor-critic network for PPO.

    Inspired by MacroHFT's hierarchical architecture:
      1. Stock-level encoder: per-stock CNN over feature window
      2. Cross-stock attention: captures correlations between stocks
      3. GRU memory: retains recent decision context
      4. Actor head: outputs Dirichlet concentration params → weights
      5. Critic head: state value estimate
    """

    def __init__(
        self,
        n_stocks: int,
        n_features: int = 7,
        n_regime_features: int = 3,
        hidden_dim: int = 128,
        gru_hidden: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_stocks = n_stocks

        # Per-stock feature encoder
        self.stock_encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Cross-stock self-attention (captures sector correlations)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim // 2,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim // 2)

        # Regime embedding
        self.regime_embed = nn.Sequential(
            nn.Linear(n_regime_features, 16),
            nn.ReLU(),
        )

        # GRU for temporal memory (portfolio state + regime)
        gru_in = hidden_dim // 2 + n_stocks + 1 + 16   # attn_out + weights + cash + regime
        self.gru = nn.GRU(gru_in, gru_hidden, batch_first=True)
        self.gru_norm = nn.LayerNorm(gru_hidden)

        # Actor head: Dirichlet concentration parameters
        # Dirichlet(α) → portfolio weights (simplex; all positive, sum to 1)
        self.actor = nn.Sequential(
            nn.Linear(gru_hidden, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_stocks + 1),  # +1 for cash
            nn.Softplus(),  # ensure positive concentrations
        )

        # Critic head: scalar value
        self.critic = nn.Sequential(
            nn.Linear(gru_hidden, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._hidden: Optional[torch.Tensor] = None

    def reset_hidden(self):
        self._hidden = None

    def forward(
        self,
        stock_features: torch.Tensor,   # (B, N_stocks, N_features)
        portfolio_weights: torch.Tensor, # (B, N_stocks + 1)
        regime: torch.Tensor,           # (B, 3)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = stock_features.shape[0]

        # Encode each stock
        encoded = self.stock_encoder(stock_features)   # (B, N, H/2)

        # Cross-stock attention
        attn_out, _ = self.attn(encoded, encoded, encoded)
        attn_out = self.attn_norm(attn_out + encoded)  # residual
        pooled = attn_out.mean(dim=1)                  # (B, H/2) — global portfolio view

        # Regime embedding
        regime_emb = self.regime_embed(regime)         # (B, 16)

        # Combine for GRU input
        gru_in = torch.cat([pooled, portfolio_weights, regime_emb], dim=-1)
        gru_in = gru_in.unsqueeze(1)                   # (B, 1, features)

        gru_out, self._hidden = self.gru(gru_in, self._hidden)
        gru_out = self.gru_norm(gru_out.squeeze(1))    # (B, gru_hidden)

        # Actor: Dirichlet concentration params
        alpha = self.actor(gru_out) + 1e-3             # (B, N+1), ensure > 0
        # Critic: state value
        value = self.critic(gru_out)                   # (B, 1)

        return alpha, value


# ─── Differential Sharpe Reward ────────────────────────────────────────────────
class DifferentialSharpeReward:
    """
    Moody & Saffell (1998) Differential Sharpe Ratio reward.
    Optimises risk-adjusted return online without end-of-episode.

    r_t = (B_{t-1} * Δr_t - 0.5 * A_{t-1} * Δr_t²) / (B_{t-1} - A_{t-1}²)^0.5
    where A, B are exponential moving averages of returns and squared returns.
    """

    def __init__(self, eta: float = 0.01, drawdown_lambda: float = 0.5):
        self.eta = eta
        self.drawdown_lambda = drawdown_lambda
        self._A: float = 0.0
        self._B: float = 0.0
        self._peak_value: float = 1.0

    def compute(self, portfolio_return: float, portfolio_value: float) -> float:
        """Compute reward for a single step."""
        r = portfolio_return
        # Update exponential averages
        self._A = self._A + self.eta * (r - self._A)
        self._B = self._B + self.eta * (r**2 - self._B)

        # Differential Sharpe
        denom = max(self._B - self._A**2, 1e-8)
        dsr = (self._B * r - 0.5 * self._A * r**2) / (denom**0.5)

        # Drawdown penalty (DeepScalper style)
        self._peak_value = max(self._peak_value, portfolio_value)
        drawdown = (self._peak_value - portfolio_value) / self._peak_value
        penalty = self.drawdown_lambda * drawdown

        return float(dsr - penalty)

    def reset(self):
        self._A = 0.0
        self._B = 0.0
        self._peak_value = 1.0


# ─── PPO Trainer ──────────────────────────────────────────────────────────────
class PPOTrainer:
    """Proximal Policy Optimisation trainer for the portfolio agent."""

    def __init__(
        self,
        n_stocks: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        update_epochs: int = 10,
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.update_epochs = update_epochs

        self.net = PortfolioActorCritic(n_stocks=n_stocks).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.reward_fn = DifferentialSharpeReward()

    def select_action(
        self,
        state: PortfolioState,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Sample portfolio weights from Dirichlet policy.
        Returns (weights, log_prob, value).
        """
        with torch.no_grad():
            features = torch.tensor(state.features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            weights  = torch.tensor(state.weights, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            regime   = torch.tensor(state.regime, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            alpha, value = self.net(features, weights, regime)
            dist = Dirichlet(alpha.squeeze(0))

            if deterministic:
                # Mode of Dirichlet: (α-1)/(sum(α)-K)
                action = (alpha.squeeze(0) - 1).clamp(min=0)
                action = action / (action.sum() + 1e-8)
                log_prob = dist.log_prob(action).item()
            else:
                action = dist.sample()
                log_prob = dist.log_prob(action).item()

        return action.cpu().numpy(), log_prob, value.item()

    def update(
        self,
        trajectories: List[Dict],
    ) -> Dict[str, float]:
        """PPO policy update from collected trajectories."""
        if not trajectories:
            return {}

        # Unpack trajectories
        all_features  = torch.stack([t["features"] for t in trajectories]).to(DEVICE)
        all_weights   = torch.stack([t["weights"] for t in trajectories]).to(DEVICE)
        all_regime    = torch.stack([t["regime"] for t in trajectories]).to(DEVICE)
        all_actions   = torch.stack([t["action"] for t in trajectories]).to(DEVICE)
        old_log_probs = torch.stack([t["log_prob"] for t in trajectories]).to(DEVICE)
        returns       = torch.stack([t["return"] for t in trajectories]).to(DEVICE)

        # Normalise returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0

        for _ in range(self.update_epochs):
            alpha, values = self.net(all_features, all_weights, all_regime)
            dist = Dirichlet(alpha)
            log_probs = dist.log_prob(all_actions)
            entropy   = dist.entropy().mean()

            # Clipped PPO objective
            ratio = (log_probs - old_log_probs).exp()
            adv   = returns - values.squeeze()
            p1    = ratio * adv
            p2    = ratio.clamp(1 - self.eps_clip, 1 + self.eps_clip) * adv
            policy_loss = -torch.min(p1, p2).mean()
            value_loss  = F.mse_loss(values.squeeze(), returns)

            loss = (policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss  += value_loss.item()
            total_entropy     += entropy.item()

        n = self.update_epochs
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss":  total_value_loss / n,
            "entropy":     total_entropy / n,
        }


# ─── RL Portfolio Agent (inference wrapper) ───────────────────────────────────
class RLPortfolioAgent:
    """
    High-level RL agent for live/paper trading inference.

    Usage:
        agent = RLPortfolioAgent.load(mlflow_run_id)
        weights = agent.allocate(prices_df, current_weights, regime_probs)
    """

    def __init__(self, n_stocks: int, tickers: List[str]):
        self.n_stocks = n_stocks
        self.tickers = tickers
        self._trainer: Optional[PPOTrainer] = None

    def allocate(
        self,
        prices: pd.DataFrame,
        current_weights: Optional[np.ndarray] = None,
        regime_probs: Optional[np.ndarray] = None,
        transaction_cost: float = 0.0015,
    ) -> Dict[str, float]:
        """
        Compute optimal portfolio weights.
        Returns {ticker: weight} including cash.
        """
        if self._trainer is None:
            # Fallback: equal weight
            w = 1.0 / (len(self.tickers) + 1)
            return {t: w for t in self.tickers} | {"CASH": w}

        features = build_state_features(prices[self.tickers], window=20)
        if current_weights is None:
            current_weights = np.ones(len(self.tickers) + 1) / (len(self.tickers) + 1)
        if regime_probs is None:
            regime_probs = np.array([0.6, 0.3, 0.1])

        state = PortfolioState(
            features=features,
            weights=current_weights,
            regime=regime_probs,
            step=0,
        )
        self._trainer.net.reset_hidden()
        weights, _, _ = self._trainer.select_action(state, deterministic=True)

        result = {t: round(float(weights[i]), 4)
                  for i, t in enumerate(self.tickers)}
        result["CASH"] = round(float(weights[-1]), 4)
        return result

    def train(
        self,
        prices: pd.DataFrame,
        tickers: List[str],
        n_episodes: int = 200,
        episode_length: int = 252,
        experiment_name: Optional[str] = None,
    ) -> Dict[str, float]:
        """Train the PPO agent via environment simulation."""
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name or settings.mlflow_experiment_name)

        self._trainer = PPOTrainer(n_stocks=len(tickers))
        all_returns   = []

        with mlflow.start_run(run_name=f"RLAgent_PPO_{len(tickers)}stocks") as run:
            mlflow.log_params({
                "n_stocks": len(tickers), "n_episodes": n_episodes,
                "episode_length": episode_length, "algorithm": "PPO",
                "architecture": "ActorCritic+GRU+MultiheadAttention",
            })

            for episode in range(n_episodes):
                ep_return, trajectories = self._run_episode(
                    prices, tickers, episode_length
                )
                update_stats = self._trainer.update(trajectories)
                all_returns.append(ep_return)

                if episode % 20 == 0:
                    mlflow.log_metrics({
                        "episode_return": ep_return,
                        "mean_return_last20": float(np.mean(all_returns[-20:])),
                        **update_stats,
                    }, step=episode)
                    logger.info(
                        "RL training episode",
                        episode=episode,
                        ep_return=round(ep_return, 4),
                        **{k: round(v, 4) for k, v in update_stats.items()},
                    )

            final_metrics = {
                "mean_return": float(np.mean(all_returns)),
                "std_return":  float(np.std(all_returns)),
                "sharpe":      float(np.mean(all_returns) / (np.std(all_returns) + 1e-8) * np.sqrt(252)),
            }
            mlflow.log_metrics(final_metrics)

            # Save model
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "rl_agent.pt")
                torch.save({
                    "net": self._trainer.net.state_dict(),
                    "n_stocks": len(tickers),
                    "tickers": tickers,
                }, path)
                mlflow.log_artifact(path, artifact_path="model")

            logger.info("RL agent training complete", **final_metrics)
            return final_metrics

    def _run_episode(
        self,
        prices: pd.DataFrame,
        tickers: List[str],
        episode_length: int,
    ) -> Tuple[float, List[Dict]]:
        """Simulate one trading episode; return (total_return, trajectories)."""
        self._trainer.net.reset_hidden()
        self._trainer.reward_fn.reset()

        n = len(prices)
        start = np.random.randint(60, max(61, n - episode_length - 1))
        weights = np.ones(len(tickers) + 1) / (len(tickers) + 1)
        portfolio_value = 1.0
        trajectories: List[Dict] = []
        episode_return = 0.0

        for t in range(episode_length):
            if start + t >= n:
                break

            window = prices.iloc[max(0, start + t - 60): start + t + 1]
            if len(window) < 5:
                continue

            features = build_state_features(window[tickers])
            regime_probs = np.array([0.6, 0.3, 0.1], dtype=np.float32)

            state = PortfolioState(
                features=features,
                weights=weights,
                regime=regime_probs,
                step=t,
            )
            action, log_prob, value = self._trainer.select_action(state)

            # Compute portfolio return
            next_idx = min(start + t + 1, n - 1)
            curr_prices = prices.iloc[start + t][tickers].values
            next_prices = prices.iloc[next_idx][tickers].values
            valid = (curr_prices > 0) & (next_prices > 0)
            stock_returns = np.where(valid, next_prices / curr_prices - 1, 0)
            stock_weights = action[:len(tickers)]
            portfolio_return = float(np.dot(stock_weights, stock_returns))

            # Transaction cost
            turnover = float(np.sum(np.abs(action - weights)))
            tc = turnover * 0.0015
            portfolio_return -= tc

            portfolio_value *= (1 + portfolio_return)
            reward = self._trainer.reward_fn.compute(portfolio_return, portfolio_value)
            episode_return += portfolio_return
            weights = action.copy()

            trajectories.append({
                "features":  torch.tensor(features, dtype=torch.float32),
                "weights":   torch.tensor(weights, dtype=torch.float32),
                "regime":    torch.tensor(regime_probs, dtype=torch.float32),
                "action":    torch.tensor(action, dtype=torch.float32),
                "log_prob":  torch.tensor(log_prob, dtype=torch.float32),
                "return":    torch.tensor(reward, dtype=torch.float32),
            })

        return episode_return, trajectories

    @classmethod
    def load(cls, mlflow_run_id: str) -> "RLPortfolioAgent":
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        artifact_uri = mlflow.get_run(mlflow_run_id).info.artifact_uri
        with tempfile.TemporaryDirectory() as tmp:
            local = mlflow.artifacts.download_artifacts(
                f"{artifact_uri}/model/rl_agent.pt", dst_path=tmp
            )
            checkpoint = torch.load(local, map_location=DEVICE)

        tickers = checkpoint["tickers"]
        agent = cls(n_stocks=checkpoint["n_stocks"], tickers=tickers)
        agent._trainer = PPOTrainer(n_stocks=checkpoint["n_stocks"])
        agent._trainer.net.load_state_dict(checkpoint["net"])
        return agent
