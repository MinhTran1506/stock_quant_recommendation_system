"""
models/tcn.py — Temporal Convolutional Network for intraday/HFT research.

TCN uses dilated causal convolutions with residual connections to model
high-frequency time series patterns at tick/minute resolution.

Key properties:
  - Strictly causal (no lookahead): conv filters only see past values
  - Dilated convolutions: exponentially growing receptive field
  - Residual connections: stable gradient flow for deep networks
  - Parallel computation: faster than RNNs at inference time

Reference: Bai et al. (2018) https://arxiv.org/abs/1803.01271
"""
import math
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import structlog

from config import get_settings
from models.base import BaseModel, ModelMetrics

settings = get_settings()
logger = structlog.get_logger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── TCN building blocks ──────────────────────────────────────────────────────
class CausalConv1d(nn.Module):
    """Causal 1-D convolution: output at time t only depends on inputs ≤ t."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        # Remove right-side padding to maintain causal property
        return out[:, :, :-self.padding] if self.padding > 0 else out


class TCNResidualBlock(nn.Module):
    """
    Residual block with two dilated causal convolutions + skip connection.
    Architecture: Conv → WeightNorm → ReLU → Dropout → Conv → WeightNorm → ReLU → Dropout
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.utils.weight_norm(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        )
        self.conv2 = nn.utils.weight_norm(
            CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        )
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # 1×1 conv for skip connection if channel dims differ
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.relu(self.conv2(out))
        out = self.dropout(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(out + residual)


class TCNModel(nn.Module):
    """
    Full Temporal Convolutional Network.

    Stack of residual blocks with exponentially growing dilations:
      dilation = 2^0, 2^1, 2^2, ..., 2^(n_layers-1)
    Receptive field = 2 * kernel_size * (2^n_layers - 1)
    """

    def __init__(
        self,
        n_features: int,
        n_channels: int = 64,
        n_layers: int = 8,
        kernel_size: int = 3,
        dropout: float = 0.2,
        n_outputs: int = 1,
        predict_quantiles: bool = True,
    ):
        super().__init__()
        self.predict_quantiles = predict_quantiles
        n_quantiles = 3 if predict_quantiles else 1  # [q10, q50, q90]
        out_size = n_outputs * n_quantiles

        layers = []
        for i in range(n_layers):
            in_ch = n_features if i == 0 else n_channels
            dilation = 2 ** i
            layers.append(
                TCNResidualBlock(in_ch, n_channels, kernel_size, dilation, dropout)
            )
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(n_channels, out_size)

        receptive_field = 2 * kernel_size * (2 ** n_layers - 1)
        logger.info("TCN built", receptive_field=receptive_field, n_params=self._count_params())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, features, seq_len) — channels-first for Conv1d
        out = self.network(x)          # (batch, n_channels, seq_len)
        out = out[:, :, -1]            # take last timestep
        return self.output_layer(out)  # (batch, n_outputs * n_quantiles)

    def _count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Dataset ──────────────────────────────────────────────────────────────────
class IntradayDataset(Dataset):
    """
    Sliding-window dataset for intraday minute-bar sequences.

    Each sample:
      X: (n_features, lookback) — feature matrix
      y: scalar — next-bar log return
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "log_return",
        lookback: int = 120,  # 120 bars = 2 hours of 1-min data
    ):
        self.data = df[feature_cols].values.astype(np.float32)
        self.targets = df[target_col].values.astype(np.float32)
        self.lookback = lookback
        self.feature_cols = feature_cols

    def __len__(self) -> int:
        return max(0, len(self.data) - self.lookback)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx: idx + self.lookback]            # (lookback, features)
        x = torch.from_numpy(x).T                          # (features, lookback)
        y = torch.tensor(self.targets[idx + self.lookback], dtype=torch.float32)
        return x, y


# ─── TCN Forecaster ───────────────────────────────────────────────────────────
class TCNForecaster(BaseModel):
    """
    Intraday TCN forecaster for 1-min bar data.
    Predicts next-bar log return with quantile uncertainty.
    """

    MODEL_NAME = "TCN_Intraday"

    # Standard intraday features (order book + technical + microstructure)
    FEATURE_COLS = [
        "log_return", "log_volume", "bid_ask_spread", "order_imbalance",
        "rsi_14", "macd_hist", "bb_pct_b", "vwap_deviation",
        "return_5bar", "return_20bar", "realised_vol_20bar",
        "trade_count_1min", "buy_trade_ratio",
    ]

    def __init__(
        self,
        n_channels: int = 64,
        n_layers: int = 8,
        kernel_size: int = 3,
        dropout: float = 0.2,
        lookback: int = 120,
        learning_rate: float = 1e-3,
        max_epochs: int = 50,
        batch_size: int = 256,
        patience: int = 5,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.lookback = lookback
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self._model: Optional[TCNModel] = None
        self._feature_cols: List[str] = self.FEATURE_COLS
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std: Optional[np.ndarray] = None

    def _build_model(self) -> TCNModel:
        return TCNModel(
            n_features=len(self._feature_cols),
            n_channels=self.n_channels,
            n_layers=self.n_layers,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            predict_quantiles=True,
        ).to(DEVICE)

    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Z-score normalise feature columns; fit on training data."""
        df = df.copy()
        avail = [c for c in self._feature_cols if c in df.columns]
        self._feature_cols = avail
        if self._scaler_mean is None:
            self._scaler_mean = df[avail].mean().values
            self._scaler_std = df[avail].std().values.clip(min=1e-8)
        df[avail] = (df[avail].values - self._scaler_mean) / self._scaler_std
        return df

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare intraday OHLCV + order book DataFrame.
        Adds log_return, log_volume, and lookback-based features.
        """
        df = df.copy().sort_values("timestamp").reset_index(drop=True)
        df["log_return"] = np.log(df["close"] / df["close"].shift(1)).fillna(0)
        df["log_volume"] = np.log1p(df["volume"].clip(lower=0))

        # Lookback aggregated features
        for w in [5, 20]:
            df[f"return_{w}bar"] = df["log_return"].rolling(w).sum().fillna(0)
        df["realised_vol_20bar"] = df["log_return"].rolling(20).std().fillna(0)

        return df.dropna().reset_index(drop=True)

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        experiment_name: Optional[str] = None,
    ) -> ModelMetrics:
        from datetime import datetime as dt
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name or settings.mlflow_experiment_name)

        train_df = self._normalise(self.prepare_data(train_df))

        with mlflow.start_run(run_name=f"TCN_{dt.now():%Y%m%d_%H%M}") as run:
            mlflow.log_params({
                "model": self.MODEL_NAME,
                "n_channels": self.n_channels,
                "n_layers": self.n_layers,
                "kernel_size": self.kernel_size,
                "lookback": self.lookback,
                "learning_rate": self.learning_rate,
            })

            self._model = self._build_model()
            optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.learning_rate)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.max_epochs
            )

            train_ds = IntradayDataset(train_df, self._feature_cols, lookback=self.lookback)
            train_loader = DataLoader(
                train_ds, batch_size=self.batch_size, shuffle=True,
                num_workers=0, pin_memory=(DEVICE.type == "cuda"),
            )

            # Pinball (quantile) loss for uncertainty estimation
            quantile_levels = torch.tensor([0.1, 0.5, 0.9], device=DEVICE)

            best_val_loss = float("inf")
            patience_count = 0
            best_state = None

            for epoch in range(self.max_epochs):
                self._model.train()
                epoch_loss = 0.0

                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(DEVICE)
                    y_batch = y_batch.to(DEVICE)

                    optimizer.zero_grad()
                    preds = self._model(X_batch)  # (batch, 3) — three quantiles

                    # Pinball loss for each quantile
                    loss = torch.tensor(0.0, device=DEVICE)
                    for i, q in enumerate(quantile_levels):
                        err = y_batch - preds[:, i]
                        loss += torch.mean(
                            torch.where(err >= 0, q * err, (q - 1) * err)
                        )
                    loss /= len(quantile_levels)

                    loss.backward()
                    nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_loss += loss.item()

                avg_loss = epoch_loss / max(len(train_loader), 1)
                scheduler.step()

                # Validation
                val_loss = avg_loss
                if val_df is not None:
                    val_df_norm = self._normalise(self.prepare_data(val_df))
                    val_ds = IntradayDataset(val_df_norm, self._feature_cols, lookback=self.lookback)
                    val_loader = DataLoader(val_ds, batch_size=self.batch_size * 2)
                    val_loss = self._eval_loss(val_loader, quantile_levels)

                mlflow.log_metrics({"train_loss": avg_loss, "val_loss": val_loss}, step=epoch)
                logger.info("TCN epoch", epoch=epoch + 1, train_loss=round(avg_loss, 6),
                            val_loss=round(val_loss, 6))

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= self.patience:
                        logger.info("Early stopping", epoch=epoch + 1)
                        break

            # Restore best weights
            if best_state:
                self._model.load_state_dict(best_state)

            # Save to MLflow
            import tempfile, pathlib, torch
            with tempfile.TemporaryDirectory() as tmp:
                model_path = pathlib.Path(tmp) / "tcn_model.pt"
                torch.save({
                    "state_dict": self._model.state_dict(),
                    "config": {
                        "n_features": len(self._feature_cols),
                        "n_channels": self.n_channels,
                        "n_layers": self.n_layers,
                        "kernel_size": self.kernel_size,
                        "dropout": self.dropout,
                    },
                    "feature_cols": self._feature_cols,
                    "scaler_mean": self._scaler_mean.tolist(),
                    "scaler_std": self._scaler_std.tolist(),
                }, str(model_path))
                mlflow.log_artifact(str(model_path), artifact_path="model")

            self._run_id = run.info.run_id
            return ModelMetrics(
                model_name=self.MODEL_NAME,
                run_id=run.info.run_id,
                extra={"best_val_loss": best_val_loss},
            )

    def _eval_loss(self, loader: DataLoader, quantile_levels: torch.Tensor) -> float:
        self._model.eval()
        total = 0.0
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                preds = self._model(X)
                loss = torch.tensor(0.0, device=DEVICE)
                for i, q in enumerate(quantile_levels):
                    err = y - preds[:, i]
                    loss += torch.mean(torch.where(err >= 0, q * err, (q - 1) * err))
                total += (loss / len(quantile_levels)).item()
        return total / max(len(loader), 1)

    def predict(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generate per-bar quantile forecasts.
        Returns DataFrame with columns: [timestamp, q10, q50, q90]
        """
        if self._model is None:
            raise RuntimeError("Model not loaded.")

        df_prep = self._normalise(self.prepare_data(df))
        ds = IntradayDataset(df_prep, self._feature_cols, lookback=self.lookback)
        loader = DataLoader(ds, batch_size=512)

        all_preds = []
        self._model.eval()
        with torch.no_grad():
            for X, _ in loader:
                preds = self._model(X.to(DEVICE)).cpu().numpy()
                all_preds.append(preds)

        if not all_preds:
            return pd.DataFrame()

        preds_arr = np.concatenate(all_preds, axis=0)
        result = df_prep.iloc[self.lookback:].copy()
        result["q10"] = preds_arr[:, 0]
        result["q50"] = preds_arr[:, 1]
        result["q90"] = preds_arr[:, 2]
        return result[["timestamp", "q10", "q50", "q90"]] if "timestamp" in result.columns else result[["q10", "q50", "q90"]]

    @classmethod
    def load(cls, mlflow_run_id: str) -> "TCNForecaster":
        import tempfile, pathlib
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        artifact_uri = mlflow.get_run(mlflow_run_id).info.artifact_uri
        model_file = f"{artifact_uri}/model/tcn_model.pt"

        checkpoint = torch.load(model_file, map_location=DEVICE)
        instance = cls()
        instance._feature_cols = checkpoint["feature_cols"]
        instance._scaler_mean = np.array(checkpoint["scaler_mean"])
        instance._scaler_std = np.array(checkpoint["scaler_std"])
        instance._model = TCNModel(**checkpoint["config"]).to(DEVICE)
        instance._model.load_state_dict(checkpoint["state_dict"])
        instance._model.eval()
        instance._run_id = mlflow_run_id
        return instance
