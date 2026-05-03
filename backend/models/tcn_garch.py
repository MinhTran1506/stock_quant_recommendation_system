"""
models/tcn_garch.py — Short-horizon intraday models.

TCN  (Temporal Convolutional Network): intraday pattern recognition
     using dilated causal convolutions — captures multi-scale temporal
     dependencies without recurrence (faster than LSTM, parallelisable).

ARIMA/GARCH: classical statistical baseline.
     - ARIMA: mean return prediction (trend + autocorrelation)
     - GARCH(1,1): volatility forecasting (heteroskedasticity-aware)

Both are used together:
  signal = TCN direction  +  GARCH vol-scaled position size
"""
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import structlog
import torch
import torch.nn as nn
import torch.optim as optim
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from torch.utils.data import DataLoader, Dataset

from config import get_settings
from models.base import BaseModel, ModelMetrics

warnings.filterwarnings("ignore", category=UserWarning)
settings = get_settings()
logger = structlog.get_logger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── TCN Building Blocks ──────────────────────────────────────────────────────
class CausalConv1d(nn.Module):
    """Causal (masked) 1-D convolution — no future leakage."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return out[:, :, : -self.padding] if self.padding else out


class TCNResidualBlock(nn.Module):
    """Residual block with two causal dilated convolutions + dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        # 1×1 conv for channel mismatch in residual path
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq_len)
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.norm1(self.conv1(x).transpose(1, 2)).transpose(1, 2))
        out = self.dropout(out)
        out = self.relu(self.norm2(self.conv2(out).transpose(1, 2)).transpose(1, 2))
        out = self.dropout(out)
        return self.relu(out + residual)


class TCNModel(nn.Module):
    """
    Temporal Convolutional Network for intraday return prediction.

    Architecture:
      - Stack of dilated residual blocks (dilation doubles each layer)
      - Global average pooling → FC head
      - Output: directional probability (up/down) + magnitude estimate
    """

    def __init__(
        self,
        n_features: int,
        n_channels: int = 64,
        n_layers: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.2,
        output_size: int = 2,  # [direction_prob, magnitude]
    ):
        super().__init__()
        layers = []
        in_ch = n_features
        for i in range(n_layers):
            dilation = 2 ** i
            layers.append(TCNResidualBlock(in_ch, n_channels, kernel_size, dilation, dropout))
            in_ch = n_channels
        self.tcn = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(n_channels, n_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_channels // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features) → (batch, n_features, seq_len)
        x = x.transpose(1, 2)
        out = self.tcn(x)          # (batch, n_channels, seq_len)
        out = self.global_pool(out).squeeze(-1)  # (batch, n_channels)
        return self.head(out)       # (batch, output_size)


# ─── Dataset ──────────────────────────────────────────────────────────────────
class IntradayDataset(Dataset):
    """
    Sliding-window dataset over intraday OHLCV + microstructure features.
    Target: sign of next-bar return (classification).
    """

    def __init__(self, features: np.ndarray, returns: np.ndarray, window: int = 60):
        self.features = features.astype(np.float32)
        self.returns = returns.astype(np.float32)
        self.window = window

    def __len__(self) -> int:
        return max(0, len(self.features) - self.window)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.features[idx: idx + self.window])
        y_raw = self.returns[idx + self.window]
        y = torch.tensor([1.0 if y_raw > 0 else 0.0])  # binary direction
        return x, y


# ─── TCN Forecaster ──────────────────────────────────────────────────────────
class TCNForecaster(BaseModel):
    """
    TCN-based intraday forecaster.
    Predicts next-bar directional probability (up/down).
    """

    MODEL_NAME = "TCN"

    def __init__(
        self,
        n_features: int = 20,
        window: int = 60,           # 60 1-min bars = 1 hour lookback
        n_channels: int = 64,
        n_layers: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 256,
    ):
        super().__init__()
        self.n_features = n_features
        self.window = window
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self._net: Optional[TCNModel] = None
        self._model_kwargs = dict(
            n_features=n_features, n_channels=n_channels,
            n_layers=n_layers, kernel_size=kernel_size, dropout=dropout,
        )

    def _build(self) -> TCNModel:
        return TCNModel(**self._model_kwargs).to(DEVICE)

    def train(
        self,
        train_df: pd.DataFrame,       # columns: feature cols + 'return'
        val_df: Optional[pd.DataFrame] = None,
        experiment_name: Optional[str] = None,
    ) -> ModelMetrics:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name or settings.mlflow_experiment_name)

        feature_cols = [c for c in train_df.columns if c != "return"]
        X_train = train_df[feature_cols].values
        y_train = train_df["return"].values

        train_ds = IntradayDataset(X_train, y_train, self.window)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,
                                  num_workers=0, pin_memory=DEVICE.type == "cuda")

        self._net = self._build()
        optimizer = optim.Adam(self._net.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        with mlflow.start_run(run_name=f"TCN_{datetime.now():%Y%m%d_%H%M}") as run:
            mlflow.log_params({
                "model": self.MODEL_NAME, "window": self.window,
                "epochs": self.epochs, "batch_size": self.batch_size,
                **self._model_kwargs,
            })

            best_val_acc = 0.0
            for epoch in range(self.epochs):
                self._net.train()
                total_loss = 0.0
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(DEVICE)
                    y_batch = y_batch.to(DEVICE)
                    optimizer.zero_grad()
                    preds = self._net(X_batch)[:, 0:1]
                    loss = criterion(preds, y_batch)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item()
                scheduler.step()

                avg_loss = total_loss / max(len(train_loader), 1)
                mlflow.log_metric("train_loss", avg_loss, step=epoch)

                if val_df is not None and epoch % 10 == 0:
                    val_acc = self._eval_accuracy(val_df, feature_cols)
                    mlflow.log_metric("val_accuracy", val_acc, step=epoch)
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc

            # Save model
            import tempfile, os
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "tcn.pt")
                torch.save(self._net.state_dict(), path)
                mlflow.log_artifact(path, artifact_path="model")

            self._run_id = run.info.run_id
            return ModelMetrics(
                model_name=self.MODEL_NAME,
                run_id=run.info.run_id,
                directional_accuracy=best_val_acc,
            )

    def _eval_accuracy(self, df: pd.DataFrame, feature_cols: List[str]) -> float:
        feature_cols = [c for c in feature_cols if c in df.columns]
        X = df[feature_cols].values.astype(np.float32)
        y = df["return"].values
        ds = IntradayDataset(X, y, self.window)
        loader = DataLoader(ds, batch_size=512, shuffle=False)
        self._net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                preds = torch.sigmoid(self._net(X_batch.to(DEVICE))[:, 0])
                pred_dir = (preds > 0.5).float().cpu()
                correct += (pred_dir == y_batch.squeeze()).sum().item()
                total += len(y_batch)
        return correct / max(total, 1)

    def predict(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if self._net is None:
            raise RuntimeError("Model not loaded")
        feature_cols = [c for c in df.columns if c != "return"]
        X = torch.tensor(df[feature_cols].values[-self.window:].astype(np.float32))
        X = X.unsqueeze(0).to(DEVICE)
        self._net.eval()
        with torch.no_grad():
            out = self._net(X)
            prob_up = float(torch.sigmoid(out[0, 0]))
        result = df.tail(1).copy()
        result["direction_prob"] = prob_up
        result["signal"] = 1 if prob_up > 0.55 else (-1 if prob_up < 0.45 else 0)
        return result

    @classmethod
    def load(cls, mlflow_run_id: str) -> "TCNForecaster":
        import tempfile, os
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        artifact_uri = mlflow.get_run(mlflow_run_id).info.artifact_uri
        instance = cls()
        instance._net = instance._build()
        with tempfile.TemporaryDirectory() as tmp:
            local = mlflow.artifacts.download_artifacts(
                f"{artifact_uri}/model/tcn.pt", dst_path=tmp
            )
            instance._net.load_state_dict(torch.load(local, map_location=DEVICE))
        instance._run_id = mlflow_run_id
        return instance


# ─── ARIMA + GARCH ────────────────────────────────────────────────────────────
class ARIMAGARCHModel:
    """
    Classical ARIMA(p,d,q) + GARCH(1,1) baseline.

    Usage:
      model = ARIMAGARCHModel()
      result = model.fit_predict(returns_series, horizon=5)
      # result: {"mean_forecast": [...], "vol_forecast": [...], "95ci": [...]}

    This model is intentionally kept simple — used as a benchmark
    and for volatility scaling of other model signals.
    """

    def __init__(self, order: Tuple[int, int, int] = (2, 0, 2)):
        self.order = order   # ARIMA (p, d, q)
        self._arima_result = None
        self._garch_result = None

    def fit(self, returns: pd.Series) -> "ARIMAGARCHModel":
        """Fit ARIMA on returns, GARCH on ARIMA residuals."""
        returns_clean = returns.dropna().astype(float)

        # ARIMA for conditional mean
        try:
            arima = ARIMA(returns_clean, order=self.order)
            self._arima_result = arima.fit(disp=False)
        except Exception as e:
            logger.warning("ARIMA fit failed, using AR(1)", error=str(e))
            arima = ARIMA(returns_clean, order=(1, 0, 0))
            self._arima_result = arima.fit(disp=False)

        # GARCH(1,1) on standardised residuals for volatility
        residuals = self._arima_result.resid.dropna() * 100  # scale to percent
        try:
            garch = arch_model(residuals, vol="Garch", p=1, q=1, mean="Zero", dist="normal")
            self._garch_result = garch.fit(disp="off", show_warning=False)
        except Exception as e:
            logger.warning("GARCH fit failed", error=str(e))
            self._garch_result = None

        return self

    def predict(self, horizon: int = 5) -> Dict:
        """
        Generate horizon-step ahead forecasts.
        Returns mean forecast, volatility forecast, and 95% confidence interval.
        """
        if self._arima_result is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        forecast = self._arima_result.get_forecast(steps=horizon)
        mean_fc = forecast.predicted_mean.values / 100  # back to decimal
        conf_int = forecast.conf_int(alpha=0.05).values / 100

        vol_fc = None
        if self._garch_result is not None:
            garch_fc = self._garch_result.forecast(horizon=horizon, reindex=False)
            vol_fc = (np.sqrt(garch_fc.variance.values[-1]) / 100).tolist()

        return {
            "mean_forecast": mean_fc.tolist(),
            "vol_forecast": vol_fc,
            "ci_lower": conf_int[:, 0].tolist(),
            "ci_upper": conf_int[:, 1].tolist(),
            "horizon": horizon,
        }

    def fit_predict(self, returns: pd.Series, horizon: int = 5) -> Dict:
        """Convenience: fit then predict in one call."""
        return self.fit(returns).predict(horizon)
