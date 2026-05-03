"""
models/tft.py — Temporal Fusion Transformer for multi-horizon stock forecasting.

Uses the NeuralForecast library which provides a production-ready TFT
implementation. This wrapper adds:
  - MLflow experiment tracking
  - Walk-forward cross-validation
  - Quantile (uncertainty) predictions
  - Serialisation / loading from MLflow model registry

Reference:
  Lim et al. (2021) — Temporal Fusion Transformers for Interpretable
  Multi-horizon Time Series Forecasting.
  https://arxiv.org/abs/1912.09363
"""
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import structlog
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import HuberMQLoss
from neuralforecast.models import TFT

from config import get_settings
from models.base import BaseModel, ModelMetrics

settings = get_settings()
logger = structlog.get_logger(__name__)


class TFTForecaster(BaseModel):
    """
    Multi-horizon price-return forecaster based on Temporal Fusion Transformer.

    Predicts multiple quantiles (10th, 50th, 90th percentile) for each
    forecast horizon (1d, 3d, 5d, 10d, 20d), enabling uncertainty-aware
    position sizing.
    """

    MODEL_NAME = "TFT"
    DEFAULT_HORIZONS = [1, 3, 5, 10, 20]  # trading days
    QUANTILE_LEVELS = [0.1, 0.5, 0.9]

    def __init__(
        self,
        horizons: Optional[List[int]] = None,
        input_size: int = 63,  # 63 trading days (~3 months) lookback
        hidden_size: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 3e-4,
        max_steps: int = 1000,
        batch_size: int = 32,
    ):
        super().__init__()
        self.horizons = horizons or self.DEFAULT_HORIZONS
        self.max_horizon = max(self.horizons)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.batch_size = batch_size
        self._nf: Optional[NeuralForecast] = None

    def _build_model(self) -> NeuralForecast:
        """Instantiate the NeuralForecast TFT pipeline."""
        tft = TFT(
            h=self.max_horizon,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            attn_dropout=self.dropout,
            dropout=self.dropout,
            n_head=self.num_heads,
            learning_rate=self.learning_rate,
            loss=HuberMQLoss(quantiles=self.QUANTILE_LEVELS),
            max_steps=self.max_steps,
            batch_size=self.batch_size,
            # Covariate columns (known future: calendar features)
            futr_exog_list=["day_of_week", "month", "quarter", "is_month_end"],
            # Historical covariates (past-only: technicals, volume)
            hist_exog_list=[
                "return_1d", "return_5d", "return_20d",
                "realised_vol_20d", "volume_ratio_20d",
                "rsi_14", "macd_hist", "bb_pct_b", "adx_14",
                "order_imbalance",
            ],
            # Static covariates per series (stock metadata)
            stat_exog_list=["sector_code", "exchange_code"],
            val_check_steps=50,
            early_stop_patience_steps=5,
            scaler_type="standard",
        )
        return NeuralForecast(models=[tft], freq="B")  # B = business-day frequency

    def prepare_training_data(
        self, panel_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Convert internal OHLCV+features DataFrame to NeuralForecast long format.

        NeuralForecast expects columns: [unique_id, ds, y, *covariates]
          unique_id: ticker symbol
          ds: date (datetime)
          y: target variable (we use log-return, predicting forward returns)

        panel_df must have: [ticker, date, close, volume, *feature_cols]
        """
        df = panel_df.copy()
        df = df.sort_values(["ticker", "date"])

        # Target: next-day log return (shifted by 1)
        df["y"] = df.groupby("ticker")["close"].transform(
            lambda x: np.log(x / x.shift(1))
        )

        # NeuralForecast long format
        df = df.rename(columns={"ticker": "unique_id", "date": "ds"})
        df["ds"] = pd.to_datetime(df["ds"])

        # Encode categoricals as integers
        sector_map = {s: i for i, s in enumerate(df["sector"].unique())}
        exchange_map = {"HOSE": 0, "HNX": 1, "UPCOM": 2}
        df["sector_code"] = df["sector"].map(sector_map).fillna(-1).astype(int)
        df["exchange_code"] = df["exchange"].map(exchange_map).fillna(0).astype(int)

        return df.dropna(subset=["y"])

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        experiment_name: Optional[str] = None,
    ) -> ModelMetrics:
        """
        Train TFT with MLflow tracking.

        Args:
            train_df: NeuralForecast-format training data
            val_df: Optional validation split
            experiment_name: MLflow experiment name (overrides config)

        Returns:
            ModelMetrics with MAE, RMSE, directional accuracy
        """
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        exp_name = experiment_name or settings.mlflow_experiment_name
        mlflow.set_experiment(exp_name)

        with mlflow.start_run(run_name=f"TFT_{datetime.now():%Y%m%d_%H%M}") as run:
            # Log hyperparameters
            mlflow.log_params({
                "model": self.MODEL_NAME,
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "learning_rate": self.learning_rate,
                "max_steps": self.max_steps,
                "horizons": str(self.horizons),
                "n_unique_ids": train_df["unique_id"].nunique(),
                "n_rows": len(train_df),
            })

            logger.info("Training TFT", tickers=train_df["unique_id"].nunique(),
                        rows=len(train_df))
            self._nf = self._build_model()
            self._nf.fit(df=train_df, val_size=0 if val_df is None else len(val_df))

            # Evaluate on validation set
            metrics = {}
            if val_df is not None:
                metrics = self._evaluate(val_df)
                mlflow.log_metrics(metrics)
                logger.info("TFT validation metrics", **metrics)

            # Save model artifact to MLflow
            model_path = Path(f"/tmp/tft_{run.info.run_id}")
            self._nf.save(str(model_path), overwrite=True)
            mlflow.log_artifacts(str(model_path), artifact_path="model")

            self._run_id = run.info.run_id
            logger.info("TFT training complete", run_id=run.info.run_id)
            return ModelMetrics(**metrics, run_id=run.info.run_id, model_name=self.MODEL_NAME)

    def predict(
        self,
        df: pd.DataFrame,
        futr_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate multi-horizon quantile forecasts.

        Returns DataFrame with columns:
          [unique_id, ds, TFT-q0.1-hN, TFT-q0.5-hN, TFT-q0.9-hN]
        for each horizon N.
        """
        if self._nf is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        forecasts = self._nf.predict(futr_df=futr_df)
        return forecasts

    def _evaluate(self, val_df: pd.DataFrame) -> Dict[str, float]:
        """Compute MAE, RMSE, directional accuracy on validation data."""
        preds = self.predict(val_df)
        # Use median (q0.5) quantile as point estimate
        pred_col = [c for c in preds.columns if "q0.5" in c and "h1" in c]
        if not pred_col:
            return {}

        merged = val_df.merge(
            preds[["unique_id", "ds", pred_col[0]]].rename(
                columns={pred_col[0]: "y_hat"}
            ),
            on=["unique_id", "ds"],
            how="inner",
        )
        if merged.empty:
            return {}

        y = merged["y"].values
        y_hat = merged["y_hat"].values
        mae = float(np.mean(np.abs(y - y_hat)))
        rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
        dir_acc = float(np.mean(np.sign(y) == np.sign(y_hat)))

        return {"mae": mae, "rmse": rmse, "directional_accuracy": dir_acc}

    def walk_forward_validate(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        test_size: int = 20,  # trading days per split
    ) -> List[Dict]:
        """
        Purged walk-forward cross-validation to prevent lookahead bias.
        Purge gap = max_horizon days between train and test to avoid leakage.
        """
        results = []
        all_dates = sorted(df["ds"].unique())
        split_size = (len(all_dates) - test_size * n_splits) // n_splits

        for split_idx in range(n_splits):
            # Compute train/gap/test date ranges
            train_end_idx = split_size * (split_idx + 1) + test_size * split_idx
            gap_end_idx = train_end_idx + self.max_horizon  # purge gap
            test_end_idx = gap_end_idx + test_size

            if test_end_idx > len(all_dates):
                break

            train_dates = all_dates[:train_end_idx]
            test_dates = all_dates[gap_end_idx:test_end_idx]

            train_split = df[df["ds"].isin(train_dates)].copy()
            test_split = df[df["ds"].isin(test_dates)].copy()

            logger.info(
                "Walk-forward split",
                split=split_idx + 1,
                train_end=str(train_dates[-1])[:10],
                test_start=str(test_dates[0])[:10],
            )

            # Re-train on each split
            self._nf = self._build_model()
            self._nf.fit(df=train_split)
            metrics = self._evaluate(test_split)
            metrics["split"] = split_idx + 1
            results.append(metrics)

        return results

    @classmethod
    def load(cls, mlflow_run_id: str) -> "TFTForecaster":
        """Load a trained TFT model from MLflow artifact store."""
        instance = cls()
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        artifact_uri = mlflow.get_run(mlflow_run_id).info.artifact_uri
        model_uri = f"{artifact_uri}/model"
        instance._nf = NeuralForecast.load(model_uri)
        instance._run_id = mlflow_run_id
        logger.info("TFT model loaded", run_id=mlflow_run_id)
        return instance
