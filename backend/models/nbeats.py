"""
models/nbeats.py — N-BEATS and N-HiTS multi-horizon forecasters.

N-BEATS (Neural Basis Expansion Analysis for Time Series)
  - Pure time-series, no covariates needed
  - Interpretable: trend + seasonality stacks
  - Ref: Oreshkin et al. (2020) https://arxiv.org/abs/1905.10437

N-HiTS (Neural Hierarchical Interpolation for Time Series)
  - Improved upon N-BEATS with hierarchical interpolation
  - More efficient on long horizons
  - Ref: Challu et al. (2023) https://arxiv.org/abs/2201.12886
"""
from datetime import datetime
from typing import Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd
import structlog
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import HuberMQLoss, MQLoss
from neuralforecast.models import NBEATS, NBEATSx, NHiTS

from config import get_settings
from models.base import BaseModel, ModelMetrics

settings = get_settings()
logger = structlog.get_logger(__name__)


class NBEATSForecaster(BaseModel):
    """
    N-BEATS / N-HiTS ensemble forecaster.

    Trains both N-BEATS and N-HiTS, then ensembles predictions
    by averaging quantile outputs — improving calibration.
    """

    MODEL_NAME = "NBEATS_NHiTS_Ensemble"
    QUANTILE_LEVELS = [0.1, 0.5, 0.9]

    def __init__(
        self,
        max_horizon: int = 20,
        input_size: int = 63,
        nbeats_stacks: int = 30,
        nhits_n_freq_downsample: Optional[List[int]] = None,
        learning_rate: float = 1e-3,
        max_steps: int = 800,
        batch_size: int = 32,
        use_ensemble: bool = True,
    ):
        super().__init__()
        self.max_horizon = max_horizon
        self.input_size = input_size
        self.nbeats_stacks = nbeats_stacks
        self.nhits_n_freq_downsample = nhits_n_freq_downsample or [2, 1, 1]
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.use_ensemble = use_ensemble
        self._nf: Optional[NeuralForecast] = None

    def _build_models(self) -> List:
        """Build N-BEATS and N-HiTS model instances."""
        loss = HuberMQLoss(quantiles=self.QUANTILE_LEVELS)

        nbeats = NBEATS(
            h=self.max_horizon,
            input_size=self.input_size,
            loss=loss,
            n_blocks=[self.nbeats_stacks // 3] * 3,   # trend + seasonality + generic stacks
            mlp_units=[[256, 256]] * 3,
            learning_rate=self.learning_rate,
            max_steps=self.max_steps,
            batch_size=self.batch_size,
            scaler_type="standard",
            val_check_steps=50,
            early_stop_patience_steps=5,
        )

        nhits = NHiTS(
            h=self.max_horizon,
            input_size=self.input_size,
            loss=loss,
            n_freq_downsample=self.nhits_n_freq_downsample,
            learning_rate=self.learning_rate,
            max_steps=self.max_steps,
            batch_size=self.batch_size,
            scaler_type="standard",
            val_check_steps=50,
            early_stop_patience_steps=5,
        )

        return [nbeats, nhits] if self.use_ensemble else [nbeats]

    def prepare_training_data(self, panel_df: pd.DataFrame) -> pd.DataFrame:
        """
        N-BEATS only uses the target series (no covariates).
        Returns minimal NeuralForecast format: [unique_id, ds, y]
        """
        df = panel_df.copy().sort_values(["ticker", "date"])
        df["y"] = df.groupby("ticker")["close"].transform(
            lambda x: np.log(x / x.shift(1))
        )
        df = df.rename(columns={"ticker": "unique_id", "date": "ds"})
        df["ds"] = pd.to_datetime(df["ds"])
        return df[["unique_id", "ds", "y"]].dropna(subset=["y"])

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        experiment_name: Optional[str] = None,
    ) -> ModelMetrics:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name or settings.mlflow_experiment_name)

        with mlflow.start_run(
            run_name=f"{self.MODEL_NAME}_{datetime.now():%Y%m%d_%H%M}"
        ) as run:
            mlflow.log_params({
                "model": self.MODEL_NAME,
                "max_horizon": self.max_horizon,
                "input_size": self.input_size,
                "use_ensemble": self.use_ensemble,
                "learning_rate": self.learning_rate,
                "max_steps": self.max_steps,
            })

            logger.info("Training N-BEATS/N-HiTS", tickers=train_df["unique_id"].nunique())
            self._nf = NeuralForecast(models=self._build_models(), freq="B")
            self._nf.fit(df=train_df, val_size=0 if val_df is None else len(val_df))

            metrics = {}
            if val_df is not None:
                metrics = self._evaluate(val_df)
                mlflow.log_metrics(metrics)
                logger.info("N-BEATS/N-HiTS validation", **metrics)

            # Persist model
            import tempfile, pathlib
            with tempfile.TemporaryDirectory() as tmp:
                model_path = pathlib.Path(tmp) / "model"
                self._nf.save(str(model_path), overwrite=True)
                mlflow.log_artifacts(str(model_path), artifact_path="model")

            self._run_id = run.info.run_id
            return ModelMetrics(**metrics, run_id=run.info.run_id, model_name=self.MODEL_NAME)

    def predict(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generate quantile forecasts.
        For ensemble, each model's predictions are averaged.
        """
        if self._nf is None:
            raise RuntimeError("Model not loaded. Call train() or load() first.")
        forecasts = self._nf.predict(df=df)

        if self.use_ensemble:
            forecasts = self._ensemble_predictions(forecasts)
        return forecasts

    def _ensemble_predictions(self, forecasts: pd.DataFrame) -> pd.DataFrame:
        """
        Average NBEATS and NHiTS quantile predictions column-by-column.
        Creates unified column names: Ensemble-q{q}-h{h}
        """
        result = forecasts[["unique_id", "ds"]].copy()

        for quantile in self.QUANTILE_LEVELS:
            q_str = f"q{int(quantile * 100)}"
            for horizon in range(1, self.max_horizon + 1):
                nbeats_col = f"NBEATS-{q_str}-h{horizon}"
                nhits_col = f"NHiTS-{q_str}-h{horizon}"
                out_col = f"Ensemble-{q_str}-h{horizon}"

                available = [c for c in [nbeats_col, nhits_col] if c in forecasts.columns]
                if available:
                    result[out_col] = forecasts[available].mean(axis=1)

        return result

    def _evaluate(self, val_df: pd.DataFrame) -> Dict[str, float]:
        preds = self.predict(val_df)
        # Use median quantile at horizon 1 for evaluation
        median_col = next(
            (c for c in preds.columns if ("q50" in c or "q0.5" in c) and "h1" in c),
            None,
        )
        if not median_col:
            return {}

        merged = val_df.merge(
            preds[["unique_id", "ds", median_col]].rename(columns={median_col: "y_hat"}),
            on=["unique_id", "ds"],
            how="inner",
        )
        if merged.empty:
            return {}

        y, y_hat = merged["y"].values, merged["y_hat"].values
        return {
            "mae": float(np.mean(np.abs(y - y_hat))),
            "rmse": float(np.sqrt(np.mean((y - y_hat) ** 2))),
            "directional_accuracy": float(np.mean(np.sign(y) == np.sign(y_hat))),
        }

    @classmethod
    def load(cls, mlflow_run_id: str) -> "NBEATSForecaster":
        instance = cls()
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        artifact_uri = mlflow.get_run(mlflow_run_id).info.artifact_uri
        instance._nf = NeuralForecast.load(f"{artifact_uri}/model")
        instance._run_id = mlflow_run_id
        return instance
