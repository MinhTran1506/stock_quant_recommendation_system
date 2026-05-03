"""
models/meta_model.py — LightGBM Meta/Ranking Model.

Combines outputs from base forecasters (TFT, N-BEATS, ARIMA/GARCH)
with fundamental ratios and sentiment features into a unified stock score
(0–100) used for portfolio construction and ranking.

Design choices:
  - LightGBM for speed, performance, and native feature importance (SHAP)
  - LambdaRank objective for learning-to-rank (cross-sectional ranking)
  - SHAP values exposed via API for explainability UI
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import shap
import structlog
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler

from config import get_settings
from models.base import BaseModel, ModelMetrics

settings = get_settings()
logger = structlog.get_logger(__name__)


# ── Feature columns fed into the meta-model ───────────────────────────────────
TFT_FEATURES = [f"tft_return_h{h}_q{q}" for h in [1, 3, 5, 10, 20]
                for q in ["10", "50", "90"]]
NBEATS_FEATURES = [f"nbeats_return_h{h}_q50" for h in [1, 3, 5, 10, 20]]
TECHNICAL_FEATURES = [
    "return_1d", "return_5d", "return_20d", "return_60d",
    "realised_vol_20d", "realised_vol_60d", "vol_ratio_5_60",
    "rsi_14", "macd_hist", "bb_pct_b", "adx_14",
    "volume_ratio_5d", "volume_ratio_20d", "obv_trend_20d", "vwap_deviation",
    "momentum_12m1m", "pct_from_52w_high", "pct_from_52w_low",
    "ma_cross_5_20", "ma_cross_10_50",
]
FUNDAMENTAL_FEATURES = [
    "pe_ratio", "pb_ratio", "roe", "roa", "debt_to_equity", "dividend_yield",
]
SENTIMENT_FEATURES = [
    "sentiment_score_1d", "sentiment_score_7d", "news_count_7d",
]
ALL_FEATURES = TFT_FEATURES + NBEATS_FEATURES + TECHNICAL_FEATURES + \
               FUNDAMENTAL_FEATURES + SENTIMENT_FEATURES


class MetaRankingModel(BaseModel):
    """
    LightGBM-based cross-sectional ranking model.

    Target: forward 5-day return rank (percentile within universe).
    Loss: LambdaRank (ranking-aware gradient boosting).
    Output: stock score 0–100 (higher = more bullish rank).
    """

    MODEL_NAME = "MetaRankingLGBM"

    def __init__(
        self,
        num_leaves: int = 63,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        feature_fraction: float = 0.8,
        bagging_fraction: float = 0.8,
        bagging_freq: int = 5,
        min_child_samples: int = 20,
        reg_alpha: float = 0.1,
        reg_lambda: float = 0.1,
        random_state: int = 42,
        target_horizon_days: int = 5,
    ):
        super().__init__()
        self.target_horizon_days = target_horizon_days
        self._lgb_params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [3, 5, 10],
            "num_leaves": num_leaves,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "feature_fraction": feature_fraction,
            "bagging_fraction": bagging_fraction,
            "bagging_freq": bagging_freq,
            "min_child_samples": min_child_samples,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }
        self._model: Optional[lgb.Booster] = None
        self._scaler = RobustScaler()
        self._feature_cols: List[str] = []
        self._shap_explainer: Optional[shap.TreeExplainer] = None

    def prepare_training_data(
        self, panel_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Build (X, y_rank, groups) for LambdaRank training.

        X: feature matrix
        y_rank: percentile rank of forward return within each date group (0–100)
        groups: number of stocks per date (required by LambdaRank)
        """
        df = panel_df.copy().sort_values(["date", "ticker"])

        # Compute forward return (target)
        df[f"fwd_return_{self.target_horizon_days}d"] = df.groupby("ticker")["close"].transform(
            lambda x: x.shift(-self.target_horizon_days) / x - 1
        )
        df = df.dropna(subset=[f"fwd_return_{self.target_horizon_days}d"])

        # Cross-sectional rank (within each date) → percentile 0–100
        df["target_rank"] = df.groupby("date")[f"fwd_return_{self.target_horizon_days}d"].transform(
            lambda x: x.rank(pct=True) * 100
        )

        # Available feature columns
        self._feature_cols = [c for c in ALL_FEATURES if c in df.columns]
        if not self._feature_cols:
            raise ValueError("No feature columns found in training DataFrame. "
                             "Run feature pipeline before training meta-model.")

        X = df[self._feature_cols].fillna(0)
        y = df["target_rank"]
        groups = df.groupby("date").size().values  # stocks per date for LambdaRank

        return X, y, groups

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        experiment_name: Optional[str] = None,
    ) -> ModelMetrics:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name or settings.mlflow_experiment_name)

        X_train, y_train, groups_train = self.prepare_training_data(train_df)
        X_train_scaled = self._scaler.fit_transform(X_train)

        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]

        val_data = None
        if val_df is not None:
            X_val, y_val, groups_val = self.prepare_training_data(val_df)
            X_val_scaled = self._scaler.transform(X_val)
            val_data = [(X_val_scaled, y_val)]

        with mlflow.start_run(
            run_name=f"{self.MODEL_NAME}_{datetime.now():%Y%m%d_%H%M}"
        ) as run:
            mlflow.log_params({**self._lgb_params, "n_features": len(self._feature_cols)})
            mlflow.log_params({"feature_cols": json.dumps(self._feature_cols[:20])})

            logger.info("Training LightGBM meta-model", n_features=len(self._feature_cols),
                        n_samples=len(X_train))

            dtrain = lgb.Dataset(X_train_scaled, label=y_train, group=groups_train)

            self._model = lgb.train(
                self._lgb_params,
                dtrain,
                valid_sets=val_data,
                callbacks=callbacks,
            )

            # SHAP explainer for inference
            self._shap_explainer = shap.TreeExplainer(self._model)

            # Feature importance logging
            importance = dict(zip(
                self._feature_cols,
                self._model.feature_importance(importance_type="gain").tolist(),
            ))
            mlflow.log_dict(importance, "feature_importance.json")

            # Evaluate
            metrics = {}
            if val_df is not None:
                y_hat = self._model.predict(X_val_scaled)
                ic = float(pd.Series(y_val.values).corr(pd.Series(y_hat)))
                metrics = {"ic": ic}
                mlflow.log_metrics(metrics)
                logger.info("Meta-model validation IC", ic=ic)

            # Save artifacts
            model_path = f"/tmp/meta_model_{run.info.run_id}.lgb"
            self._model.save_model(model_path)
            mlflow.log_artifact(model_path, artifact_path="model")
            import joblib
            scaler_path = f"/tmp/scaler_{run.info.run_id}.pkl"
            joblib.dump(self._scaler, scaler_path)
            mlflow.log_artifact(scaler_path, artifact_path="model")
            mlflow.log_dict({"feature_cols": self._feature_cols}, "model/feature_cols.json")

            self._run_id = run.info.run_id
            return ModelMetrics(**metrics, run_id=run.info.run_id, model_name=self.MODEL_NAME)

    def predict(
        self,
        df: pd.DataFrame,
        return_shap: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Score stocks in df. Returns df with 'score' column (0–100).
        If return_shap=True, also returns SHAP values for explainability.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded.")

        feature_cols = [c for c in self._feature_cols if c in df.columns]
        X = df[feature_cols].fillna(0)
        X_scaled = self._scaler.transform(X)

        raw_scores = self._model.predict(X_scaled)
        # Normalise to 0–100 cross-sectionally
        min_s, max_s = raw_scores.min(), raw_scores.max()
        if max_s > min_s:
            scores = (raw_scores - min_s) / (max_s - min_s) * 100
        else:
            scores = np.full_like(raw_scores, 50.0)

        result = df[["ticker", "date"]].copy() if "ticker" in df.columns else df.copy()
        result["score"] = scores

        if return_shap and self._shap_explainer is not None:
            shap_values = self._shap_explainer.shap_values(X_scaled)
            # Store top-5 feature importances per row
            top_features = []
            for row_shap in shap_values:
                top_idx = np.argsort(np.abs(row_shap))[::-1][:5]
                top_features.append({
                    feature_cols[i]: float(row_shap[i]) for i in top_idx
                })
            result["feature_importances"] = top_features

        return result

    def get_top_stocks(
        self, df: pd.DataFrame, top_n: int = 20
    ) -> pd.DataFrame:
        """Return top N highest-scored stocks."""
        scored = self.predict(df, return_shap=True)
        return scored.nlargest(top_n, "score").reset_index(drop=True)

    @classmethod
    def load(cls, mlflow_run_id: str) -> "MetaRankingModel":
        import joblib
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        artifact_uri = mlflow.get_run(mlflow_run_id).info.artifact_uri
        model_path = f"{artifact_uri}/model/meta_model_{mlflow_run_id}.lgb"
        scaler_path = f"{artifact_uri}/model/scaler_{mlflow_run_id}.pkl"
        feature_path = f"{artifact_uri}/model/feature_cols.json"

        instance = cls()
        instance._model = lgb.Booster(model_file=model_path)
        instance._scaler = joblib.load(scaler_path)
        with open(feature_path) as f:
            instance._feature_cols = json.load(f)["feature_cols"]
        instance._shap_explainer = shap.TreeExplainer(instance._model)
        instance._run_id = mlflow_run_id
        return instance
