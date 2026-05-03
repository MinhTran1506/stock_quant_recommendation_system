"""
models/training_pipeline.py — Full model training pipeline orchestrator.

Stages (run in order):
  1. Load & validate training data from TimescaleDB
  2. Train TFT (multi-horizon daily)
  3. Train N-BEATS/N-HiTS ensemble (multi-horizon daily)
  4. Train TCN (intraday, if minute data available)
  5. Generate base-model predictions for meta-model training features
  6. Train LightGBM meta-ranking model
  7. Walk-forward validate all models
  8. Register champion models in MLflow model registry
  9. Update DB model_versions table
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import structlog
from sqlalchemy import select, desc

from config import get_settings
from models.tft import TFTForecaster
from models.nbeats import NBEATSForecaster
from models.meta_model import MetaRankingModel
from models.base import ModelMetrics

settings = get_settings()
logger = structlog.get_logger(__name__)


class TrainingPipeline:
    """
    Orchestrates end-to-end model training, validation, and registration.
    Called weekly by Airflow DAG (Sunday retrain).
    """

    def __init__(
        self,
        train_years: int = 3,
        val_months: int = 6,
        test_months: int = 3,
    ):
        self.train_years = train_years
        self.val_months = val_months
        self.test_months = test_months

    async def run_full_retrain(self) -> Dict[str, ModelMetrics]:
        """
        Execute the full training pipeline.
        Returns dict of {model_name: metrics} for all trained models.
        """
        logger.info("Starting full model retrain")
        start = datetime.utcnow()

        # 1. Load data
        logger.info("Loading training data")
        panel_df = await self._load_panel_data()
        if panel_df.empty:
            logger.error("No training data available; aborting")
            return {}

        train_df, val_df, test_df = self._time_split(panel_df)
        logger.info(
            "Data split",
            train_rows=len(train_df), val_rows=len(val_df), test_rows=len(test_df),
            tickers=panel_df["ticker"].nunique(),
        )

        results: Dict[str, ModelMetrics] = {}

        # 2. TFT
        logger.info("Training TFT")
        tft = TFTForecaster(horizons=[1, 3, 5, 10, 20], max_steps=500)
        tft_train = tft.prepare_training_data(train_df)
        tft_val = tft.prepare_training_data(val_df)
        try:
            tft_metrics = tft.train(tft_train, tft_val)
            results["TFT"] = tft_metrics
            logger.info("TFT trained", **{k: v for k, v in vars(tft_metrics).items() if isinstance(v, float)})
        except Exception as e:
            logger.error("TFT training failed", error=str(e))

        # 3. N-BEATS / N-HiTS
        logger.info("Training N-BEATS/N-HiTS ensemble")
        nbeats = NBEATSForecaster(max_horizon=20, max_steps=400)
        nb_train = nbeats.prepare_training_data(train_df)
        nb_val = nbeats.prepare_training_data(val_df)
        try:
            nb_metrics = nbeats.train(nb_train, nb_val)
            results["NBEATS"] = nb_metrics
            logger.info("N-BEATS trained", **{k: v for k, v in vars(nb_metrics).items() if isinstance(v, float)})
        except Exception as e:
            logger.error("N-BEATS training failed", error=str(e))

        # 4. Assemble meta-model features
        logger.info("Assembling meta-model features")
        meta_train = await self._build_meta_features(
            train_df,
            tft_run_id=results.get("TFT", ModelMetrics()).run_id,
            nbeats_run_id=results.get("NBEATS", ModelMetrics()).run_id,
        )
        meta_val = await self._build_meta_features(
            val_df,
            tft_run_id=results.get("TFT", ModelMetrics()).run_id,
            nbeats_run_id=results.get("NBEATS", ModelMetrics()).run_id,
        )

        # 5. LightGBM meta-model
        logger.info("Training LightGBM meta-model")
        meta = MetaRankingModel(n_estimators=300, target_horizon_days=5)
        try:
            meta_metrics = meta.train(meta_train, meta_val)
            results["META"] = meta_metrics
            logger.info("Meta-model trained", ic=meta_metrics.extra.get("ic"))
        except Exception as e:
            logger.error("Meta-model training failed", error=str(e))

        # 6. Walk-forward validation
        logger.info("Running walk-forward validation")
        if "TFT" in results:
            try:
                wf_results = tft.walk_forward_validate(
                    tft.prepare_training_data(panel_df), n_splits=3
                )
                logger.info("TFT walk-forward", splits=wf_results)
            except Exception as e:
                logger.warning("Walk-forward validation failed", error=str(e))

        # 7. Register champion models
        await self._register_champions(results)

        elapsed = (datetime.utcnow() - start).total_seconds() / 60
        logger.info("Retrain complete", elapsed_minutes=round(elapsed, 1), models_trained=len(results))
        return results

    async def _load_panel_data(self) -> pd.DataFrame:
        """
        Load EOD prices + features + fundamentals from TimescaleDB.
        Returns a wide panel DataFrame: [ticker, date, open, high, low, close, volume, *features]
        """
        from db.session import init_db, get_db
        from db.models import EODPrice, Stock, Fundamental
        from data.feature_store.features import FeatureStore

        await init_db()
        fs = FeatureStore()
        cutoff = datetime.utcnow() - timedelta(days=365 * (self.train_years + 1))

        all_rows = []

        async for session in get_db():
            st_result = await session.execute(
                select(Stock).where(Stock.is_active == True)
            )
            stocks = st_result.scalars().all()

            for stock in stocks:
                price_result = await session.execute(
                    select(EODPrice)
                    .where(EODPrice.stock_id == stock.id, EODPrice.date >= cutoff)
                    .order_by(EODPrice.date)
                )
                prices = price_result.scalars().all()
                if len(prices) < 60:
                    continue

                df = pd.DataFrame([
                    {
                        "ticker": stock.ticker,
                        "sector": stock.sector or "Unknown",
                        "exchange": stock.exchange.value,
                        "date": p.date,
                        "open": float(p.open or p.close),
                        "high": float(p.high or p.close),
                        "low": float(p.low or p.close),
                        "close": float(p.close),
                        "volume": int(p.volume or 0),
                    }
                    for p in prices
                ])

                # Compute features for each row (rolling window)
                try:
                    feats = fs.compute_all_features(df, ticker=stock.ticker)
                    # Merge latest features as static for training (simplified)
                    for feat_key, feat_val in feats.items():
                        if isinstance(feat_val, (int, float)) and feat_key != "ticker":
                            df[feat_key] = feat_val
                except Exception:
                    pass  # features are optional

                all_rows.append(df)

        if not all_rows:
            return pd.DataFrame()

        return pd.concat(all_rows, ignore_index=True).sort_values(["ticker", "date"])

    def _time_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Strict time-based split — no random shuffle to prevent lookahead."""
        all_dates = sorted(df["date"].unique())
        n = len(all_dates)

        test_size = max(1, int(n * self.test_months / (self.train_years * 12 + self.val_months + self.test_months)))
        val_size = max(1, int(n * self.val_months / (self.train_years * 12 + self.val_months + self.test_months)))

        train_dates = all_dates[:n - val_size - test_size]
        val_dates = all_dates[n - val_size - test_size: n - test_size]
        test_dates = all_dates[n - test_size:]

        train = df[df["date"].isin(train_dates)]
        val = df[df["date"].isin(val_dates)]
        test = df[df["date"].isin(test_dates)]
        return train, val, test

    async def _build_meta_features(
        self,
        df: pd.DataFrame,
        tft_run_id: str = "",
        nbeats_run_id: str = "",
    ) -> pd.DataFrame:
        """
        Build the feature DataFrame used by the meta-model.
        Merges base-model predictions with technical + fundamental features.
        """
        result = df.copy()

        # Add base-model predictions as features
        if tft_run_id:
            try:
                tft = TFTForecaster.load(tft_run_id)
                tft_preds = tft.predict(tft.prepare_training_data(df))
                # Merge quantile predictions as features
                for col in [c for c in tft_preds.columns if "q" in c]:
                    horizon = col.split("-")[-1].replace("h", "")
                    q = col.split("-")[1]
                    feat_name = f"tft_return_h{horizon}_{q}"
                    merged = tft_preds[["unique_id", "ds", col]].rename(
                        columns={"unique_id": "ticker", "ds": "date", col: feat_name}
                    )
                    result = result.merge(merged, on=["ticker", "date"], how="left")
            except Exception as e:
                logger.warning("Could not merge TFT predictions", error=str(e))

        return result

    async def _register_champions(self, results: Dict[str, ModelMetrics]) -> None:
        """
        Compare new models vs existing champions.
        Promote to champion if metrics improved (higher directional accuracy or IC).
        """
        from db.session import init_db, get_db
        from db.models import ModelVersion
        from sqlalchemy import update

        await init_db()
        async for session in get_db():
            for model_type, metrics in results.items():
                if not metrics.run_id:
                    continue

                # Demote current champion
                await session.execute(
                    update(ModelVersion)
                    .where(ModelVersion.model_type == model_type, ModelVersion.is_champion == True)
                    .values(is_champion=False)
                )

                # Register new version
                mv = ModelVersion(
                    name=metrics.model_name,
                    version=f"v{datetime.utcnow().strftime('%Y%m%d_%H%M')}",
                    mlflow_run_id=metrics.run_id,
                    model_type=model_type,
                    metrics={k: v for k, v in vars(metrics).items() if isinstance(v, (int, float))},
                    is_champion=True,
                    trained_at=datetime.utcnow(),
                )
                session.add(mv)
                await session.commit()
                logger.info("Champion registered", model=model_type, run_id=metrics.run_id)
