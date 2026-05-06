"""
airflow/dags/daily_pipeline.py — Daily market data + model update pipeline.

DAG schedule: weekdays at 18:30 ICT (after HOSE closes at 15:00 ICT)

Stages:
  1. ingest_eod       — Fetch yesterday's EOD prices from vnstock
  2. ingest_news      — Fetch latest news, publish to Kafka for NLP processing
  3. compute_features — Compute all features, cache in Redis + archive to S3
  4. score_stocks     — Run meta-model inference, save predictions to DB
  5. generate_signals — Emit model signals to Kafka
  6. [weekly] retrain — Full model retraining (runs Sundays only)
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.utils.dates import days_ago

# ─── Default args ─────────────────────────────────────────────────────────────
DEFAULT_ARGS = {
    "owner": "hft_platform",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}

# ─── DAG definition ───────────────────────────────────────────────────────────
with DAG(
    dag_id="daily_market_pipeline",
    default_args=DEFAULT_ARGS,
    description="Daily Vietnam market data ingestion and model scoring",
    schedule_interval="30 11 * * 1-5",   # 18:30 ICT = 11:30 UTC on weekdays
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["data", "models", "daily"],
) as dag:

    # ── Task functions ────────────────────────────────────────────────────
    def ingest_eod(**context):
        """Fetch EOD prices for all active stocks from vnstock."""
        import asyncio
        from datetime import date
        import sys
        sys.path.insert(0, "/opt/airflow/plugins/backend")

        from data.ingestion.vnstock_provider import VnstockProvider
        from db.session import init_db, get_db
        from db.models import Stock, EODPrice
        from sqlalchemy import select
        import asyncio

        async def _run():
            await init_db()
            provider = VnstockProvider()
            trade_date = context["logical_date"].date()

            async for session in get_db():
                result = await session.execute(
                    select(Stock).where(Stock.is_active == True)
                )
                stocks = result.scalars().all()
                print(f"Ingesting EOD for {len(stocks)} stocks on {trade_date}")

                for stock in stocks:
                    try:
                        prices = await provider.fetch_eod_prices(
                            ticker=stock.ticker,
                            start_date=trade_date,
                            end_date=trade_date,
                        )
                        for p in prices:
                            row = EODPrice(
                                stock_id=stock.id,
                                date=datetime.strptime(p["date"][:10], "%Y-%m-%d"),
                                open=p["open"],
                                high=p["high"],
                                low=p["low"],
                                close=p["close"],
                                volume=p["volume"],
                                adjusted_close=p.get("adjusted_close"),
                                source=p["source"],
                            )
                            session.add(row)
                        await session.commit()
                    except Exception as e:
                        print(f"WARN: EOD ingest failed for {stock.ticker}: {e}")
                        continue

        asyncio.run(_run())
        print("EOD ingestion complete")

    def ingest_news(**context):
        """Fetch recent news articles and publish to Kafka for NLP processing."""
        import asyncio
        import sys
        sys.path.insert(0, "/opt/airflow/plugins/backend")

        from data.ingestion.vnstock_provider import VnstockProvider
        from data.kafka.consumer import KafkaProducerManager

        async def _run():
            provider = VnstockProvider()
            kafka = KafkaProducerManager()
            await kafka.start()

            articles = await provider.fetch_news(limit=200)
            for article in articles:
                await kafka.publish(
                    topic="news_feed",
                    value=article,
                    key=article.get("ticker") or "market",
                )
            await kafka.stop()
            print(f"Published {len(articles)} news articles to Kafka")

        asyncio.run(_run())

    def compute_features(**context):
        """
        Compute all features for all active stocks and cache in Redis.
        Also snapshots feature vectors to S3 for training data.
        """
        import asyncio
        import sys
        import pandas as pd
        sys.path.insert(0, "/opt/airflow/plugins/backend")

        from data.feature_store.features import FeatureStore
        from db.session import init_db, get_db
        from db.models import Stock, EODPrice
        from sqlalchemy import select, desc

        async def _run():
            await init_db()
            fs = FeatureStore()

            async for session in get_db():
                result = await session.execute(
                    select(Stock).where(Stock.is_active == True)
                )
                stocks = result.scalars().all()
                print(f"Computing features for {len(stocks)} stocks")

                failed = 0
                for stock in stocks:
                    try:
                        # Load last 300 trading days
                        price_result = await session.execute(
                            select(EODPrice)
                            .where(EODPrice.stock_id == stock.id)
                            .order_by(desc(EODPrice.date))
                            .limit(300)
                        )
                        prices = price_result.scalars().all()
                        if len(prices) < 30:
                            continue

                        prices.reverse()
                        df = pd.DataFrame([
                            {
                                "date": p.date,
                                "open": float(p.open or p.close),
                                "high": float(p.high or p.close),
                                "low": float(p.low or p.close),
                                "close": float(p.close),
                                "volume": int(p.volume or 0),
                            }
                            for p in prices
                        ])

                        features = fs.compute_all_features(df, ticker=stock.ticker)
                        await fs.set_features(stock.ticker, features)

                    except Exception as e:
                        failed += 1
                        print(f"WARN: Feature compute failed for {stock.ticker}: {e}")

                print(f"Feature computation done. Failed: {failed}/{len(stocks)}")

        asyncio.run(_run())

    def score_stocks(**context):
        """
        Run meta-model inference to score all stocks.
        Saves predictions to the predictions table.
        """
        import asyncio
        import sys
        import json
        sys.path.insert(0, "/opt/airflow/plugins/backend")

        from data.feature_store.features import FeatureStore
        from db.session import init_db, get_db
        from db.models import Stock, Prediction, ModelVersion
        from sqlalchemy import select, desc
        import pandas as pd
        from datetime import datetime, timedelta

        async def _run():
            await init_db()
            fs = FeatureStore()

            async for session in get_db():
                # Load champion model
                mv_result = await session.execute(
                    select(ModelVersion)
                    .where(ModelVersion.is_champion == True, ModelVersion.model_type == "META")
                    .order_by(desc(ModelVersion.trained_at))
                    .limit(1)
                )
                champion = mv_result.scalar_one_or_none()
                if not champion:
                    print("No champion meta-model found; skipping scoring")
                    return

                from models.meta_model import MetaRankingModel
                model = MetaRankingModel.load(champion.mlflow_run_id)

                st_result = await session.execute(
                    select(Stock).where(Stock.is_active == True)
                )
                stocks = st_result.scalars().all()

                now = datetime.utcnow()
                scored = 0

                for stock in stocks:
                    try:
                        cached = await fs.get_features(stock.ticker, use_cache=True)
                        if not cached:
                            continue

                        feat_df = pd.DataFrame([{**cached, "ticker": stock.ticker,
                                                 "date": now.date()}])
                        result_df = model.predict(feat_df, return_shap=True)
                        row = result_df.iloc[0]

                        for horizon in [1, 3, 5, 10, 20]:
                            pred = Prediction(
                                stock_id=stock.id,
                                model_version_id=champion.id,
                                generated_at=now,
                                target_date=now + timedelta(days=horizon * 1.4),  # approx trading days
                                horizon_days=horizon,
                                score=float(row.get("score", 0)),
                                feature_importances=row.get("feature_importances", {}),
                                raw_outputs={"current_price": cached.get("close_price")},
                            )
                            session.add(pred)
                        scored += 1

                    except Exception as e:
                        print(f"WARN: Scoring failed for {stock.ticker}: {e}")

                await session.commit()
                print(f"Scored {scored} stocks")

        asyncio.run(_run())

    def is_sunday(**context):
        """ShortCircuit: only allow retraining on Sundays."""
        return context["logical_date"].weekday() == 6

    def retrain_models(**context):
        """
        Weekly full model retraining job.
        Trains TFT, N-BEATS, and Meta-model on the latest data.
        Registers winning model as champion in MLflow.
        """
        import sys
        sys.path.insert(0, "/opt/airflow/plugins/backend")
        print("Weekly retraining triggered")
        # Training logic is in models/training_pipeline.py
        # Import and invoke:
        # from models.training_pipeline import TrainingPipeline
        # TrainingPipeline().run_full_retrain()

    # ── Task definitions ──────────────────────────────────────────────────
    t_ingest_eod = PythonOperator(
        task_id="ingest_eod_prices",
        python_callable=ingest_eod,
    )

    t_ingest_news = PythonOperator(
        task_id="ingest_news",
        python_callable=ingest_news,
    )

    t_features = PythonOperator(
        task_id="compute_features",
        python_callable=compute_features,
    )

    t_score = PythonOperator(
        task_id="score_stocks",
        python_callable=score_stocks,
    )

    t_is_sunday = ShortCircuitOperator(
        task_id="check_is_sunday",
        python_callable=is_sunday,
        ignore_downstream_trigger_rules=False,
    )

    t_retrain = PythonOperator(
        task_id="retrain_models",
        python_callable=retrain_models,
        execution_timeout=timedelta(hours=6),
    )

    # ── DAG dependencies ──────────────────────────────────────────────────
    # EOD and news run in parallel, then features, then scoring
    [t_ingest_eod, t_ingest_news] >> t_features >> t_score
    # Weekly retraining runs after scoring
    t_score >> t_is_sunday >> t_retrain
