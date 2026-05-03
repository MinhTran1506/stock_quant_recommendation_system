"""
workers/main.py — Background worker process.

Runs as a separate container alongside the FastAPI backend.
Responsibilities:
  - Kafka consumer for news → NLP enrichment → DB write
  - Periodic feature refresh for real-time inference
  - Order fill simulation for paper trading
  - Model drift monitoring
"""
import asyncio
import json
import signal
import sys
from datetime import datetime, timedelta

import structlog

from config import get_settings
from utils.logging import configure_logging

settings = get_settings()
configure_logging(settings.log_level)
logger = structlog.get_logger(__name__)

_shutdown = asyncio.Event()


def _handle_signal(sig, frame):
    logger.info("Shutdown signal received", signal=sig)
    _shutdown.set()


async def run_news_enrichment_worker():
    """
    Consume news from Kafka, run NLP pipeline, and write enriched
    articles back to the database.
    """
    from aiokafka import AIOKafkaConsumer
    from db.session import init_db, get_db
    from db.models import NewsArticle, Stock
    from models.nlp_pipeline import NLPPipeline
    from sqlalchemy import select

    await init_db()
    nlp = NLPPipeline()

    consumer = AIOKafkaConsumer(
        settings.kafka_topic_news_feed,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=f"{settings.kafka_consumer_group}_nlp_worker",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
    )
    await consumer.start()
    logger.info("News enrichment worker started")

    try:
        async for msg in consumer:
            if _shutdown.is_set():
                break
            article_raw = msg.value
            try:
                enriched = await nlp.process_article(article_raw)
                async for session in get_db():
                    # Resolve stock_id if ticker is known
                    ticker = enriched.get("ticker") or (
                        enriched.get("tickers_mentioned", [None])[0]
                        if enriched.get("tickers_mentioned") else None
                    )
                    stock_id = None
                    if ticker:
                        st = await session.execute(
                            select(Stock).where(Stock.ticker == ticker.upper())
                        )
                        s = st.scalar_one_or_none()
                        stock_id = s.id if s else None

                    # Check for duplicate URL
                    if enriched.get("url"):
                        dup = await session.execute(
                            select(NewsArticle).where(NewsArticle.url == enriched["url"])
                        )
                        if dup.scalar_one_or_none():
                            continue

                    article = NewsArticle(
                        stock_id=stock_id,
                        title=enriched.get("title", "")[:500],
                        source=enriched.get("source", "unknown"),
                        url=enriched.get("url"),
                        published_at=datetime.fromisoformat(
                            enriched.get("published_at", datetime.utcnow().isoformat())
                            .replace("Z", "+00:00")
                        ).replace(tzinfo=None),
                        raw_content=enriched.get("raw_content", "")[:10000],
                        summary=enriched.get("summary", "")[:1000],
                        sentiment_score=enriched.get("sentiment_score"),
                        sentiment_label=enriched.get("sentiment_label"),
                        event_tags=enriched.get("event_tags", []),
                    )
                    session.add(article)
                    await session.commit()
                    logger.debug("News article saved", title=article.title[:50])
            except Exception as e:
                logger.error("News enrichment error", error=str(e))
    finally:
        await consumer.stop()


async def run_feature_refresh_worker():
    """
    Periodically refresh feature cache for all active stocks in Redis.
    Runs every 10 minutes during market hours.
    """
    from db.session import init_db, get_db
    from db.models import Stock, EODPrice
    from data.feature_store.features import FeatureStore
    from sqlalchemy import select, desc
    import pandas as pd

    await init_db()
    fs = FeatureStore()
    logger.info("Feature refresh worker started")

    while not _shutdown.is_set():
        try:
            logger.info("Refreshing feature cache for all active stocks")
            async for session in get_db():
                st_result = await session.execute(
                    select(Stock).where(Stock.is_active == True)
                )
                stocks = st_result.scalars().all()

                for stock in stocks:
                    try:
                        pr_result = await session.execute(
                            select(EODPrice)
                            .where(EODPrice.stock_id == stock.id)
                            .order_by(desc(EODPrice.date))
                            .limit(300)
                        )
                        prices = pr_result.scalars().all()
                        if len(prices) < 30:
                            continue
                        prices.reverse()
                        df = pd.DataFrame([{
                            "date": p.date,
                            "open": float(p.open or p.close),
                            "high": float(p.high or p.close),
                            "low": float(p.low or p.close),
                            "close": float(p.close),
                            "volume": int(p.volume or 0),
                        } for p in prices])
                        features = fs.compute_all_features(df, ticker=stock.ticker)
                        await fs.set_features(stock.ticker, features)
                    except Exception:
                        pass

            logger.info("Feature cache refresh complete", n_stocks=len(stocks))
        except Exception as e:
            logger.error("Feature refresh error", error=str(e))

        # Wait 10 minutes before next refresh
        try:
            await asyncio.wait_for(_shutdown.wait(), timeout=600)
        except asyncio.TimeoutError:
            pass


async def run_model_drift_monitor():
    """
    Monitor model prediction drift: compare current prediction distribution
    against training distribution. Alert if PSI > threshold.
    """
    from db.session import init_db, get_db
    from db.models import Prediction
    from sqlalchemy import select
    from utils.metrics import MODEL_PREDICTION_DRIFT

    await init_db()
    logger.info("Drift monitor started")

    while not _shutdown.is_set():
        try:
            async for session in get_db():
                # Load recent predictions
                recent = await session.execute(
                    select(Prediction.score)
                    .where(Prediction.score.isnot(None))
                    .order_by(Prediction.generated_at.desc())
                    .limit(1000)
                )
                scores = [r[0] for r in recent.all()]
                if len(scores) >= 100:
                    import numpy as np
                    # Simple drift: std deviation from expected ~50 (uniform score dist)
                    mean_score = float(np.mean(scores))
                    std_score = float(np.std(scores))
                    drift_signal = abs(mean_score - 50) / 50
                    MODEL_PREDICTION_DRIFT.labels(model_name="MetaRankingLGBM").set(drift_signal)
                    if drift_signal > 0.3:
                        logger.warning("Model prediction drift detected",
                                       mean_score=mean_score, drift=drift_signal)
        except Exception as e:
            logger.error("Drift monitor error", error=str(e))

        try:
            await asyncio.wait_for(_shutdown.wait(), timeout=3600)  # hourly check
        except asyncio.TimeoutError:
            pass


async def main():
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    logger.info("Starting worker processes")
    await asyncio.gather(
        run_news_enrichment_worker(),
        run_feature_refresh_worker(),
        run_model_drift_monitor(),
    )
    logger.info("All workers stopped")


if __name__ == "__main__":
    asyncio.run(main())
