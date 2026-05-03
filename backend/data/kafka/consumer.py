"""
data/kafka/consumer.py — Async Kafka consumer manager.

Consumes from:
  - tick_data topic → broadcasts to WebSocket price subscribers
  - model_signals topic → broadcasts to WebSocket signal subscribers
  - news_feed topic → triggers NLP pipeline

data/kafka/producer.py (also in this file) — Async Kafka producer.
"""
import asyncio
import json
from datetime import datetime
from typing import Any, Dict, Optional

import structlog
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import KafkaConnectionError

from config import get_settings
from utils.connection_manager import WebSocketConnectionManager

settings = get_settings()
logger = structlog.get_logger(__name__)


# ─── Producer ─────────────────────────────────────────────────────────────────
class KafkaProducerManager:
    """
    Thread-safe async Kafka producer.
    Use as a FastAPI dependency or inject via DI.
    """

    def __init__(self):
        self._producer: Optional[AIOKafkaProducer] = None

    async def start(self) -> None:
        self._producer = AIOKafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            compression_type="gzip",
            acks="all",              # wait for all replicas
            enable_idempotence=True, # exactly-once semantics
        )
        await self._producer.start()
        logger.info("Kafka producer started")

    async def stop(self) -> None:
        if self._producer:
            await self._producer.stop()
            logger.info("Kafka producer stopped")

    async def publish(
        self,
        topic: str,
        value: Dict[str, Any],
        key: Optional[str] = None,
    ) -> None:
        """Publish a message to a Kafka topic."""
        if not self._producer:
            raise RuntimeError("KafkaProducerManager not started")
        await self._producer.send(topic, value=value, key=key)

    async def publish_tick(self, ticker: str, tick_data: Dict) -> None:
        """Publish a market tick (price update)."""
        payload = {
            "ticker": ticker,
            "ts": datetime.utcnow().isoformat(),
            **tick_data,
        }
        await self.publish(settings.kafka_topic_tick_data, payload, key=ticker)

    async def publish_order_event(self, order_id: str, event: Dict) -> None:
        """Publish an order lifecycle event."""
        await self.publish(settings.kafka_topic_order_events, event, key=order_id)

    async def publish_signal(self, ticker: str, signal: Dict) -> None:
        """Publish a model-generated trading signal."""
        await self.publish(settings.kafka_topic_model_signals, signal, key=ticker)


# ─── Consumer Manager ─────────────────────────────────────────────────────────
class KafkaConsumerManager:
    """
    Manages multiple async Kafka consumers.
    Routes messages to appropriate handlers (WebSocket push, DB writes, etc.)
    """

    def __init__(self, ws_manager: WebSocketConnectionManager):
        self._ws_manager = ws_manager
        self._consumers: Dict[str, AIOKafkaConsumer] = {}
        self._running = False

    async def start(self) -> None:
        """Start all topic consumers concurrently."""
        self._running = True
        logger.info("Starting Kafka consumers")

        await asyncio.gather(
            self._consume_tick_data(),
            self._consume_model_signals(),
            self._consume_news_feed(),
        )

    async def stop(self) -> None:
        self._running = False
        for consumer in self._consumers.values():
            await consumer.stop()
        logger.info("Kafka consumers stopped")

    async def _consume_tick_data(self) -> None:
        """
        Consume real-time tick data and broadcast to WebSocket subscribers.
        Each message is forwarded to the group matching the ticker.
        """
        consumer = AIOKafkaConsumer(
            settings.kafka_topic_tick_data,
            bootstrap_servers=settings.kafka_bootstrap_servers,
            group_id=f"{settings.kafka_consumer_group}_ticks",
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
        )
        self._consumers["tick_data"] = consumer
        await consumer.start()

        try:
            async for msg in consumer:
                if not self._running:
                    break
                payload = msg.value
                ticker = payload.get("ticker", "")
                if ticker:
                    await self._ws_manager.broadcast(
                        group=ticker.upper(),
                        message=json.dumps(payload),
                    )
        except Exception as e:
            logger.error("Tick consumer error", error=str(e))
        finally:
            await consumer.stop()

    async def _consume_model_signals(self) -> None:
        """Consume model signals and push to signal WebSocket subscribers."""
        consumer = AIOKafkaConsumer(
            settings.kafka_topic_model_signals,
            bootstrap_servers=settings.kafka_bootstrap_servers,
            group_id=f"{settings.kafka_consumer_group}_signals",
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="latest",
        )
        self._consumers["signals"] = consumer
        await consumer.start()

        try:
            async for msg in consumer:
                if not self._running:
                    break
                await self._ws_manager.broadcast(
                    group="signals",
                    message=json.dumps(msg.value),
                )
        except Exception as e:
            logger.error("Signal consumer error", error=str(e))
        finally:
            await consumer.stop()

    async def _consume_news_feed(self) -> None:
        """Consume news articles and trigger NLP processing."""
        consumer = AIOKafkaConsumer(
            settings.kafka_topic_news_feed,
            bootstrap_servers=settings.kafka_bootstrap_servers,
            group_id=f"{settings.kafka_consumer_group}_news",
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="earliest",
        )
        self._consumers["news"] = consumer
        await consumer.start()

        try:
            async for msg in consumer:
                if not self._running:
                    break
                # Trigger NLP pipeline asynchronously (non-blocking)
                asyncio.create_task(self._process_news_article(msg.value))
        except Exception as e:
            logger.error("News consumer error", error=str(e))
        finally:
            await consumer.stop()

    async def _process_news_article(self, article: Dict) -> None:
        """
        Background task: run NLP pipeline on a news article.
        (Sentiment, summarisation, entity extraction)
        Called asynchronously so it doesn't block the consumer loop.
        """
        from models.nlp_pipeline import NLPPipeline  # lazy import to avoid circular deps
        try:
            nlp = NLPPipeline()
            enriched = await nlp.process_article(article)
            logger.debug("News article processed", title=enriched.get("title", "")[:60])
        except Exception as e:
            logger.error("NLP processing failed", error=str(e))
