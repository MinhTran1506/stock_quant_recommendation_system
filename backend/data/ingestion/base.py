"""
data/ingestion/base.py — Abstract base class for all market data providers.

Enforces a consistent interface across Vietstock, FiinGroup, broker feeds, etc.
Raw responses are always persisted to S3 before processing (data lineage).
"""
import asyncio
import gzip
import json
import logging
from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import boto3
import httpx
import structlog
from botocore.exceptions import ClientError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import get_settings

settings = get_settings()
logger = structlog.get_logger(__name__)


class DataProviderError(Exception):
    """Raised when a data provider returns an error response."""


class RateLimitError(DataProviderError):
    """Raised when the provider rate-limits our requests."""


class BaseDataProvider(ABC):
    """
    Abstract base for all Vietnam market data providers.

    Subclasses must implement:
        - fetch_eod_prices()
        - fetch_intraday_prices()
        - fetch_order_book()
        - fetch_fundamentals()
        - fetch_news()
    """

    provider_name: str = "base"

    def __init__(self):
        self._http: Optional[httpx.AsyncClient] = None
        self._s3 = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
        )
        self.log = logger.bind(provider=self.provider_name)

    # ── HTTP client (lazy-initialised) ─────────────────────────────────────
    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=10.0),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
                headers=self._default_headers(),
            )
        return self._http

    def _default_headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    async def close(self) -> None:
        if self._http and not self._http.is_closed:
            await self._http.aclose()

    # ── Resilient HTTP request with retry ─────────────────────────────────
    async def _request(
        self,
        method: str,
        url: str,
        *,
        params: Optional[Dict] = None,
        json_body: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        max_retries: int = 4,
    ) -> Any:
        """Execute an HTTP request with exponential backoff retry."""
        http = await self._get_http()
        merged_headers = {**self._default_headers(), **(headers or {})}

        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type((httpx.TransportError, RateLimitError)),
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=60),
            reraise=True,
        ):
            with attempt:
                response = await http.request(
                    method,
                    url,
                    params=params,
                    json=json_body,
                    headers=merged_headers,
                )
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 5))
                    self.log.warning("Rate limited", retry_after=retry_after)
                    await asyncio.sleep(retry_after)
                    raise RateLimitError("Rate limited by provider")

                if response.status_code >= 400:
                    raise DataProviderError(
                        f"{self.provider_name} returned {response.status_code}: {response.text[:200]}"
                    )

                return response.json()

    # ── S3 raw data archival ───────────────────────────────────────────────
    def _save_raw_to_s3(
        self,
        data: Any,
        data_type: str,
        ticker: Optional[str] = None,
        partition_date: Optional[date] = None,
    ) -> str:
        """
        Persist raw provider response to S3 for lineage and replay.
        Key format: raw/{provider}/{data_type}/{year}/{month}/{day}/{ticker}_{ts}.json.gz
        """
        ts = datetime.utcnow()
        partition = partition_date or ts.date()
        ticker_part = f"{ticker}_" if ticker else ""
        key = (
            f"{self.provider_name}/{data_type}/"
            f"{partition.year}/{partition.month:02d}/{partition.day:02d}/"
            f"{ticker_part}{ts.strftime('%H%M%S%f')}.json.gz"
        )
        compressed = gzip.compress(
            json.dumps(data, default=str).encode("utf-8")
        )
        try:
            self._s3.put_object(
                Bucket=settings.s3_bucket_raw,
                Key=key,
                Body=compressed,
                ContentEncoding="gzip",
                ContentType="application/json",
            )
            self.log.debug("Raw data saved to S3", key=key, size_bytes=len(compressed))
        except ClientError as e:
            self.log.error("Failed to save raw data to S3", error=str(e), key=key)
        return key

    # ── Abstract interface ─────────────────────────────────────────────────
    @abstractmethod
    async def fetch_eod_prices(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> List[Dict]:
        """
        Return list of daily OHLCV dicts:
        [{"date": "2024-01-02", "open": 10000, "high": 10500,
          "low": 9800, "close": 10200, "volume": 1500000}, ...]
        """

    @abstractmethod
    async def fetch_intraday_prices(
        self,
        ticker: str,
        date_: date,
        interval_minutes: int = 1,
    ) -> List[Dict]:
        """
        Return list of intraday OHLCV bars for a single session.
        [{"timestamp": "2024-01-02T09:15:00", "open": ..., ...}, ...]
        """

    @abstractmethod
    async def fetch_order_book(self, ticker: str) -> Dict:
        """
        Return current order book snapshot.
        {"bids": [[price, qty], ...], "asks": [[price, qty], ...]}
        """

    @abstractmethod
    async def fetch_fundamentals(self, ticker: str) -> Dict:
        """Return latest fundamental / financial ratios for a ticker."""

    @abstractmethod
    async def fetch_news(
        self,
        ticker: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """Return recent news articles, optionally filtered by ticker."""

    @abstractmethod
    async def fetch_stock_list(self) -> List[Dict]:
        """Return full list of listed stocks on supported exchanges."""
