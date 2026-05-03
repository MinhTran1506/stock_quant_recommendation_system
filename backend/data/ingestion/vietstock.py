"""
data/ingestion/vietstock.py — Vietstock API connector.

Vietstock (https://vietstock.vn) provides Vietnam-market data including
EOD prices, intraday data, corporate filings, and financial news.
API docs: https://dichvu.vietstock.vn/du-lieu-tai-chinh/datafeed

NOTE: Replace endpoint paths and request schemas with actual Vietstock
API spec once you have an active subscription and access credentials.
"""
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import structlog

from config import get_settings
from data.ingestion.base import BaseDataProvider, DataProviderError

settings = get_settings()
logger = structlog.get_logger(__name__)


class VietstockProvider(BaseDataProvider):
    """
    Connector for Vietstock DataFeed API.

    Authentication: Bearer token header (refreshed automatically).
    Rate limits: Respect X-RateLimit-* response headers.
    """

    provider_name = "vietstock"

    def __init__(self):
        super().__init__()
        self._api_key = settings.vietstock_api_key
        self._base_url = settings.vietstock_api_url
        self._token: Optional[str] = None

    def _default_headers(self) -> Dict[str, str]:
        headers = super()._default_headers()
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    # ── Authentication ─────────────────────────────────────────────────────
    async def authenticate(self) -> None:
        """Obtain/refresh access token. Called lazily before first request."""
        # Vietstock may use OAuth2 or API-key-based auth — adjust to actual spec
        self.log.info("Authenticating with Vietstock API")
        # Example OAuth2 flow:
        # response = await self._request("POST", f"{self._base_url}/oauth/token",
        #     json_body={"client_id": ..., "client_secret": ..., "grant_type": "client_credentials"})
        # self._token = response["access_token"]

    # ── EOD Prices ────────────────────────────────────────────────────────
    async def fetch_eod_prices(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> List[Dict]:
        """
        Fetch daily OHLCV from Vietstock for a given ticker and date range.
        Endpoint: GET /data/price-history (example path — verify with Vietstock docs)
        """
        self.log.info("Fetching EOD prices", ticker=ticker,
                      start=str(start_date), end=str(end_date))

        raw = await self._request(
            "GET",
            f"{self._base_url}/data/price-history",
            params={
                "symbol": ticker,
                "startDate": start_date.strftime("%Y-%m-%d"),
                "endDate": end_date.strftime("%Y-%m-%d"),
                "resolution": "D",
            },
        )

        # Archive raw response
        self._save_raw_to_s3(raw, "eod_prices", ticker=ticker, partition_date=start_date)

        return self._parse_eod(raw, ticker)

    def _parse_eod(self, raw: Any, ticker: str) -> List[Dict]:
        """Normalise Vietstock EOD response to internal schema."""
        # Adjust field names to match Vietstock's actual API response format
        records = raw.get("data", raw) if isinstance(raw, dict) else raw
        result = []
        for r in records:
            result.append({
                "ticker": ticker,
                "date": r.get("tradingDate") or r.get("date"),
                "open": float(r.get("openPrice", r.get("open", 0))),
                "high": float(r.get("highPrice", r.get("high", 0))),
                "low": float(r.get("lowPrice", r.get("low", 0))),
                "close": float(r.get("closePrice", r.get("close", 0))),
                "volume": int(r.get("totalVolume", r.get("volume", 0))),
                "adjusted_close": float(r.get("adjustedClose", r.get("close", 0))),
                "source": self.provider_name,
            })
        return result

    # ── Intraday ──────────────────────────────────────────────────────────
    async def fetch_intraday_prices(
        self,
        ticker: str,
        date_: date,
        interval_minutes: int = 1,
    ) -> List[Dict]:
        self.log.info("Fetching intraday", ticker=ticker, date=str(date_),
                      interval=interval_minutes)

        raw = await self._request(
            "GET",
            f"{self._base_url}/data/intraday",
            params={
                "symbol": ticker,
                "date": date_.strftime("%Y-%m-%d"),
                "resolution": f"{interval_minutes}",
            },
        )
        self._save_raw_to_s3(raw, "intraday", ticker=ticker, partition_date=date_)

        return self._parse_intraday(raw, ticker, interval_minutes)

    def _parse_intraday(
        self, raw: Any, ticker: str, interval_minutes: int
    ) -> List[Dict]:
        records = raw.get("data", raw) if isinstance(raw, dict) else raw
        result = []
        for r in records:
            ts_raw = r.get("time") or r.get("timestamp")
            result.append({
                "ticker": ticker,
                "timestamp": ts_raw,
                "interval_minutes": interval_minutes,
                "open": float(r.get("open", 0)),
                "high": float(r.get("high", 0)),
                "low": float(r.get("low", 0)),
                "close": float(r.get("close", 0)),
                "volume": int(r.get("volume", 0)),
            })
        return result

    # ── Order Book ────────────────────────────────────────────────────────
    async def fetch_order_book(self, ticker: str) -> Dict:
        self.log.info("Fetching order book", ticker=ticker)
        raw = await self._request(
            "GET",
            f"{self._base_url}/data/orderbook",
            params={"symbol": ticker},
        )
        self._save_raw_to_s3(raw, "orderbook", ticker=ticker)

        bids = [[float(b["price"]), int(b["volume"])] for b in raw.get("bids", [])]
        asks = [[float(a["price"]), int(a["volume"])] for a in raw.get("asks", [])]
        mid = (bids[0][0] + asks[0][0]) / 2 if bids and asks else None
        spread = (asks[0][0] - bids[0][0]) if bids and asks else None

        return {
            "ticker": ticker,
            "timestamp": datetime.utcnow().isoformat(),
            "bids": bids,
            "asks": asks,
            "mid_price": mid,
            "spread": spread,
        }

    # ── Fundamentals ──────────────────────────────────────────────────────
    async def fetch_fundamentals(self, ticker: str) -> Dict:
        self.log.info("Fetching fundamentals", ticker=ticker)
        raw = await self._request(
            "GET",
            f"{self._base_url}/data/financials",
            params={"symbol": ticker, "type": "latest"},
        )
        self._save_raw_to_s3(raw, "fundamentals", ticker=ticker)

        data = raw.get("data", raw) if isinstance(raw, dict) else raw
        if isinstance(data, list) and data:
            data = data[0]

        return {
            "ticker": ticker,
            "report_date": data.get("reportDate"),
            "period": data.get("period"),
            "pe_ratio": data.get("pe"),
            "pb_ratio": data.get("pb"),
            "roe": data.get("roe"),
            "roa": data.get("roa"),
            "debt_to_equity": data.get("debtToEquity"),
            "revenue": data.get("revenue"),
            "net_income": data.get("netIncome"),
            "eps": data.get("eps"),
            "dividend_yield": data.get("dividendYield"),
            "raw_data": raw,
        }

    # ── News ──────────────────────────────────────────────────────────────
    async def fetch_news(
        self,
        ticker: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        params: Dict = {"limit": limit}
        if ticker:
            params["symbol"] = ticker

        raw = await self._request(
            "GET",
            f"{self._base_url}/news/latest",
            params=params,
        )
        self._save_raw_to_s3(raw, "news", ticker=ticker)

        articles = raw.get("data", raw) if isinstance(raw, dict) else raw
        return [
            {
                "title": a.get("title"),
                "source": a.get("source", self.provider_name),
                "url": a.get("url") or a.get("link"),
                "published_at": a.get("publishedAt") or a.get("date"),
                "raw_content": a.get("content") or a.get("body"),
                "ticker": ticker,
            }
            for a in (articles or [])
        ]

    # ── Stock list ────────────────────────────────────────────────────────
    async def fetch_stock_list(self) -> List[Dict]:
        self.log.info("Fetching stock list")
        raw = await self._request("GET", f"{self._base_url}/data/stock-list")
        self._save_raw_to_s3(raw, "stock_list")

        stocks = raw.get("data", raw) if isinstance(raw, dict) else raw
        return [
            {
                "ticker": s.get("symbol") or s.get("ticker"),
                "name": s.get("name") or s.get("companyName"),
                "exchange": s.get("exchange", "HOSE"),
                "sector": s.get("sector"),
                "industry": s.get("industry"),
                "market_cap": s.get("marketCap"),
                "listing_date": s.get("listingDate"),
            }
            for s in (stocks or [])
        ]
