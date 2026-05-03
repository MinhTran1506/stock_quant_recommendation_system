"""
data/ingestion/fiingroup.py — FiinGroup API connector.

FiinGroup (https://fiingroup.vn) provides professional-grade Vietnam market
data including financial statements, industry benchmarks, ownership data,
and ESG scores — complementing Vietstock's price feeds.

API docs: https://api.fiingroup.vn/docs (requires enterprise subscription)
"""
from datetime import date
from typing import Any, Dict, List, Optional

import structlog

from config import get_settings
from data.ingestion.base import BaseDataProvider

settings = get_settings()
logger = structlog.get_logger(__name__)


class FiinGroupProvider(BaseDataProvider):
    """
    Connector for FiinGroup DataFeed API.
    Speciality: financial statements, corporate events, ownership data.
    Use alongside VietstockProvider for complete data coverage.
    """

    provider_name = "fiingroup"

    def __init__(self):
        super().__init__()
        self._api_key = settings.fiingroup_api_key
        self._base_url = settings.fiingroup_api_url

    def _default_headers(self) -> Dict[str, str]:
        headers = super()._default_headers()
        if self._api_key:
            headers["x-api-key"] = self._api_key
        return headers

    async def fetch_eod_prices(
        self, ticker: str, start_date: date, end_date: date
    ) -> List[Dict]:
        raw = await self._request(
            "GET",
            f"{self._base_url}/market/price-history",
            params={
                "ticker": ticker,
                "from": start_date.isoformat(),
                "to": end_date.isoformat(),
                "type": "daily",
            },
        )
        self._save_raw_to_s3(raw, "eod_prices", ticker=ticker, partition_date=start_date)

        records = raw.get("data", raw) if isinstance(raw, dict) else raw
        return [
            {
                "ticker": ticker,
                "date": r.get("date") or r.get("tradingDate"),
                "open": float(r.get("open", 0)),
                "high": float(r.get("high", 0)),
                "low": float(r.get("low", 0)),
                "close": float(r.get("close", 0)),
                "volume": int(r.get("volume", 0)),
                "adjusted_close": float(r.get("adjustedClose", r.get("close", 0))),
                "source": self.provider_name,
            }
            for r in (records or [])
        ]

    async def fetch_intraday_prices(
        self, ticker: str, date_: date, interval_minutes: int = 1
    ) -> List[Dict]:
        raw = await self._request(
            "GET",
            f"{self._base_url}/market/intraday",
            params={
                "ticker": ticker,
                "date": date_.isoformat(),
                "interval": interval_minutes,
            },
        )
        self._save_raw_to_s3(raw, "intraday", ticker=ticker, partition_date=date_)
        records = raw.get("data", raw) if isinstance(raw, dict) else raw
        return [
            {
                "ticker": ticker,
                "timestamp": r.get("time") or r.get("timestamp"),
                "interval_minutes": interval_minutes,
                "open": float(r.get("open", 0)),
                "high": float(r.get("high", 0)),
                "low": float(r.get("low", 0)),
                "close": float(r.get("close", 0)),
                "volume": int(r.get("volume", 0)),
            }
            for r in (records or [])
        ]

    async def fetch_order_book(self, ticker: str) -> Dict:
        raw = await self._request(
            "GET", f"{self._base_url}/market/orderbook", params={"ticker": ticker}
        )
        self._save_raw_to_s3(raw, "orderbook", ticker=ticker)
        bids = [[float(b["price"]), int(b["vol"])] for b in raw.get("bids", [])]
        asks = [[float(a["price"]), int(a["vol"])] for a in raw.get("asks", [])]
        mid = (bids[0][0] + asks[0][0]) / 2 if bids and asks else None
        return {"ticker": ticker, "bids": bids, "asks": asks, "mid_price": mid}

    async def fetch_fundamentals(self, ticker: str) -> Dict:
        """FiinGroup speciality: detailed financial statements and ratios."""
        raw = await self._request(
            "GET",
            f"{self._base_url}/financials/ratios",
            params={"ticker": ticker, "period": "latest", "type": "quarterly"},
        )
        self._save_raw_to_s3(raw, "fundamentals", ticker=ticker)
        d = raw.get("data", {}) if isinstance(raw, dict) else {}
        return {
            "ticker": ticker,
            "report_date": d.get("reportDate"),
            "period": d.get("period"),
            "pe_ratio": d.get("pe"),
            "pb_ratio": d.get("pb"),
            "roe": d.get("roe"),
            "roa": d.get("roa"),
            "debt_to_equity": d.get("der"),
            "revenue": d.get("revenue"),
            "net_income": d.get("netProfit"),
            "eps": d.get("eps"),
            "dividend_yield": d.get("dividendYield"),
            # FiinGroup extras
            "gross_margin": d.get("grossMargin"),
            "operating_margin": d.get("operatingMargin"),
            "current_ratio": d.get("currentRatio"),
            "quick_ratio": d.get("quickRatio"),
            "asset_turnover": d.get("assetTurnover"),
            "raw_data": raw,
        }

    async def fetch_news(
        self, ticker: Optional[str] = None, limit: int = 50
    ) -> List[Dict]:
        params: Dict = {"limit": limit, "type": "corporate_events,news"}
        if ticker:
            params["ticker"] = ticker
        raw = await self._request("GET", f"{self._base_url}/news", params=params)
        self._save_raw_to_s3(raw, "news", ticker=ticker)
        articles = raw.get("data", raw) if isinstance(raw, dict) else raw
        return [
            {
                "title": a.get("title"),
                "source": a.get("source", self.provider_name),
                "url": a.get("url"),
                "published_at": a.get("publishedAt") or a.get("eventDate"),
                "raw_content": a.get("content"),
                "ticker": ticker,
                "event_type": a.get("eventType"),  # earnings | dividend | agm | etc
            }
            for a in (articles or [])
        ]

    async def fetch_stock_list(self) -> List[Dict]:
        raw = await self._request("GET", f"{self._base_url}/market/securities")
        self._save_raw_to_s3(raw, "stock_list")
        stocks = raw.get("data", raw) if isinstance(raw, dict) else raw
        return [
            {
                "ticker": s.get("ticker"),
                "name": s.get("companyName") or s.get("name"),
                "exchange": s.get("exchange", "HOSE"),
                "sector": s.get("icbSector") or s.get("sector"),
                "industry": s.get("icbIndustry") or s.get("industry"),
                "market_cap": s.get("marketCap"),
                "listing_date": s.get("listingDate"),
                "outstanding_shares": s.get("outstandingShares"),
                "foreign_room": s.get("foreignRoom"),
            }
            for s in (stocks or [])
        ]

    async def fetch_ownership(self, ticker: str) -> Dict:
        """FiinGroup-specific: major shareholder and foreign ownership data."""
        raw = await self._request(
            "GET", f"{self._base_url}/ownership/{ticker}"
        )
        self._save_raw_to_s3(raw, "ownership", ticker=ticker)
        return raw

    async def fetch_corporate_events(
        self, ticker: str, event_types: Optional[List[str]] = None
    ) -> List[Dict]:
        """Corporate actions: dividends, rights issues, AGM dates."""
        params: Dict = {"ticker": ticker}
        if event_types:
            params["types"] = ",".join(event_types)
        raw = await self._request(
            "GET", f"{self._base_url}/corporate-events", params=params
        )
        self._save_raw_to_s3(raw, "corporate_events", ticker=ticker)
        events = raw.get("data", raw) if isinstance(raw, dict) else raw
        return events or []
