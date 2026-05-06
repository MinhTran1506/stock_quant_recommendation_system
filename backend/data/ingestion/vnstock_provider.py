"""
data/ingestion/vnstock_provider.py — vnstock open-source data connector.

vnstock (https://github.com/thinh-vu/vnstock) provides free access to
Vietnam market data (HOSE & HNX) via KBS and VCI sources — no paid API
subscription required.

Install:  pip install -U vnstock
Docs:     https://vnstocks.com/docs
API tree: from vnstock import show_api; show_api()

Rate limits (as of v4):
  - Guest (no key): 20 requests/minute, up to 4 financial report periods
  - Community (free key): 60 requests/minute, up to 8 financial report periods
  - Sponsor: 3–5× higher limits, full history access

Get a free API key at: https://vnstocks.com/login
Set VNSTOCK_API_KEY in your .env to use it.

Note: vnstock is a synchronous library. All calls are executed in the default
ThreadPoolExecutor so they never block the asyncio event loop.
"""

import asyncio
from datetime import date, datetime
from functools import partial
from typing import Dict, List, Optional

import structlog

from config import get_settings
from data.ingestion.base import BaseDataProvider, DataProviderError

settings = get_settings()
logger = structlog.get_logger(__name__)

# Register optional API key once at import time for the whole process lifetime.
# Skipped silently in guest mode — the library still works without a key.
_key = settings.vnstock_api_key
if _key:
    try:
        from vnstock import register_user  # type: ignore

        register_user(api_key=_key)
        logger.info("vnstock: registered with API key", key_prefix=_key[:8])
    except Exception as _exc:
        logger.warning("vnstock: could not register API key", error=str(_exc))


class VnstockProvider(BaseDataProvider):
    """
    Connector wrapping the vnstock open-source library (v4+ Unified UI).

    Default source is KBS — works on all environments without restrictions.
    Switch to 'VCI' as an alternative if KBS is unavailable.

    Covered interfaces
    ──────────────────
    fetch_eod_prices        → Quote.history()
    fetch_intraday_prices   → Quote.intraday()
    fetch_order_book        → Trading.price_board()
    fetch_fundamentals      → Finance.ratio() + Company.overview()
    fetch_news              → Reference.company.news()
    fetch_stock_list        → Listing.all_symbols()
    """

    provider_name = "vnstock"

    def __init__(self, source: str = "KBS"):
        super().__init__()
        self._source = source

    # ── asyncio bridge ────────────────────────────────────────────────────────

    async def _run_sync(self, func, *args, **kwargs):
        """Run a blocking vnstock call in the default thread executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(func, *args, **kwargs))

    # ── EOD Prices ────────────────────────────────────────────────────────────

    async def fetch_eod_prices(
        self, ticker: str, start_date: date, end_date: date
    ) -> List[Dict]:
        """Fetch daily OHLCV from vnstock for *ticker* in [start_date, end_date]."""
        from vnstock import Quote  # type: ignore

        self.log.info(
            "Fetching EOD prices",
            ticker=ticker,
            start=str(start_date),
            end=str(end_date),
        )

        def _fetch():
            q = Quote(symbol=ticker, source=self._source)
            return q.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval="d",
            )

        try:
            df = await self._run_sync(_fetch)
        except Exception as exc:
            raise DataProviderError(
                f"vnstock EOD fetch failed for {ticker}: {exc}"
            ) from exc

        if df is None or df.empty:
            return []

        raw = df.to_dict(orient="records")
        self._save_raw_to_s3(raw, "eod_prices", ticker=ticker, partition_date=start_date)

        return [
            {
                "ticker": ticker,
                # vnstock KBS returns 'time' as the date/timestamp column
                "date": str(r.get("time", r.get("date", "")))[:10],
                "open": float(r.get("open", 0)),
                "high": float(r.get("high", 0)),
                "low": float(r.get("low", 0)),
                "close": float(r.get("close", 0)),
                "volume": int(r.get("volume", 0)),
                # vnstock prices are already adjusted; use close as adjusted_close
                "adjusted_close": float(r.get("close", 0)),
                "source": self.provider_name,
            }
            for r in raw
        ]

    # ── Intraday ──────────────────────────────────────────────────────────────

    async def fetch_intraday_prices(
        self, ticker: str, date_: date, interval_minutes: int = 1
    ) -> List[Dict]:
        """
        Fetch intraday tick-level trades from vnstock.

        vnstock returns matched-order tick data rather than OHLCV bars.
        Each row represents a single matched transaction.  The
        `interval_minutes` parameter is preserved in the output schema for
        compatibility but is not used to aggregate bars here.
        """
        from vnstock import Quote  # type: ignore

        self.log.info("Fetching intraday", ticker=ticker, date=str(date_))

        def _fetch():
            q = Quote(symbol=ticker, source=self._source)
            return q.intraday(symbol=ticker, page_size=10_000, show_log=False)

        try:
            df = await self._run_sync(_fetch)
        except Exception as exc:
            raise DataProviderError(
                f"vnstock intraday fetch failed for {ticker}: {exc}"
            ) from exc

        if df is None or df.empty:
            return []

        raw = df.to_dict(orient="records")
        self._save_raw_to_s3(raw, "intraday", ticker=ticker, partition_date=date_)

        return [
            {
                "ticker": ticker,
                "timestamp": str(r.get("time", r.get("timestamp", ""))),
                "interval_minutes": interval_minutes,
                # Tick data: price / volume per matched order
                "open": float(r.get("open", r.get("price", 0))),
                "high": float(r.get("high", r.get("price", 0))),
                "low": float(r.get("low", r.get("price", 0))),
                "close": float(r.get("close", r.get("price", 0))),
                "volume": int(r.get("volume", r.get("vol", 0))),
            }
            for r in raw
        ]

    # ── Order Book / Price Board ──────────────────────────────────────────────

    async def fetch_order_book(self, ticker: str) -> Dict:
        """
        Fetch the real-time bid/ask price board for *ticker*.

        vnstock's Trading.price_board() returns a flat row per ticker with
        three bid and three ask price/volume levels.
        """
        from vnstock import Trading  # type: ignore

        self.log.info("Fetching price board", ticker=ticker)

        def _fetch():
            return Trading(source=self._source).price_board([ticker])

        try:
            df = await self._run_sync(_fetch)
        except Exception as exc:
            raise DataProviderError(
                f"vnstock price board failed for {ticker}: {exc}"
            ) from exc

        empty = {
            "ticker": ticker,
            "timestamp": datetime.utcnow().isoformat(),
            "bids": [],
            "asks": [],
            "mid_price": None,
            "spread": None,
        }
        if df is None or df.empty:
            return empty

        raw = df.to_dict(orient="records")
        self._save_raw_to_s3(raw, "orderbook", ticker=ticker)

        row = raw[0]
        # vnstock price_board columns use multi-level headers flattened to tuples
        # or flat string names depending on the version. Try common patterns.
        def _pick(row, *keys):
            for k in keys:
                if k in row and row[k] is not None:
                    return row[k]
            # Also try tuple keys (multi-index flattened)
            for k, v in row.items():
                if isinstance(k, tuple) and any(str(part) in keys for part in k):
                    return v
            return None

        bids = [
            [_pick(row, f"bid_price_{i}", f"bidPrice{i}", f"g3_bidPrice{i}"),
             _pick(row, f"bid_volume_{i}", f"bidVol{i}", f"g3_bidVol{i}")]
            for i in range(1, 4)
        ]
        asks = [
            [_pick(row, f"ask_price_{i}", f"askPrice{i}", f"g3_askPrice{i}"),
             _pick(row, f"ask_volume_{i}", f"askVol{i}", f"g3_askVol{i}")]
            for i in range(1, 4)
        ]
        bids = [[float(p), int(v)] for p, v in bids if p is not None and v is not None]
        asks = [[float(p), int(v)] for p, v in asks if p is not None and v is not None]

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

    # ── Fundamentals ──────────────────────────────────────────────────────────

    async def fetch_fundamentals(self, ticker: str) -> Dict:
        """
        Fetch financial ratios and company profile from vnstock.

        Combines Finance.ratio() (latest quarterly) with Company.overview().
        """
        from vnstock import Finance, Company  # type: ignore

        self.log.info("Fetching fundamentals", ticker=ticker)

        def _fetch_ratios():
            return Finance(symbol=ticker, source=self._source).ratio(
                period="quarter", lang="en"
            )

        def _fetch_overview():
            return Company(symbol=ticker, source=self._source).overview()

        try:
            df_ratios, df_overview = await asyncio.gather(
                self._run_sync(_fetch_ratios),
                self._run_sync(_fetch_overview),
            )
        except Exception as exc:
            raise DataProviderError(
                f"vnstock fundamentals failed for {ticker}: {exc}"
            ) from exc

        ratios_raw = (
            df_ratios.to_dict(orient="records")
            if df_ratios is not None and not df_ratios.empty
            else []
        )
        overview_raw = (
            df_overview.to_dict(orient="records")
            if df_overview is not None and not df_overview.empty
            else []
        )
        self._save_raw_to_s3(
            {"ratios": ratios_raw, "overview": overview_raw},
            "fundamentals",
            ticker=ticker,
        )

        # Take the most recent (first) ratio row
        r = ratios_raw[0] if ratios_raw else {}
        ov = overview_raw[0] if overview_raw else {}

        return {
            "ticker": ticker,
            # vnstock KBS ratio column names (lang='en')
            "report_date": r.get("yearReport") or r.get("report_date"),
            "period": r.get("lengthReport") or r.get("period"),
            "pe_ratio": r.get("priceToEarning") or r.get("pe"),
            "pb_ratio": r.get("priceToBook") or r.get("pb"),
            "roe": r.get("returnOnEquity") or r.get("roe"),
            "roa": r.get("returnOnAsset") or r.get("roa"),
            "debt_to_equity": r.get("debtOnEquity") or r.get("debtToEquity"),
            "revenue": r.get("revenue"),
            "net_income": r.get("postTaxProfit") or r.get("netIncome"),
            "eps": r.get("earningPerShare") or r.get("eps"),
            "dividend_yield": r.get("dividend") or r.get("dividendYield"),
            # Company profile extras from overview()
            "company_name": ov.get("shortName") or ov.get("companyName"),
            "exchange": ov.get("exchange"),
            "industry": ov.get("industryName") or ov.get("industry"),
            "raw_data": {"ratios": r, "overview": ov},
        }

    # ── Income / Balance / Cash-flow statements ───────────────────────────────

    async def fetch_financial_statements(
        self, ticker: str, period: str = "quarter"
    ) -> Dict:
        """
        Fetch full financial statement set from vnstock Finance class.
        Returns balance sheet, income statement, cash flow, and ratios.

        period: 'quarter' | 'year'
        """
        from vnstock import Finance  # type: ignore

        self.log.info("Fetching financial statements", ticker=ticker, period=period)

        fin = Finance(symbol=ticker, source=self._source)

        def _bs():
            return fin.balance_sheet(period=period)

        def _is():
            return fin.income_statement(period=period)

        def _cf():
            return fin.cash_flow(period=period)

        def _ratio():
            return fin.ratio(period=period, lang="en")

        try:
            df_bs, df_is, df_cf, df_ratio = await asyncio.gather(
                self._run_sync(_bs),
                self._run_sync(_is),
                self._run_sync(_cf),
                self._run_sync(_ratio),
            )
        except Exception as exc:
            raise DataProviderError(
                f"vnstock financial statements failed for {ticker}: {exc}"
            ) from exc

        def _to_records(df):
            return df.to_dict(orient="records") if df is not None and not df.empty else []

        result = {
            "ticker": ticker,
            "period": period,
            "balance_sheet": _to_records(df_bs),
            "income_statement": _to_records(df_is),
            "cash_flow": _to_records(df_cf),
            "ratios": _to_records(df_ratio),
        }
        self._save_raw_to_s3(result, "financial_statements", ticker=ticker)
        return result

    # ── News ──────────────────────────────────────────────────────────────────

    async def fetch_news(
        self, ticker: Optional[str] = None, limit: int = 50
    ) -> List[Dict]:
        """
        Fetch company or market news via the vnstock Reference domain.

        For market-wide news (ticker=None), fetches from a liquid benchmark
        ticker (VNM) as a proxy — vnstock does not yet provide a market-wide
        news endpoint.
        """
        from vnstock import Reference  # type: ignore

        self.log.info("Fetching news", ticker=ticker, limit=limit)
        proxy = ticker or "VNM"

        def _fetch():
            return Reference().company.news(symbol=proxy)

        try:
            df = await self._run_sync(_fetch)
        except Exception as exc:
            self.log.warning("vnstock news fetch failed", ticker=ticker, error=str(exc))
            return []

        if df is None or df.empty:
            return []

        raw = df.head(limit).to_dict(orient="records")
        self._save_raw_to_s3(raw, "news", ticker=ticker)

        return [
            {
                "title": a.get("title") or a.get("newsTitle"),
                "source": a.get("source", self.provider_name),
                "url": a.get("url") or a.get("newsUrl"),
                "published_at": str(
                    a.get("publishDate") or a.get("newsDate") or ""
                ),
                "raw_content": a.get("content") or a.get("newsContent"),
                "ticker": ticker,
            }
            for a in raw
        ]

    # ── Corporate Events ──────────────────────────────────────────────────────

    async def fetch_corporate_events(
        self, ticker: str, event_types: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Fetch upcoming corporate events (dividends, AGM, rights issues) via
        the vnstock Reference domain.
        """
        from vnstock import Reference  # type: ignore

        self.log.info("Fetching corporate events", ticker=ticker)

        def _fetch():
            return Reference().company.events(symbol=ticker)

        try:
            df = await self._run_sync(_fetch)
        except Exception as exc:
            self.log.warning(
                "vnstock corporate events fetch failed", error=str(exc)
            )
            return []

        if df is None or df.empty:
            return []

        raw = df.to_dict(orient="records")
        self._save_raw_to_s3(raw, "corporate_events", ticker=ticker)
        return raw

    # ── Ownership / Shareholders ──────────────────────────────────────────────

    async def fetch_ownership(self, ticker: str) -> Dict:
        """Fetch major shareholder data from vnstock Reference domain."""
        from vnstock import Reference  # type: ignore

        self.log.info("Fetching ownership", ticker=ticker)

        def _fetch():
            return Reference().company.shareholders(symbol=ticker)

        try:
            df = await self._run_sync(_fetch)
        except Exception as exc:
            raise DataProviderError(
                f"vnstock ownership fetch failed for {ticker}: {exc}"
            ) from exc

        raw = df.to_dict(orient="records") if df is not None and not df.empty else []
        self._save_raw_to_s3(raw, "ownership", ticker=ticker)
        return {"ticker": ticker, "shareholders": raw}

    # ── Stock List ────────────────────────────────────────────────────────────

    async def fetch_stock_list(self) -> List[Dict]:
        """Return all equity symbols listed on HOSE & HNX via vnstock Listing."""
        from vnstock import Listing  # type: ignore

        self.log.info("Fetching stock list")

        def _fetch():
            return Listing(source=self._source).all_symbols()

        try:
            df = await self._run_sync(_fetch)
        except Exception as exc:
            raise DataProviderError(f"vnstock stock list failed: {exc}") from exc

        if df is None or df.empty:
            return []

        raw = df.to_dict(orient="records")
        self._save_raw_to_s3(raw, "stock_list")

        return [
            {
                # KBS Listing columns: symbol, organ_name, type (exchange)
                "ticker": s.get("symbol") or s.get("ticker"),
                "name": s.get("organ_name") or s.get("name") or s.get("companyName"),
                "exchange": s.get("type") or s.get("exchange", "HOSE"),
                "sector": s.get("industry_name") or s.get("sector"),
                "industry": s.get("industry_name") or s.get("industry"),
                "market_cap": None,  # Not in Listing; fetch via fetch_fundamentals()
                "listing_date": str(s.get("listing_date") or ""),
            }
            for s in raw
        ]
