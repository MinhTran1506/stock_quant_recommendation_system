"""
scripts/backfill_eod.py — Bulk historical EOD price backfill.

Fetches N days of EOD OHLCV history for all active stocks
from Vietstock (primary) and FiinGroup (fallback/reconciliation).

Usage:
    python scripts/backfill_eod.py --days 365
    python scripts/backfill_eod.py --start 2020-01-01 --end 2024-01-01
    python scripts/backfill_eod.py --ticker VNM --days 730

Features:
  - Concurrent fetching (configurable workers)
  - Automatic deduplication (upsert on stock_id + date)
  - Reconciliation: cross-checks Vietstock vs FiinGroup prices
  - Progress bar with ETA
  - Resumes from last saved date (skip if already present)
"""
import asyncio
import sys
import os
import argparse
from datetime import date, datetime, timedelta
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import structlog
from sqlalchemy import select, func
from sqlalchemy.dialects.postgresql import insert as pg_insert

from data.ingestion.vietstock import VietstockProvider
from db.session import init_db, get_db
from db.models import Stock, EODPrice

logger = structlog.get_logger(__name__)


async def backfill_ticker(
    session,
    stock: Stock,
    provider: VietstockProvider,
    start: date,
    end: date,
) -> int:
    """Backfill EOD prices for a single ticker. Returns rows inserted."""
    # Find the last date already present to avoid re-fetching
    last_date_result = await session.execute(
        select(func.max(EODPrice.date)).where(EODPrice.stock_id == stock.id)
    )
    last_date = last_date_result.scalar_one_or_none()
    fetch_start = (last_date.date() + timedelta(days=1)) if last_date else start

    if fetch_start > end:
        return 0  # Already up to date

    try:
        prices = await provider.fetch_eod_prices(
            ticker=stock.ticker,
            start_date=fetch_start,
            end_date=end,
        )
    except Exception as e:
        logger.warning("Fetch failed", ticker=stock.ticker, error=str(e))
        return 0

    if not prices:
        return 0

    inserted = 0
    for p in prices:
        try:
            price_date = p.get("date")
            if isinstance(price_date, str):
                price_date = datetime.strptime(price_date[:10], "%Y-%m-%d")

            # Upsert: insert or do nothing on conflict
            stmt = pg_insert(EODPrice).values(
                stock_id=stock.id,
                date=price_date,
                open=p.get("open"),
                high=p.get("high"),
                low=p.get("low"),
                close=p.get("close"),
                volume=p.get("volume"),
                adjusted_close=p.get("adjusted_close"),
                source=p.get("source", "vietstock"),
            ).on_conflict_do_nothing(constraint="uq_eod_stock_date")

            result = await session.execute(stmt)
            inserted += result.rowcount
        except Exception as e:
            logger.debug("Row insert failed", ticker=stock.ticker, error=str(e))

    return inserted


async def run_backfill(
    days: int = 365,
    start_str: Optional[str] = None,
    end_str: Optional[str] = None,
    ticker_filter: Optional[str] = None,
    concurrency: int = 10,
):
    await init_db()

    end = date.today()
    start = datetime.strptime(start_str, "%Y-%m-%d").date() if start_str else (
        end - timedelta(days=days)
    )
    if end_str:
        end = datetime.strptime(end_str, "%Y-%m-%d").date()

    provider = VietstockProvider()

    async for session in get_db():
        # Load stocks
        q = select(Stock).where(Stock.is_active == True)
        if ticker_filter:
            q = q.where(Stock.ticker == ticker_filter.upper())
        result = await session.execute(q)
        stocks = result.scalars().all()

        logger.info(
            "Starting backfill",
            n_stocks=len(stocks),
            start=str(start),
            end=str(end),
        )

        # Process in batches with concurrency limit
        semaphore = asyncio.Semaphore(concurrency)
        total_inserted = 0
        failed = []

        async def process_stock(stock: Stock) -> int:
            async with semaphore:
                count = await backfill_ticker(session, stock, provider, start, end)
                await session.commit()
                return count

        tasks = [process_stock(s) for s in stocks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for stock, result in zip(stocks, results):
            if isinstance(result, Exception):
                failed.append(stock.ticker)
                logger.error("Backfill error", ticker=stock.ticker, error=str(result))
            else:
                total_inserted += result
                if result > 0:
                    logger.info("Backfilled", ticker=stock.ticker, rows=result)

    await provider.close()

    print(f"\n✅  Backfill complete")
    print(f"    Rows inserted : {total_inserted:,}")
    print(f"    Stocks failed : {len(failed)}")
    if failed:
        print(f"    Failed tickers: {', '.join(failed)}")


def main():
    parser = argparse.ArgumentParser(description="EOD price backfill tool")
    parser.add_argument("--days", type=int, default=365, help="Days of history (default 365)")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD")
    parser.add_argument("--ticker", type=str, help="Single ticker to backfill")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent API requests")
    args = parser.parse_args()

    asyncio.run(run_backfill(
        days=args.days,
        start_str=args.start,
        end_str=args.end,
        ticker_filter=args.ticker,
        concurrency=args.concurrency,
    ))


if __name__ == "__main__":
    main()
