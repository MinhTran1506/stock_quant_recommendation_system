"""
api/routes/stocks.py — Stock universe and price data endpoints.
"""
from datetime import date
from typing import List, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.middleware.auth import get_current_user
from db.models import EODPrice, Stock, Fundamental, NewsArticle
from db.session import get_db

router = APIRouter()
logger = structlog.get_logger(__name__)


# ─── Schemas ──────────────────────────────────────────────────────────────────
class StockSummary(BaseModel):
    id: str
    ticker: str
    name: str
    exchange: str
    sector: Optional[str]
    market_cap: Optional[float]
    is_active: bool


class PriceBar(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float]


class StockDetail(StockSummary):
    industry: Optional[str]
    listing_date: Optional[str]
    latest_price: Optional[float]
    change_pct_1d: Optional[float]


# ─── Endpoints ────────────────────────────────────────────────────────────────
@router.get("", response_model=List[StockSummary])
async def list_stocks(
    exchange: Optional[str] = Query(None, description="HOSE | HNX | UPCOM"),
    sector: Optional[str] = Query(None),
    active_only: bool = Query(True),
    limit: int = Query(200, le=1000),
    offset: int = Query(0, ge=0),
    _user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all stocks in the universe with optional filters."""
    q = select(Stock)
    if active_only:
        q = q.where(Stock.is_active == True)
    if exchange:
        q = q.where(Stock.exchange == exchange.upper())
    if sector:
        q = q.where(Stock.sector.ilike(f"%{sector}%"))
    q = q.offset(offset).limit(limit)

    result = await db.execute(q)
    stocks = result.scalars().all()
    return [
        StockSummary(
            id=str(s.id),
            ticker=s.ticker,
            name=s.name,
            exchange=s.exchange.value,
            sector=s.sector,
            market_cap=float(s.market_cap) if s.market_cap else None,
            is_active=s.is_active,
        )
        for s in stocks
    ]


@router.get("/{ticker}", response_model=StockDetail)
async def get_stock(
    ticker: str,
    db: AsyncSession = Depends(get_db),
    _user=Depends(get_current_user),
):
    """Get full detail for a single stock by ticker."""
    result = await db.execute(
        select(Stock).where(Stock.ticker == ticker.upper())
    )
    stock = result.scalar_one_or_none()
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock {ticker} not found")

    # Latest price
    price_result = await db.execute(
        select(EODPrice)
        .where(EODPrice.stock_id == stock.id)
        .order_by(EODPrice.date.desc())
        .limit(2)
    )
    prices = price_result.scalars().all()
    latest_price = float(prices[0].close) if prices else None
    change_pct = None
    if len(prices) >= 2:
        prev = float(prices[1].close)
        change_pct = round((latest_price - prev) / prev * 100, 2) if prev else None

    return StockDetail(
        id=str(stock.id),
        ticker=stock.ticker,
        name=stock.name,
        exchange=stock.exchange.value,
        sector=stock.sector,
        industry=stock.industry,
        market_cap=float(stock.market_cap) if stock.market_cap else None,
        is_active=stock.is_active,
        listing_date=str(stock.listing_date.date()) if stock.listing_date else None,
        latest_price=latest_price,
        change_pct_1d=change_pct,
    )


@router.get("/{ticker}/prices", response_model=List[PriceBar])
async def get_prices(
    ticker: str,
    start: Optional[date] = Query(None),
    end: Optional[date] = Query(None),
    interval: str = Query("1d", description="1d | 1w | 1m"),
    limit: int = Query(504, le=2000),
    db: AsyncSession = Depends(get_db),
    _user=Depends(get_current_user),
):
    """
    Return OHLCV price history for a ticker.
    interval: 1d=daily, 1w=weekly aggregate, 1m=monthly aggregate
    """
    result = await db.execute(
        select(Stock).where(Stock.ticker == ticker.upper())
    )
    stock = result.scalar_one_or_none()
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock {ticker} not found")

    q = (
        select(EODPrice)
        .where(EODPrice.stock_id == stock.id)
        .order_by(EODPrice.date.desc())
        .limit(limit)
    )
    if start:
        q = q.where(EODPrice.date >= start)
    if end:
        q = q.where(EODPrice.date <= end)

    price_result = await db.execute(q)
    rows = price_result.scalars().all()
    rows.reverse()  # ascending order

    return [
        PriceBar(
            date=str(r.date.date()),
            open=float(r.open or r.close),
            high=float(r.high or r.close),
            low=float(r.low or r.close),
            close=float(r.close),
            volume=int(r.volume or 0),
            adjusted_close=float(r.adjusted_close) if r.adjusted_close else None,
        )
        for r in rows
    ]
