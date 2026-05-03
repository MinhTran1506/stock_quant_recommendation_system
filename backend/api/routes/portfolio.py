"""
api/routes/portfolio.py — Portfolio, positions, and order management endpoints.
"""
from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from api.middleware.auth import get_current_user
from db.models import Order, OrderSide, OrderStatus, OrderType, Portfolio, Position, Stock
from db.session import get_db

router = APIRouter()
logger = structlog.get_logger(__name__)


# ─── Schemas ──────────────────────────────────────────────────────────────────
class PortfolioCreate(BaseModel):
    name: str = Field(..., max_length=100)
    description: Optional[str] = None
    initial_capital: float = Field(..., gt=0)
    currency: str = "VND"


class PortfolioOut(BaseModel):
    id: str
    name: str
    description: Optional[str]
    initial_capital: float
    currency: str
    is_paper: bool
    created_at: str
    # Computed
    current_value: Optional[float] = None
    total_pnl_pct: Optional[float] = None
    n_positions: int = 0


class PositionOut(BaseModel):
    id: str
    ticker: str
    stock_name: str
    quantity: int
    avg_cost: Optional[float]
    current_price: Optional[float]
    market_value: Optional[float]
    unrealised_pnl: Optional[float]
    unrealised_pnl_pct: Optional[float]
    is_open: bool


class OrderCreate(BaseModel):
    ticker: str
    side: str = Field(..., pattern="^(BUY|SELL)$")
    order_type: str = Field("MARKET", pattern="^(MARKET|LIMIT|STOP|STOP_LIMIT)$")
    quantity: int = Field(..., gt=0)
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None


class OrderOut(BaseModel):
    id: str
    ticker: str
    side: str
    order_type: str
    status: str
    quantity: int
    limit_price: Optional[float]
    filled_quantity: int
    avg_fill_price: Optional[float]
    commission: Optional[float]
    is_paper: bool
    submitted_at: str
    filled_at: Optional[str]


# ─── Endpoints ────────────────────────────────────────────────────────────────
@router.get("/", response_model=List[PortfolioOut])
async def list_portfolios(
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    result = await db.execute(
        select(Portfolio).where(Portfolio.user_id == user.id)
    )
    portfolios = result.scalars().all()

    out = []
    for p in portfolios:
        # Count open positions
        pos_result = await db.execute(
            select(func.count()).where(
                Position.portfolio_id == p.id,
                Position.is_open == True,
            )
        )
        n_pos = pos_result.scalar() or 0
        out.append(
            PortfolioOut(
                id=str(p.id),
                name=p.name,
                description=p.description,
                initial_capital=float(p.initial_capital),
                currency=p.currency,
                is_paper=p.is_paper,
                created_at=p.created_at.isoformat(),
                n_positions=n_pos,
            )
        )
    return out


@router.post("/", response_model=PortfolioOut, status_code=201)
async def create_portfolio(
    payload: PortfolioCreate,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    portfolio = Portfolio(
        user_id=user.id,
        name=payload.name,
        description=payload.description,
        initial_capital=payload.initial_capital,
        currency=payload.currency,
        is_paper=True,  # all portfolios are paper by default
    )
    db.add(portfolio)
    await db.commit()
    await db.refresh(portfolio)
    logger.info("Portfolio created", portfolio_id=str(portfolio.id), user=str(user.id))

    return PortfolioOut(
        id=str(portfolio.id),
        name=portfolio.name,
        description=portfolio.description,
        initial_capital=float(portfolio.initial_capital),
        currency=portfolio.currency,
        is_paper=portfolio.is_paper,
        created_at=portfolio.created_at.isoformat(),
        n_positions=0,
    )


@router.get("/{portfolio_id}", response_model=PortfolioOut)
async def get_portfolio(
    portfolio_id: str,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    result = await db.execute(
        select(Portfolio).where(
            Portfolio.id == portfolio_id,
            Portfolio.user_id == user.id,
        )
    )
    p = result.scalar_one_or_none()
    if not p:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    pos_result = await db.execute(
        select(func.count()).where(
            Position.portfolio_id == p.id,
            Position.is_open == True,
        )
    )
    n_pos = pos_result.scalar() or 0

    return PortfolioOut(
        id=str(p.id),
        name=p.name,
        description=p.description,
        initial_capital=float(p.initial_capital),
        currency=p.currency,
        is_paper=p.is_paper,
        created_at=p.created_at.isoformat(),
        n_positions=n_pos,
    )


@router.get("/{portfolio_id}/positions", response_model=List[PositionOut])
async def list_positions(
    portfolio_id: str,
    open_only: bool = Query(True),
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    # Verify ownership
    pf = await db.execute(
        select(Portfolio).where(Portfolio.id == portfolio_id, Portfolio.user_id == user.id)
    )
    if not pf.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Portfolio not found")

    q = select(Position, Stock).join(Stock, Position.stock_id == Stock.id).where(
        Position.portfolio_id == portfolio_id
    )
    if open_only:
        q = q.where(Position.is_open == True)

    result = await db.execute(q)
    rows = result.all()

    out = []
    for pos, stock in rows:
        avg_cost = float(pos.avg_cost) if pos.avg_cost else None
        # In production, fetch live price from cache
        current_price = avg_cost  # placeholder
        market_value = current_price * pos.quantity if current_price else None
        cost_basis = avg_cost * pos.quantity if avg_cost else None
        unrealised_pnl = (market_value - cost_basis) if (market_value and cost_basis) else None
        unrealised_pnl_pct = (
            (unrealised_pnl / cost_basis * 100) if (unrealised_pnl and cost_basis and cost_basis != 0) else None
        )
        out.append(
            PositionOut(
                id=str(pos.id),
                ticker=stock.ticker,
                stock_name=stock.name,
                quantity=pos.quantity,
                avg_cost=avg_cost,
                current_price=current_price,
                market_value=market_value,
                unrealised_pnl=unrealised_pnl,
                unrealised_pnl_pct=unrealised_pnl_pct,
                is_open=pos.is_open,
            )
        )
    return out


@router.get("/{portfolio_id}/orders", response_model=List[OrderOut])
async def list_orders(
    portfolio_id: str,
    limit: int = Query(50, le=200),
    status_filter: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    pf = await db.execute(
        select(Portfolio).where(Portfolio.id == portfolio_id, Portfolio.user_id == user.id)
    )
    if not pf.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Portfolio not found")

    q = (
        select(Order, Stock)
        .join(Stock, Order.stock_id == Stock.id)
        .where(Order.portfolio_id == portfolio_id)
        .order_by(desc(Order.submitted_at))
        .limit(limit)
    )
    if status_filter:
        q = q.where(Order.status == status_filter.upper())

    result = await db.execute(q)
    rows = result.all()

    return [
        OrderOut(
            id=str(o.id),
            ticker=s.ticker,
            side=o.side.value,
            order_type=o.order_type.value,
            status=o.status.value,
            quantity=o.quantity,
            limit_price=float(o.limit_price) if o.limit_price else None,
            filled_quantity=o.filled_quantity or 0,
            avg_fill_price=float(o.avg_fill_price) if o.avg_fill_price else None,
            commission=float(o.commission) if o.commission else None,
            is_paper=o.is_paper,
            submitted_at=o.submitted_at.isoformat(),
            filled_at=o.filled_at.isoformat() if o.filled_at else None,
        )
        for o, s in rows
    ]


@router.post("/{portfolio_id}/orders", response_model=OrderOut, status_code=201)
async def submit_order(
    portfolio_id: str,
    payload: OrderCreate,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """Submit a paper order to a portfolio (live orders require regulatory clearance)."""
    from config import get_settings
    settings = get_settings()
    if not settings.live_trading_enabled:
        # Enforce paper-only
        pass

    # Validate portfolio ownership
    pf_result = await db.execute(
        select(Portfolio).where(Portfolio.id == portfolio_id, Portfolio.user_id == user.id)
    )
    portfolio = pf_result.scalar_one_or_none()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    # Validate stock
    stock_result = await db.execute(
        select(Stock).where(Stock.ticker == payload.ticker.upper())
    )
    stock = stock_result.scalar_one_or_none()
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock {payload.ticker} not found")

    order = Order(
        portfolio_id=portfolio_id,
        stock_id=stock.id,
        side=OrderSide(payload.side),
        order_type=OrderType(payload.order_type),
        status=OrderStatus.PENDING,
        quantity=payload.quantity,
        limit_price=payload.limit_price,
        stop_price=payload.stop_price,
        is_paper=True,
        submitted_at=datetime.utcnow(),
    )
    db.add(order)

    # Paper fill simulation: immediate fill at limit or last known price
    order.status = OrderStatus.FILLED
    order.filled_quantity = payload.quantity
    order.avg_fill_price = payload.limit_price or 0
    order.commission = float(order.avg_fill_price or 0) * payload.quantity * 0.0015
    order.filled_at = datetime.utcnow()

    await db.commit()
    await db.refresh(order)
    logger.info("Paper order submitted", ticker=payload.ticker, side=payload.side, qty=payload.quantity)

    return OrderOut(
        id=str(order.id),
        ticker=stock.ticker,
        side=order.side.value,
        order_type=order.order_type.value,
        status=order.status.value,
        quantity=order.quantity,
        limit_price=float(order.limit_price) if order.limit_price else None,
        filled_quantity=order.filled_quantity,
        avg_fill_price=float(order.avg_fill_price) if order.avg_fill_price else None,
        commission=float(order.commission) if order.commission else None,
        is_paper=order.is_paper,
        submitted_at=order.submitted_at.isoformat(),
        filled_at=order.filled_at.isoformat() if order.filled_at else None,
    )
