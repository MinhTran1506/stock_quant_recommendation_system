"""
api/routes/universe.py — Stock universe management.
api/routes/strategy.py — Strategy CRUD and activation.
"""
from typing import Dict, List, Optional
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from api.middleware.auth import get_current_user, get_current_superuser
from db.models import Stock, Strategy, StrategyStatus
from db.session import get_db

# ─── Universe router ──────────────────────────────────────────────────────────
universe_router = APIRouter()
logger = structlog.get_logger(__name__)


@universe_router.get("/sectors")
async def list_sectors(
    db: AsyncSession = Depends(get_db),
    _user=Depends(get_current_user),
):
    """Return all sectors and stock counts."""
    from sqlalchemy import func
    result = await db.execute(
        select(Stock.sector, func.count().label("count"))
        .where(Stock.is_active == True, Stock.sector.isnot(None))
        .group_by(Stock.sector)
        .order_by(desc("count"))
    )
    return [{"sector": r.sector, "count": r.count} for r in result.all()]


@universe_router.get("/exchanges")
async def list_exchanges(
    db: AsyncSession = Depends(get_db),
    _user=Depends(get_current_user),
):
    from sqlalchemy import func
    result = await db.execute(
        select(Stock.exchange, func.count().label("count"))
        .where(Stock.is_active == True)
        .group_by(Stock.exchange)
    )
    return [{"exchange": r.exchange.value, "count": r.count} for r in result.all()]


router = universe_router  # exported as `universe`


# ─── Strategy router ──────────────────────────────────────────────────────────
strategy_router = APIRouter()


class StrategyCreate(BaseModel):
    name: str
    description: Optional[str] = None
    config: Dict = {}
    universe_filter: Optional[Dict] = None


class StrategyOut(BaseModel):
    id: str
    name: str
    description: Optional[str]
    status: str
    config: Dict
    created_at: str


@strategy_router.get("/", response_model=List[StrategyOut])
async def list_strategies(
    db: AsyncSession = Depends(get_db),
    _user=Depends(get_current_user),
):
    result = await db.execute(select(Strategy).order_by(desc(Strategy.created_at)))
    strategies = result.scalars().all()
    return [_strategy_out(s) for s in strategies]


@strategy_router.post("/", response_model=StrategyOut, status_code=201)
async def create_strategy(
    payload: StrategyCreate,
    db: AsyncSession = Depends(get_db),
    _user=Depends(get_current_user),
):
    s = Strategy(
        name=payload.name,
        description=payload.description,
        config=payload.config,
        universe_filter=payload.universe_filter,
        status=StrategyStatus.INACTIVE,
    )
    db.add(s)
    await db.commit()
    await db.refresh(s)
    return _strategy_out(s)


@strategy_router.patch("/{strategy_id}/activate")
async def activate_strategy(
    strategy_id: str,
    mode: str = Query("paper", pattern="^(paper|live)$"),
    db: AsyncSession = Depends(get_db),
    _user=Depends(get_current_superuser),  # superuser only
):
    """
    Activate a strategy in paper or live mode.
    Live mode requires LIVE_TRADING_ENABLED=true (regulatory gate).
    """
    from config import get_settings
    settings = get_settings()

    if mode == "live" and not settings.live_trading_enabled:
        raise HTTPException(
            status_code=403,
            detail="Live trading is disabled. Complete Phase 0 regulatory review first.",
        )

    result = await db.execute(select(Strategy).where(Strategy.id == strategy_id))
    strategy = result.scalar_one_or_none()
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")

    strategy.status = StrategyStatus.PAPER if mode == "paper" else StrategyStatus.LIVE
    await db.commit()
    logger.warning("Strategy activated", strategy=strategy.name, mode=mode)
    return {"id": str(strategy.id), "status": strategy.status.value}


@strategy_router.patch("/{strategy_id}/deactivate")
async def deactivate_strategy(
    strategy_id: str,
    db: AsyncSession = Depends(get_db),
    _user=Depends(get_current_user),
):
    result = await db.execute(select(Strategy).where(Strategy.id == strategy_id))
    strategy = result.scalar_one_or_none()
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    strategy.status = StrategyStatus.INACTIVE
    await db.commit()
    return {"id": str(strategy.id), "status": "INACTIVE"}


def _strategy_out(s: Strategy) -> StrategyOut:
    from datetime import datetime
    return StrategyOut(
        id=str(s.id),
        name=s.name,
        description=s.description,
        status=s.status.value,
        config=s.config or {},
        created_at=s.created_at.isoformat() if s.created_at else "",
    )
