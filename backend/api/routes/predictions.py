"""
api/routes/predictions.py — Model prediction and stock ranking endpoints.
"""
from datetime import date, datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from api.middleware.auth import get_current_user
from db.models import Prediction, Stock, ModelVersion
from db.session import get_db

router = APIRouter()
logger = structlog.get_logger(__name__)


# ─── Schemas ──────────────────────────────────────────────────────────────────
class PredictionOut(BaseModel):
    id: str
    ticker: str
    generated_at: str
    target_date: str
    horizon_days: int
    predicted_return: Optional[float]
    predicted_price: Optional[float]
    confidence_lower: Optional[float]
    confidence_upper: Optional[float]
    score: Optional[float]
    feature_importances: Optional[Dict[str, float]]
    model_name: Optional[str]


class RankingEntry(BaseModel):
    rank: int
    ticker: str
    name: str
    sector: Optional[str]
    score: float
    predicted_return_5d: Optional[float]
    sentiment_score: Optional[float]
    top_features: Optional[Dict[str, float]]
    current_price: Optional[float]
    change_pct_1d: Optional[float]


# ─── Endpoints ────────────────────────────────────────────────────────────────
@router.get("/rankings", response_model=List[RankingEntry])
async def get_rankings(
    date_: Optional[date] = Query(None, alias="date", description="Prediction date (default: latest)"),
    horizon: int = Query(5, description="Horizon in trading days: 1 | 3 | 5 | 10 | 20"),
    top_n: int = Query(50, le=200),
    sector: Optional[str] = Query(None),
    exchange: Optional[str] = Query(None),
    min_score: float = Query(0.0, ge=0, le=100),
    db: AsyncSession = Depends(get_db),
    _user=Depends(get_current_user),
):
    """
    Return the top-ranked stocks by meta-model score for a given date and horizon.
    Includes explainability (top SHAP features) for each stock.
    """
    # Get latest prediction date if not specified
    if date_ is None:
        latest = await db.execute(
            select(Prediction.generated_at).order_by(desc(Prediction.generated_at)).limit(1)
        )
        latest_ts = latest.scalar_one_or_none()
        if not latest_ts:
            return []
        date_ = latest_ts.date()

    # Build query
    q = (
        select(Prediction, Stock)
        .join(Stock, Prediction.stock_id == Stock.id)
        .where(
            Prediction.horizon_days == horizon,
            Prediction.score >= min_score,
        )
        .order_by(desc(Prediction.score))
        .limit(top_n)
    )
    if sector:
        q = q.where(Stock.sector.ilike(f"%{sector}%"))
    if exchange:
        q = q.where(Stock.exchange == exchange.upper())

    result = await db.execute(q)
    rows = result.all()

    rankings = []
    for i, (pred, stock) in enumerate(rows):
        rankings.append(RankingEntry(
            rank=i + 1,
            ticker=stock.ticker,
            name=stock.name,
            sector=stock.sector,
            score=round(float(pred.score or 0), 2),
            predicted_return_5d=pred.predicted_return,
            sentiment_score=pred.raw_outputs.get("sentiment_score") if pred.raw_outputs else None,
            top_features=pred.feature_importances,
            current_price=pred.raw_outputs.get("current_price") if pred.raw_outputs else None,
            change_pct_1d=pred.raw_outputs.get("change_pct_1d") if pred.raw_outputs else None,
        ))
    return rankings


@router.get("/{ticker}", response_model=List[PredictionOut])
async def get_predictions(
    ticker: str,
    horizon: Optional[int] = Query(None, description="Filter by horizon days"),
    limit: int = Query(30, le=200),
    db: AsyncSession = Depends(get_db),
    _user=Depends(get_current_user),
):
    """Return recent predictions for a single stock across all horizons."""
    stock_result = await db.execute(
        select(Stock).where(Stock.ticker == ticker.upper())
    )
    stock = stock_result.scalar_one_or_none()
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock {ticker} not found")

    q = (
        select(Prediction, ModelVersion)
        .outerjoin(ModelVersion, Prediction.model_version_id == ModelVersion.id)
        .where(Prediction.stock_id == stock.id)
        .order_by(desc(Prediction.generated_at))
        .limit(limit)
    )
    if horizon:
        q = q.where(Prediction.horizon_days == horizon)

    result = await db.execute(q)
    rows = result.all()

    return [
        PredictionOut(
            id=str(pred.id),
            ticker=ticker.upper(),
            generated_at=pred.generated_at.isoformat(),
            target_date=pred.target_date.isoformat(),
            horizon_days=pred.horizon_days,
            predicted_return=pred.predicted_return,
            predicted_price=pred.predicted_price,
            confidence_lower=pred.confidence_lower,
            confidence_upper=pred.confidence_upper,
            score=float(pred.score) if pred.score is not None else None,
            feature_importances=pred.feature_importances,
            model_name=mv.name if mv else None,
        )
        for pred, mv in rows
    ]


@router.get("/models/registry", response_model=List[Dict])
async def list_model_versions(
    champion_only: bool = Query(False),
    db: AsyncSession = Depends(get_db),
    _user=Depends(get_current_user),
):
    """List all registered model versions from the model registry."""
    q = select(ModelVersion).order_by(desc(ModelVersion.trained_at))
    if champion_only:
        q = q.where(ModelVersion.is_champion == True)

    result = await db.execute(q)
    versions = result.scalars().all()

    return [
        {
            "id": str(v.id),
            "name": v.name,
            "version": v.version,
            "model_type": v.model_type,
            "horizon_days": v.horizon_days,
            "metrics": v.metrics,
            "is_champion": v.is_champion,
            "trained_at": v.trained_at.isoformat(),
            "mlflow_run_id": v.mlflow_run_id,
        }
        for v in versions
    ]
