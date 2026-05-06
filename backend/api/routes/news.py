"""
api/routes/news.py — News and sentiment endpoints.
"""
from typing import List, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.middleware.auth import get_current_user
from db.models import NewsArticle
from db.session import get_db

router = APIRouter()
logger = structlog.get_logger(__name__)


# ─── Schemas ──────────────────────────────────────────────────────────────────
class NewsArticleOut(BaseModel):
    id: str
    ticker: Optional[str]
    headline: str
    source: Optional[str]
    sentiment_score: Optional[float]
    event_type: Optional[str]
    published_at: Optional[str]

    class Config:
        from_attributes = True


# ─── Endpoints ────────────────────────────────────────────────────────────────
@router.get("", response_model=List[NewsArticleOut])
async def list_news(
    ticker: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
    _: UUID = Depends(get_current_user),
):
    """Return recent news articles, optionally filtered by ticker."""
    q = select(NewsArticle).order_by(NewsArticle.published_at.desc()).limit(limit)
    if ticker:
        q = q.where(NewsArticle.ticker == ticker.upper())
    result = await db.execute(q)
    articles = result.scalars().all()
    return [
        NewsArticleOut(
            id=str(a.id),
            ticker=a.ticker,
            headline=a.headline,
            source=a.source,
            sentiment_score=float(a.sentiment_score) if a.sentiment_score is not None else None,
            event_type=a.event_type,
            published_at=str(a.published_at) if a.published_at else None,
        )
        for a in articles
    ]
