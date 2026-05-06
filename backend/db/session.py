"""
db/session.py — Async SQLAlchemy session factory and lifecycle management.
"""
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
import structlog

from config import get_settings

settings = get_settings()
logger = structlog.get_logger(__name__)

# Global engine — created once at startup
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker | None = None


async def init_db() -> None:
    """Initialise async engine and session factory. Called at app startup."""
    global _engine, _session_factory

    _engine = create_async_engine(
        settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout,
        pool_pre_ping=True,
        echo=settings.debug,
    )
    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )
    logger.info("Database connection pool initialised")


async def close_db() -> None:
    """Dispose connection pool on shutdown."""
    global _engine
    if _engine:
        await _engine.dispose()
        logger.info("Database connection pool closed")


def AsyncSessionLocal() -> AsyncSession:
    """
    Return a new AsyncSession for use outside of FastAPI request scope.
    The caller is responsible for committing and closing.

    Usage::
        async with AsyncSessionLocal() as session:
            ...
            await session.commit()
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialised. Call init_db() first.")
    return _session_factory()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency — yields an async DB session per request.
    Automatically rolls back on exception and closes the session.

    Usage:
        @router.get("/items")
        async def list_items(db: AsyncSession = Depends(get_db)):
            ...
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialised. Call init_db() first.")

    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
