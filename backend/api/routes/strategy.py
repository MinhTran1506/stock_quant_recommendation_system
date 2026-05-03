"""
api/routes/strategy.py — re-exports strategy_router from universe module.
"""
from api.routes.universe import strategy_router as router

__all__ = ["router"]
