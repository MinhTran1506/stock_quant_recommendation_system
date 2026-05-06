"""
data/ingestion/vietstock.py — Backward-compatibility shim.

Vietstock requires a paid API subscription that is not yet available.
All data ingestion is now handled by VnstockProvider (open-source, free).

This module re-exports VnstockProvider as VietstockProvider so that any
existing import continues to work without modification.

When a Vietstock subscription is obtained, replace this file with a
dedicated connector following the BaseDataProvider interface in base.py.
"""
from data.ingestion.vnstock_provider import VnstockProvider as VietstockProvider  # noqa: F401

__all__ = ["VietstockProvider"]
