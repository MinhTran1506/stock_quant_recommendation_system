"""
data/ingestion/fiingroup.py — Backward-compatibility shim.

FiinGroup requires an enterprise API subscription that is not yet available.
All data ingestion is now handled by VnstockProvider (open-source, free).

This module re-exports VnstockProvider as FiinGroupProvider so that any
existing import continues to work without modification.

When a FiinGroup subscription is obtained, replace this file with a
dedicated connector following the BaseDataProvider interface in base.py.
"""
from data.ingestion.vnstock_provider import VnstockProvider as FiinGroupProvider  # noqa: F401

__all__ = ["FiinGroupProvider"]
