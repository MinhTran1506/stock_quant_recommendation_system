"""
models/base.py — Abstract base class for all forecasting models.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class ModelMetrics:
    """Standardised metrics container returned from model training."""
    model_name: str = ""
    run_id: str = ""
    mae: float = 0.0
    rmse: float = 0.0
    directional_accuracy: float = 0.0
    sharpe_ratio: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseModel(ABC):
    """Abstract base for all forecasting models in the platform."""

    MODEL_NAME: str = "base"

    def __init__(self):
        self._run_id: Optional[str] = None

    @abstractmethod
    def train(self, train_df: pd.DataFrame, **kwargs) -> ModelMetrics:
        """Train the model and return evaluation metrics."""

    @abstractmethod
    def predict(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate forecasts. Returns DataFrame with predictions."""

    @classmethod
    @abstractmethod
    def load(cls, mlflow_run_id: str) -> "BaseModel":
        """Load a saved model from MLflow."""

    @property
    def run_id(self) -> Optional[str]:
        return self._run_id
