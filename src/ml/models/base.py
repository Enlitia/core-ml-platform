from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class BaseModel(ABC, BaseEstimator):
    """Abstract base class for models."""

    def __init__(self, params: dict | None = None) -> None:
        """
        Initialize base model attributes.

        Args:
            params: Model-specific hyperparameters and configuration options.

        Attributes must be set by subclasses:
            model_name: The name of the model (e.g., 'kd3', 'positive_linear', 'xgboost').
                      Used for model identification and MLflow tracking.
        """
        self.model_name: str
        self.params = params if params is not None else {}

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        """Fit the model."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        pass

    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Evaluate model."""
        pass
