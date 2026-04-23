import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from core_ml.models.base import BaseModel


class PositiveLinearModel(BaseModel):
    """Positive LinearRegression for Advanced Power Forecast."""

    def __init__(self, params: dict | None = None) -> None:
        self.model_name = "positive_linear"
        self.params = params or {}
        self.model = LinearRegression(positive=True)
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PositiveLinearModel":
        self.feature_names_ = X.columns.tolist()
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        prediction: np.ndarray = self.model.predict(X)
        return prediction

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        predictions = self.predict(X)
        return {
            "mae": round(mean_absolute_error(y, predictions), 2),
            "rmse": round(root_mean_squared_error(y, predictions), 2),
        }

    def get_feature_weights(self) -> dict[str, float]:
        """Get feature importance as percentage contribution.

        Returns:
            Dictionary mapping feature names to their percentage contribution (0-100),
            sorted by importance (descending). Shows how much each feature matters.

        Raises:
            ValueError: If model hasn't been fitted yet.
        """
        if not hasattr(self.model, "coef_") or not self.feature_names_:
            raise ValueError("Model must be fitted before getting feature weights")

        coefficients = self.model.coef_
        total_coef = np.sum(np.abs(coefficients))

        if total_coef == 0:
            # Edge case: all coefficients are zero
            return {name: 0.0 for name in self.feature_names_}

        # Calculate percentage contribution
        weights = {}
        for name, coef in zip(self.feature_names_, coefficients):
            weight_pct = (np.abs(coef) / total_coef) * 100
            weights[name] = round(weight_pct, 1)

        # Sort by importance (descending)
        weights = dict(sorted(weights.items(), key=lambda x: x[1], reverse=True))
        return weights
