"""Configuration for Advanced Power Forecast task."""

from ml.tasks.base import BaseTaskConfig


class AdvancedPowerForecastConfig(BaseTaskConfig):
    """Configuration for the Advanced Power Forecast task.

    Supports multiple models: positive_linear, xgboost, random_forest.
    Model selection happens at runtime via CLI --model flag or uses default.
    """

    # Task Configuration
    task_name: str = "advanced_power_forecast"

    # Model Configuration
    default_model_name: str = "positive_linear"
    available_models: list[str] = ["positive_linear", "xgboost", "random_forest"]

    # Model-specific parameters (optional overrides for each model)
    model_params: dict[str, dict] = {
        # "xgboost": {"n_estimators": 200, "max_depth": 8},  # Example
        # "random_forest": {"n_estimators": 150},  # Example
    }

    # Training parameters
    training_interval: str = "1 year"
    test_size: float = 0.2
    random_state: int = 42

    # Prediction parameters
    prediction_days: int = 15
    delta_minutes: int = 15

    # Data Validation parameters
    power_min: float = 0.0
    power_max: float = 1_000_000.0

    # Model Quality Thresholds
    min_size_train: int = 100

    model_quality_thresholds: dict[str, dict[str, float]] = {
        "mae": {"max": 10_000.0},
        "rmse": {"max": 15_000.0},
    }
