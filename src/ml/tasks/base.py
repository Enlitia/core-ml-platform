"""Base configuration class for ML tasks."""

from abc import ABC

from pydantic_settings import BaseSettings


class BaseTaskConfig(BaseSettings, ABC):
    """Abstract base class for task configurations.

    Attributes must be set by subclasses:
        task_name: The folder name where the task code lives (e.g., 'advanced_power_forecast').
                   Used for task identification, logging, and MLflow tracking.
        default_model_type: Default model to use from MODEL_REGISTRY (e.g., 'positive_linear').
                           Can be overridden at runtime via CLI --model flag.
        available_models: List of valid model types for this task.
        model_params: Optional dict of model-specific parameters {model_type: {param: value}}.

    All other parameters are optional and task-specific (e.g., training_interval,
    prediction_days, validation thresholds).
    """

    # Required: Task and Model identification
    task_name: str  # Must match the task folder name
    default_model_type: str  # Default model (overridable via CLI)
    available_models: list[str]  # Valid models for this task
    model_params: dict[str, dict] = {}  # Optional model-specific params
