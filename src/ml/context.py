"""Context management for ML tasks - provides task-specific dependencies.

Note: Infrastructure settings (DB, MLflow URI, table names) are accessed via
toolkit.configuration.settings throughout the codebase. This Context only
manages task-specific config and utilities.
"""

from dataclasses import dataclass
from typing import Any

from toolkit.logging import StructuredLogger, get_logger

from ml.infrastructure.ml_flow import MLflowGateway


@dataclass
class Context:
    """Context object for ML task execution.

    Provides task-specific configuration and utilities needed during
    train/predict operations.

    Note: For infrastructure settings (DB host, table names, etc.),
    use toolkit.configuration.settings directly.
    """

    model_name: str  # Active model name (resolved from CLI or config default)
    model_type_id: int  # Model type ID from ml_model_type table (for saving predictions)
    task_config: Any  # Task-specific ML parameters (training_interval, etc.)
    logger: StructuredLogger  # Logger instance for this task
    mlflow_gateway: MLflowGateway  # MLflow interface for model storage


def get_context(task_name: str, model_name: str | None = None) -> Context:
    """Initialize context for ML task execution.

    Args:
        task_name: Name of the task (e.g., 'advanced_power_forecast')
        model_name: Model name override (optional, uses config default if not provided)

    Returns:
        Context with task config, model name, logger, and MLflow gateway initialized

    Example:
        context = get_context("advanced_power_forecast")
        context = get_context("advanced_power_forecast", model_name="xgboost")
    """
    # Import here to avoid circular import
    from ml.queries.ml_models import get_ml_model_type_id
    from ml.tasks import get_task_config

    task_config = get_task_config(task_name)

    # Resolve model name (CLI override or config default)
    resolved_model_name = model_name or task_config.default_model_name

    # Validate model is available for this task
    if resolved_model_name not in task_config.available_models:
        raise ValueError(
            f"Invalid model '{resolved_model_name}' for task '{task_name}'. Available: {task_config.available_models}"
        )

    # Get model_type_id from model_name (used when saving predictions)
    model_type_id = get_ml_model_type_id(resolved_model_name)

    logger = get_logger(task_config.task_name)
    mlflow_gateway = MLflowGateway(task_config.task_name)

    return Context(
        model_name=resolved_model_name,
        model_type_id=model_type_id,
        task_config=task_config,
        logger=logger,
        mlflow_gateway=mlflow_gateway,
    )
