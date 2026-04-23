"""Task configuration registry."""

from typing import Any, Callable

from core_ml.tasks.advanced_power_forecast.config import AdvancedPowerForecastConfig
from core_ml.tasks.advanced_power_forecast.predict import predict as apf_predict
from core_ml.tasks.advanced_power_forecast.train import train as apf_train

TASK_CONFIG_REGISTRY: dict[str, Any] = {
    "advanced_power_forecast": AdvancedPowerForecastConfig(),
}

TASK_HANDLERS: dict[str, dict[str, Callable]] = {
    "advanced_power_forecast": {
        "train": apf_train,
        "predict": apf_predict,
    },
    # Add more tasks here as they're implemented
}


def get_task_config(task_name: str) -> Any:
    """Get task configuration by task name."""
    return TASK_CONFIG_REGISTRY[task_name]


def get_task_handler(task_name: str, operation: str) -> Callable:
    """Get task handler function for train or predict operations.

    Args:
        task_name: Name of the task (e.g., 'advanced_power_forecast')
        operation: Either 'train' or 'predict'

    Returns:
        Callable function for the requested operation

    Raises:
        KeyError: If task or operation not found
    """
    if task_name not in TASK_HANDLERS:
        available = ", ".join(TASK_HANDLERS.keys())
        raise KeyError(f"No handlers found for task '{task_name}'. Available: {available}")

    if operation not in TASK_HANDLERS[task_name]:
        available = ", ".join(TASK_HANDLERS[task_name].keys())
        raise KeyError(f"Operation '{operation}' not found for task '{task_name}'. Available: {available}")

    return TASK_HANDLERS[task_name][operation]
