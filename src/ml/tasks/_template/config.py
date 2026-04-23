"""Configuration for Template task - copy and adapt for your task."""

from ml.tasks.base import BaseTaskConfig


class TemplateTaskConfig(BaseTaskConfig):
    """Configuration for the Template task.

    Copy this file when creating a new task:
    1. Rename class to match your task (e.g., YourTaskConfig)
    2. Set task_name to match your folder name
    3. Choose which model to use
    4. Add/remove task-specific parameters as needed
    """

    # Required: Task and Model Configuration
    task_name: str = "your_task_name"  # TODO: Change to match folder name
    model_name: str = "positive_linear"  # TODO: Choose model from MODEL_REGISTRY

    # Optional: Training parameters
    training_interval: str = "1 year"
    min_size_train: int = 100
    test_size: float = 0.2
    random_state: int = 42

    # Optional: Prediction parameters
    prediction_days: int = 15
    delta_minutes: int = 15

    # Optional: Data Validation parameters
    power_min: float = 0.0
    power_max: float = 1_000_000.0

    # TODO: Add your task-specific parameters here
