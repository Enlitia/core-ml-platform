"""ML Model queries."""

from base_models.machine_learning import MLModel
from toolkit.database import database


def get_ml_model_id(model_name: str) -> int:
    """Get model_id from model_name by querying ml_model table.

    Args:
        model_name: Name of the ML model (e.g., 'xgboost', 'positive_linear')

    Returns:
        int: The model.id from machine_learning.ml_model table

    Raises:
        ValueError: If model_name not found in ml_model table
    """
    with database.session() as session:
        model = session.query(MLModel).filter(MLModel.name == model_name).first()
        if not model:
            raise ValueError(
                f"Model '{model_name}' not found in ml_model table. "
                f"Please ensure the model exists in machine_learning.ml_model"
            )
        return int(model.id)
