"""ML Model queries."""

from base_models.machine_learning import MLModelType
from toolkit.database import database


def get_ml_model_type_id(model_name: str) -> int:
    """Get model_type_id from model_name by querying ml_model_type table.

    Args:
        model_name: Name of the ML model type (e.g., 'xgboost', 'positive_linear')

    Returns:
        int: The model_type.id from machine_learning.ml_model_type table

    Raises:
        ValueError: If model_name not found in ml_model_type table
    """
    with database.session() as session:
        model_type = session.query(MLModelType).filter(MLModelType.name == model_name).first()
        if not model_type:
            raise ValueError(
                f"Model type '{model_name}' not found in ml_model_type table. "
                f"Please ensure the model type exists in machine_learning.ml_model_type"
            )
        return int(model_type.id)
