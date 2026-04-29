"""ML Model queries."""

from base_models.machine_learning import MLModelType
from toolkit.database import database


def get_ml_model_type_id(model_type: str) -> int:
    """Get model_type_id from model_type by querying ml_model_type table.

    Args:
        model_type: Type of the ML model (e.g., 'xgboost', 'positive_linear')

    Returns:
        int: The model_type.id from machine_learning.ml_model_type table

    Raises:
        ValueError: If model_type not found in ml_model_type table
    """
    with database.session() as session:
        model_type_row = session.query(MLModelType).filter(MLModelType.name == model_type).first()
        if not model_type_row:
            raise ValueError(
                f"Model type '{model_type}' not found in ml_model_type table. "
                f"Please ensure the model type exists in machine_learning.ml_model_type"
            )
        return int(model_type_row.id)
