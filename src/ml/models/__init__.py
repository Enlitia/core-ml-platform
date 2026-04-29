from .base import BaseModel
from .positive_linear import PositiveLinearModel
from .random_forest_model import RandomForestModel
from .xgboost_model import XGBoostModel

MODEL_REGISTRY = {
    "positive_linear": PositiveLinearModel,
    "xgboost": XGBoostModel,
    "random_forest": RandomForestModel,
}


def get_model(model_type: str, params: dict | None = None) -> BaseModel:
    """Get a model instance by model type.

    Args:
        model_type: Type of the model to instantiate
        params: Optional parameters to pass to the model constructor

    Returns:
        Instantiated model
    """
    return MODEL_REGISTRY[model_type](params=params)
