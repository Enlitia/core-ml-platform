import logging
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from toolkit.configuration import configuration

from ml.models.base import BaseModel

# Suppress MLflow's verbose logging
logging.getLogger("mlflow").setLevel(logging.WARNING)


class MLflowGateway:
    """Simple gateway for MLflow operations."""

    def __init__(self, task_name: str):
        """Initialize MLflow configuration.

        Args:
            task: The ML task/problem type (e.g., 'advanced_power_forecast')
        """
        self.tracking_uri = configuration.settings.ml_flow_tracking_uri
        self.client_name = configuration.settings.client_name
        self.task_name = task_name

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.task_name)
        self.mlflow_client: mlflow.tracking.MlflowClient = mlflow.tracking.MlflowClient()

    def save_model(
        self,
        model: BaseModel,
        input_example: pd.DataFrame,
        asset_id: int | None = None,
        metrics: dict[str, float] | None = None,
        log_params: dict[str, Any] | None = None,
    ) -> None:
        """Save trained model to MLflow Model Registry."""
        run_name = f"{model.model_name}_client_{self.client_name}"
        if asset_id is not None:
            run_name += f"_asset_{asset_id}"

        with mlflow.start_run(run_name=run_name):
            # Tags for metadata/identifiers
            mlflow.set_tag("client_name", self.client_name)
            mlflow.set_tag("model_name", model.model_name)
            if asset_id is not None:
                mlflow.set_tag("asset_id", asset_id)

            # Log any additional parameters
            if log_params is not None:
                for param_name, param_value in log_params.items():
                    mlflow.log_param(param_name, param_value)

            # Metrics
            if metrics is not None:
                mlflow.log_metrics(metrics)

            # Register model (creates new version under model_name name)
            model_info = mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=run_name,
                input_example=input_example,
            )

        # Set alias to champion (replaces deprecated Production stage)
        model_version = model_info.registered_model_version
        self.mlflow_client.set_registered_model_alias(name=run_name, alias="champion", version=model_version)

    def load_model(self, model_name: str, asset_id: int | None = None) -> tuple[BaseModel, dict[str, Any]]:
        """Load trained model from MLflow Model Registry."""
        registered_model_name = f"{model_name}_client_{self.client_name}"
        if asset_id is not None:
            registered_model_name += f"_asset_{asset_id}"

        # Load model by alias (replaces deprecated Production stage)
        try:
            model_uri = f"models:/{registered_model_name}@champion"
            trained_model = mlflow.sklearn.load_model(model_uri)

            # Get model version info to retrieve run_id and params
            model_version_info = self.mlflow_client.get_model_version_by_alias(registered_model_name, "champion")
            run = self.mlflow_client.get_run(model_version_info.run_id)

            # Retrieve all logged parameters
            params = {key: eval(value) if value.startswith("[") else value for key, value in run.data.params.items()}

            return trained_model, params
        except mlflow.exceptions.RestException:
            raise ValueError(f"No champion model found for {registered_model_name}")
