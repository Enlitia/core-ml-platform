import pandas as pd
import typer

from ml.common.assets import convert_input_from_df_to_dict, select_only_valid_asset_ids
from ml.common.split_data import split_data_by_day
from ml.common.validations import validate_inputs_training, validate_model_quality, validate_out_of_range
from ml.context import Context, get_context
from ml.models import BaseModel, get_model
from ml.queries.advanced_power_forecast import (
    fetch_power_forecast_data_for_train,
    fetch_power_real_data_for_train,
    get_ml_model_id,
)
from ml.tasks.advanced_power_forecast.utils.preprocess import preprocess_power_forecast_data

app = typer.Typer()


def preprocess_power_real_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess real data by converting date columns to datetime."""

    df["prediction_date"] = pd.to_datetime(df["prediction_date"], utc=True)
    return df


def get_training_inputs_all_assets(list_asset_ids: list[int], context: Context) -> dict[int, pd.DataFrame]:
    """
    Get Data to train Datasets.
    Build a dict[asset_id: input df].
    """
    # Forecast Data
    df_power_forecast = fetch_power_forecast_data_for_train(list_asset_ids, context.task_config.training_interval)
    df_power_forecast = preprocess_power_forecast_data(df_power_forecast)

    # Real Data
    df_power_real = fetch_power_real_data_for_train(list_asset_ids, context.task_config.training_interval)
    df_power_real = preprocess_power_real_data(df_power_real)

    # Merge
    df_input_all_assets = df_power_real.merge(df_power_forecast, on=["asset_id", "prediction_date"], how="inner")

    # Sort
    df_input_all_assets = df_input_all_assets.sort_values(
        ["asset_id", "available_date", "prediction_date"]
    ).reset_index(drop=True)

    # one df with all assets -> dict of one df per asset_id
    dict_inputs_all_assets = convert_input_from_df_to_dict(df_input_all_assets, list_asset_ids)
    return dict_inputs_all_assets


def preprocess_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Process raw data into features and target."""
    # Select features
    providers = df.columns.difference(["asset_id", "available_date", "prediction_date", "power_real"]).tolist()
    X = df.set_index(["available_date", "prediction_date"])[providers].copy()

    # Drop cols and rows with all NaN values
    X.dropna(axis=1, how="all", inplace=True)
    X.dropna(axis=0, how="all", inplace=True)

    # Fillna power with row avg
    X.loc[:, providers] = X[providers].apply(lambda row: row.fillna(row.mean()), axis=1)

    # Select target
    y = df.set_index(["available_date", "prediction_date"])["power_real"].copy()
    return X, y


def save_model(
    model: BaseModel,
    context: Context,
    asset_id: int,
    X: pd.DataFrame,
    metrics: dict[str, float],
    model_params: dict | None = None,
) -> None:
    """Save trained model to MLflow Model Registry."""
    providers: list[str] = X.columns.tolist()
    input_example: pd.DataFrame = X.head(1)

    # Get model_id from model_name (will be used when saving predictions)
    model_id = get_ml_model_id(context.model_name)

    log_params = {"model_id": model_id, "providers": providers, **(model_params or {})}

    context.mlflow_gateway.save_model(
        model=model,
        input_example=input_example,
        asset_id=asset_id,
        metrics=metrics,
        log_params=log_params,
    )


def train_one_asset(data: pd.DataFrame, asset_id: int, context: Context) -> None:
    context.logger.info(f"Training asset {asset_id} with {len(data)} samples")

    # Get model-specific params from config (if any)
    model_params = context.task_config.model_params.get(context.model_name)
    model = get_model(context.model_name, params=model_params)

    # Validations
    validate_out_of_range(data, "power_real", 0, context.task_config.power_max, asset_id)

    X, y = preprocess_training_data(data)

    validate_inputs_training(X, y, context.task_config.min_size_train, asset_id)

    dates = pd.Series(X.index.get_level_values("prediction_date"))
    X_train, X_test, y_train, y_test = split_data_by_day(
        X, y, dates, context.task_config.test_size, context.task_config.random_state
    )

    model.fit(X_train, y_train)

    metrics = model.evaluate(X_test, y_test)
    context.logger.info(f"Asset {asset_id} - Metrics: {metrics}")

    validate_model_quality(metrics, context.task_config.model_quality_thresholds, asset_id)

    save_model(model, context, asset_id, X, metrics, model_params)
    context.logger.info(f"Model saved for asset {asset_id}")


@app.command()
def train(
    asset_ids: str = "all",
    task_name: str = "advanced_power_forecast",
    model_name: str | None = None,
) -> None:
    # Context: Model Name, Task Config, Logger, MLflow Gateway
    context = get_context(task_name=task_name, model_name=model_name)

    list_asset_ids = select_only_valid_asset_ids(asset_ids)
    context.logger.info(
        f"Starting training for {len(list_asset_ids)} asset(s): {list_asset_ids} using model '{context.model_name}'"
    )

    dict_inputs_all_assets = get_training_inputs_all_assets(list_asset_ids, context)

    for asset_id, data in dict_inputs_all_assets.items():
        try:
            train_one_asset(data, asset_id, context)
        except Exception as e:
            context.logger.error(f"Error training asset {asset_id}: {e}")

    context.logger.info("Training completed for all assets")


if __name__ == "__main__":
    train()
