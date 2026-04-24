from datetime import datetime

import pandas as pd
import typer

from ml.common.assets import convert_input_from_df_to_dict, select_only_valid_asset_ids
from ml.common.dates import get_dates
from ml.common.queries import fetch_power_forecast_data_for_prediction, save_advanced_power_forecast_predictions
from ml.common.validations import validate_inputs_prediction, validate_not_empty
from ml.context import Context, get_context
from ml.tasks.advanced_power_forecast.utils.preprocess import preprocess_power_forecast_data

app = typer.Typer()


def get_prediction_inputs_all_assets(
    list_asset_ids: list[int],
    start_date: datetime,
    end_date: datetime,
    delta_minutes: int,
) -> dict[int, pd.DataFrame]:
    """Get and preprocess data for prediction."""
    df_power_forecast = fetch_power_forecast_data_for_prediction(list_asset_ids, start_date, end_date, delta_minutes)
    df_power_forecast = preprocess_power_forecast_data(df_power_forecast)

    # Set index to prediction_date and cols to providers
    df_input_all_assets = df_power_forecast.drop(columns=["available_date"]).set_index("prediction_date")

    # one df with all assets -> dict of one df per asset_id
    dict_input_all_assets = convert_input_from_df_to_dict(df_input_all_assets, list_asset_ids)
    return dict_input_all_assets


def preprocess_prediction_data(df: pd.DataFrame, providers: list[str]) -> pd.DataFrame:
    """Preprocess power forecast data for prediction."""
    X = df.drop(columns=["asset_id"]).copy()

    X = X[providers]

    # Fillna power with row avg, then with 0 if row avg is NaN
    X = X.apply(lambda row: row.fillna(row.mean()), axis=1)
    X = X.fillna(0)
    return X  # type: ignore[no-any-return]


def predict_one_asset(
    data: pd.DataFrame,
    asset_id: int,
    start_date: datetime,
    context: Context,
) -> pd.DataFrame:
    # Define output columns
    output_cols = [
        "available_date",
        "prediction_date",
        "asset_id",
        "model_name",
        "providers",
        "model_params",
        "prediction",
    ]

    validate_not_empty(data, asset_id)

    context.logger.info(f"Predicting for asset {asset_id} with {len(data)} timestamps")

    # Get Model
    model, params = context.mlflow_gateway.load_model(context.model_name, asset_id)
    providers = params.get("providers", [])

    # Preprocess
    X = preprocess_prediction_data(data, providers)

    validate_inputs_prediction(X, asset_id)

    # Generate predictions
    predictions = model.predict(X)

    # Create output df
    df_output = pd.DataFrame(
        data={
            "available_date": start_date,
            "prediction_date": X.index.values,
            "asset_id": asset_id,
            "model_name": context.model_name,
            "providers": str(providers),
            "model_params": [params] * len(predictions),
            "prediction": predictions,
        },
        columns=output_cols,
    )
    return df_output


@app.command()
def predict(
    asset_ids: str = "all",
    task_name: str = "advanced_power_forecast",
    model_name: str | None = None,
    start_date: datetime | None = None,
) -> pd.DataFrame:
    """Generate predictions for a specific asset.

    Args:
        asset_ids: Asset identifier(s) - single int, comma-separated, or 'all'
        task_name: Task name
        model_name: Model to use (optional, uses config default if not provided)
        start_date: Start datetime (default: now)

    Returns:
        DataFrame with timestamps and predictions
    """
    context = get_context(task_name=task_name, model_name=model_name)

    list_asset_ids = select_only_valid_asset_ids(asset_ids)

    start_date, end_date, delta_minutes = get_dates(
        start_date=start_date,
        prediction_days=context.task_config.prediction_days,
        delta_minutes=context.task_config.delta_minutes,
    )

    context.logger.info(
        f"Starting predictions for {len(list_asset_ids)} asset(s): {list_asset_ids} "
        f"using model '{context.model_name}'"
    )

    dict_inputs_all_assets = get_prediction_inputs_all_assets(list_asset_ids, start_date, end_date, delta_minutes)

    list_df_prediction = []
    for asset_id, data in dict_inputs_all_assets.items():
        try:
            df_prediction = predict_one_asset(data, asset_id, start_date, context)
            if len(df_prediction) > 0:
                list_df_prediction.append(df_prediction)
        except Exception as e:
            context.logger.error(f"Error predicting asset {asset_id}: {e}")

    df_all_predictions = pd.concat(list_df_prediction, ignore_index=True)

    context.logger.info(f"Predictions completed: {len(df_all_predictions)} total predictions")

    # Save predictions to database
    if len(df_all_predictions) > 0:
        save_advanced_power_forecast_predictions(df_all_predictions)
        context.logger.info(f"Saved {len(df_all_predictions)} predictions to database")

    return df_all_predictions


if __name__ == "__main__":
    predict()
