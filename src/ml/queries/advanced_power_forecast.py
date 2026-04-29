from datetime import datetime

import pandas as pd
from base_models.machine_learning import AdvancedPowerForecastData
from toolkit.data.query import Query
from toolkit.database import database


# Train Advanced Power Forecast


def fetch_power_forecast_data_for_train(list_asset_ids: list[int], training_interval: str) -> pd.DataFrame:
    """Get forecast data for specific assets."""
    sql_query = """
        SELECT
            asset_id,
            available_date,
            prediction_date,
            provider_id,
            forecast_value AS power_forecast
        FROM
            data_lake.power_forecast_data
        WHERE available_date > NOW() - INTERVAL ':training_interval'
        ORDER BY 1, 2, 3, 4
    """
    query = Query(sql_query)
    query.with_parameter("training_interval", training_interval)
    if list_asset_ids:
        query.with_in("asset_id", list_asset_ids)

    with database.session() as session:
        result = session.execute(query.prepared_statement).mappings().fetchall()
    result = pd.DataFrame(result)
    return result


def fetch_power_real_data_for_train(list_asset_ids: list[int], training_interval: str) -> pd.DataFrame:
    """Get real data for specific assets."""
    sql_query = """
        SELECT
            farm_id AS asset_id,
            read_at AS prediction_date,
            power_real
        FROM
            data_warehouse.farm_data_power_real
        WHERE read_at > NOW() - INTERVAL ':training_interval'
        ORDER BY 1, 2, 3
    """
    query = Query(sql_query)
    query.with_parameter("training_interval", training_interval)
    if list_asset_ids:
        query.with_in("farm_id", list_asset_ids)

    with database.session() as session:
        result = session.execute(query.prepared_statement).mappings().fetchall()
    result = pd.DataFrame(result)
    return result


# Predict Advanced Power Forecast


def fetch_power_forecast_data_for_prediction(
    list_asset_ids: list[int],
    start_date: datetime,
    end_date: datetime,
    delta_minutes: int,
) -> pd.DataFrame:
    """Get forecast data for a specific asset."""
    sql_query = """
    WITH forecast_data AS (
        SELECT
            asset_id,
            available_date,
            prediction_date,
            provider_id,
            forecast_value AS power_forecast,
            MAX(available_date) OVER (PARTITION BY asset_id, prediction_date, provider_id) AS max_available_date
        FROM
            data_lake.power_forecast_data
        WHERE
            EXTRACT(MINUTE FROM prediction_date) % :delta_minutes = 0
    )

    SELECT
        asset_id,
        available_date,
        prediction_date,
        provider_id,
        power_forecast
    FROM
        forecast_data
    WHERE
        available_date = max_available_date
    """
    query = Query(sql_query)
    query.with_parameter("delta_minutes", delta_minutes)
    query.with_date(str(start_date), "prediction_date", ">=")
    query.with_date(str(end_date), "prediction_date", "<=")
    if list_asset_ids:
        query.with_in("asset_id", list_asset_ids)

    with database.session() as session:
        result = session.execute(query.prepared_statement).mappings().fetchall()
    result = pd.DataFrame(result)
    return result


def save_advanced_power_forecast_predictions(df: pd.DataFrame) -> None:
    """Save advanced power forecast predictions to database using ORM.

    Uses session.merge() for automatic INSERT or UPDATE based on primary key.

    Expected DataFrame columns:
        - asset_id: int
        - model_type_id: int (FK to machine_learning.ml_model_type.id)
        - available_date: datetime
        - prediction_date: datetime
        - forecast_value: float
        - lower_limit: float (optional)
        - upper_limit: float (optional)
    """
    if df.empty:
        return

    with database.session() as session:
        for _, row in df.iterrows():
            # Create ORM object - merge will INSERT or UPDATE based on primary key
            prediction = AdvancedPowerForecastData(
                asset_id=int(row["asset_id"]),
                model_type_id=int(row["model_type_id"]),
                available_date=row["available_date"],
                prediction_date=row["prediction_date"],
                forecast_value=float(row["forecast_value"]),
                lower_limit=float(row["lower_limit"]) if pd.notna(row["lower_limit"]) else None,
                upper_limit=float(row["upper_limit"]) if pd.notna(row["upper_limit"]) else None,
            )
            session.merge(prediction)

        session.commit()
