from datetime import datetime

import pandas as pd
from toolkit.configuration import configuration

from core_ml.infrastructure import DBGateway

# Assets


def _fetch_list_all_available_asset_ids() -> list[int]:
    """Get list of all available asset ids."""
    query = f"""
    SELECT
        asset.id AS asset_id
    FROM
        {configuration.settings.asset_table} AS asset
    INNER JOIN
        {configuration.settings.asset_type_table} AS asset_type
    	ON asset.asset_type_id = asset_type.id
    WHERE
        asset_type.name IN ('Wind farm', 'Solar farm')
    ORDER BY 1
    """
    df_assets = DBGateway.fetch_df(query)
    list_asset_ids = df_assets["asset_id"].tolist()
    return list_asset_ids


# Train Advanced Power Forecast


def fetch_power_forecast_data_for_train(list_asset_ids: list[int], training_interval: str) -> pd.DataFrame:
    """Get forecast data for specific assets."""
    query = f"""
        SELECT
            asset_id,
            available_date,
            prediction_date,
            provider_id, 
            forecast_value AS power_forecast
        FROM 
            {configuration.settings.power_forecast_table}
        WHERE 
            asset_id IN ({', '.join(map(str, list_asset_ids))})
            AND available_date > NOW() - INTERVAL '{training_interval}'
        ORDER BY 1, 2, 3, 4
    """
    df_power_forecast = DBGateway.fetch_df(query)
    return df_power_forecast


def fetch_power_real_data_for_train(list_asset_ids: list[int], training_interval: str) -> pd.DataFrame:
    """Get real data for specific assets."""
    query = f"""
        SELECT
            farm_id AS asset_id,
            read_at AS prediction_date,
            power_real
        FROM
            {configuration.settings.power_real_table}
        WHERE
            farm_id IN ({', '.join(map(str, list_asset_ids))})
            AND read_at > NOW() - INTERVAL '{training_interval}'
        ORDER BY 1, 2, 3
    """
    df = DBGateway.fetch_df(query)
    return df


# Predict Advanced Power Forecast


def fetch_power_forecast_data_for_prediction(
    list_asset_ids: list[int],
    start_date: datetime,
    end_date: datetime,
    delta_minutes: int,
) -> pd.DataFrame:
    """Get forecast data for a specific asset."""
    query = f"""
    WITH forecast_data AS (
        SELECT
            asset_id,
            available_date,
            prediction_date,
            provider_id, 
            forecast_value AS power_forecast,
            MAX(available_date) OVER (PARTITION BY asset_id, prediction_date, provider_id) AS max_available_date
        FROM 
            {configuration.settings.power_forecast_table} 
        WHERE 
            asset_id IN ({', '.join(map(str, list_asset_ids))})
            AND prediction_date BETWEEN '{start_date}' AND '{end_date}'
            AND EXTRACT(MINUTE FROM prediction_date) % {delta_minutes} = 0
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
    df_power_forecast = DBGateway.fetch_df(query)
    return df_power_forecast


def save_advanced_power_forecast_predictions(df: pd.DataFrame) -> None:
    """Save advanced power forecast predictions to database."""
    DBGateway.insert_update_df(
        df=df,
        table_name=configuration.settings.advanced_power_forecast_table,
        conflict_columns=["asset_id", "prediction_date", "available_date"],
    )
