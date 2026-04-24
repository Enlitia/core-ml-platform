from datetime import datetime

import pandas as pd
from sqlalchemy import text
from toolkit.data.query import Query
from toolkit.database import database


def _fetch_list_all_available_asset_ids(list_asset_types: list[str] = ["Wind farm", "Solar farm"]) -> list[int]:
    """Get list of all available asset ids."""
    sql_query = """
    SELECT
        asset.id AS asset_id
    FROM
        data_lake.asset AS asset
    INNER JOIN
        data_lake.asset_type AS asset_type
    	ON asset.asset_type_id = asset_type.id
    ORDER BY 1
    """
    query = Query(sql_query)
    query.with_in("asset_type.name", list_asset_types)
    with database.session() as session:
        result = session.execute(query.prepared_statement).mappings().fetchall()

    list_asset_ids = [row["asset_id"] for row in result]
    return list_asset_ids


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


def _prepare_dataframe_for_insert(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataframe for database insert by replacing NaN with None."""
    df_obj = df.astype(object)
    null_mask: pd.DataFrame = pd.notnull(df_obj)
    df_clean: pd.DataFrame = df_obj.where(null_mask, None)  # type: ignore[call-overload]
    return df_clean


def upsert_df(df: pd.DataFrame, table_name: str, conflict_columns: list[str]) -> None:
    """Generic function to save dataframe to database with conflict resolution."""
    if df.empty:
        return

    df_clean = _prepare_dataframe_for_insert(df)

    columns = list(df_clean.columns)

    # PostgreSQL: ON CONFLICT (columns) DO UPDATE SET ...
    cols_str = ", ".join(columns)
    placeholders = ", ".join([f":{col}" for col in columns])
    conflict_str = ", ".join(conflict_columns)
    update_pairs = ", ".join([f"{col} = EXCLUDED.{col}" for col in columns])

    query_str = f"""
        INSERT INTO {table_name} ({cols_str})
        VALUES ({placeholders})
        ON CONFLICT ({conflict_str}) DO UPDATE SET {update_pairs}
    """

    with database.session() as session:
        for _, row in df_clean.iterrows():
            params = {str(k): v for k, v in row.to_dict().items()}
            session.execute(text(query_str), params)
        session.commit()


def save_advanced_power_forecast_predictions(df: pd.DataFrame) -> None:
    """Save advanced power forecast predictions to database."""
    upsert_df(
        df=df,
        table_name="data_lake.advanced_power_forecast_data",
        conflict_columns=["asset_id", "prediction_date", "available_date"],
    )
