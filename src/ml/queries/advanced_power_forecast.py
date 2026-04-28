from datetime import datetime

import pandas as pd
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from toolkit.data.query import Query
from toolkit.database import database

from .base import Base


class AdvancedPowerForecastData(Base):
    """Model for advanced power forecast predictions."""

    __tablename__ = "advanced_power_forecast_data"
    __table_args__ = (
        UniqueConstraint(
            "asset_id",
            "prediction_date",
            "available_date",
            name="uq_advanced_power_forecast_data",
        ),
        {"schema": "machine_learning"},
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(Integer, ForeignKey("data_lake.asset.id"), nullable=False, index=True)
    model_name = Column(String, nullable=False, index=True)
    available_date = Column(DateTime, nullable=False, index=True)
    prediction_date = Column(DateTime, nullable=False, index=True)
    forecast_value = Column(Float, nullable=False)
    upper_limit = Column(Float, nullable=True)
    lower_limit = Column(Float, nullable=True)
    created_at = Column(DateTime, nullable=False, server_default="CURRENT_TIMESTAMP")
    updated_at = Column(DateTime, nullable=True, server_default="CURRENT_TIMESTAMP")


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
        - model_name: str (e.g., 'positive_linear', 'xgboost')
        - available_date: datetime
        - prediction_date: datetime
        - forecast_value: float
        - upper_limit: float (optional)
        - lower_limit: float (optional)
    """
    if df.empty:
        return

    with database.session() as session:
        for _, row in df.iterrows():
            # Create ORM object - merge will INSERT or UPDATE based on unique constraint
            prediction = AdvancedPowerForecastData(
                asset_id=int(row["asset_id"]),
                model_name=str(row["model_name"]),
                available_date=row["available_date"],
                prediction_date=row["prediction_date"],
                forecast_value=float(row["forecast_value"]),
                upper_limit=float(row["upper_limit"]) if pd.notna(row["upper_limit"]) else None,
                lower_limit=float(row["lower_limit"]) if pd.notna(row["lower_limit"]) else None,
            )
            session.merge(prediction)

        session.commit()
