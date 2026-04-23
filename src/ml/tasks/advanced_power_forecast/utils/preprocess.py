import pandas as pd


def pivot_power_forecast_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot data by provider_id."""
    df_pivot = df.pivot_table(
        index=["asset_id", "available_date", "prediction_date"],
        columns="provider_id",
        values="power_forecast",
    ).reset_index()
    df_pivot.columns.name = None  # Remove the pivot column name
    return df_pivot


def preprocess_power_forecast_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data by converting date columns to datetime."""

    # Pivot data by provider_id
    df = pivot_power_forecast_data(df)

    # Convert date columns to datetime with UTC timezone
    df["prediction_date"] = pd.to_datetime(df["prediction_date"], utc=True)
    df["available_date"] = pd.to_datetime(df["available_date"], utc=True)
    return df
