import os
from urllib.parse import quote_plus

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Specific attributes for the PRODUCTION environment"""

    # Client Configuration
    client_name: str = os.getenv("CLIENT_NAME", "")

    # Database Configuration
    db_name: str = os.getenv("DB_NAME", client_name)

    databases: dict[str, str] = {
        "default": (
            "postgresql://{user}:{password}@192.168.60.18:5432/{db_name}".format(
                user=os.getenv("DB_USER", ""),
                password=quote_plus(os.getenv("DB_PASSWORD", "")),
                db_name=db_name,
            )
        ),
    }

    power_forecast_table: str = "data_lake.power_forecast_data"
    power_real_table: str = "data_warehouse.farm_data_power_real"
    asset_table: str = "data_lake.asset"
    asset_type_table: str = "data_lake.asset_type"
    advanced_power_forecast_table: str = "data_lake.advanced_power_forecast_data"

    # Nomad Configuration
    nomad_host: str = "192.168.60.18"

    # Mlflow Configuration
    ml_flow_tracking_uri: str = f"/app/mlruns/{client_name}"

    # Logging Configuration
    loki_url: str = os.getenv("LOKI_URL", "")
