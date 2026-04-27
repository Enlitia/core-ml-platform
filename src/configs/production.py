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

    # Nomad Configuration
    nomad_host: str = "192.168.60.18"

    # Mlflow Configuration
    ml_flow_tracking_uri: str = f"/app/mlruns/{client_name}"

    # Logging Configuration
    loki_url: str = os.getenv("LOKI_URL", "")
