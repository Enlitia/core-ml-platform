from __future__ import annotations

import os
from typing import TYPE_CHECKING
from urllib.parse import quote_plus

from pydantic_settings import BaseSettings

if TYPE_CHECKING:
    from config import ClientConfig  # type: ignore

# Ensure SM_SETTINGS_MODULE is set when this module is imported
if not os.getenv("SM_SETTINGS_MODULE"):
    os.environ["SM_SETTINGS_MODULE"] = "dev"


def get_client_config() -> ClientConfig | None:
    """Get client configuration if available."""
    try:
        from config import ClientConfig  # type: ignore

        return ClientConfig()
    except ImportError:
        return None


client_config = get_client_config()


class Settings(BaseSettings):
    """Specific attributes for the DEVELOPMENT environment"""

    # Client Configuration
    client_name: str = client_config.client_name if client_config else os.getenv("CLIENT_NAME", "")

    # Database Configuration
    db_name: str = client_config.db_name if client_config else os.getenv("DB_NAME", client_name)
    db_user: str = client_config.db_user if client_config else os.getenv("DB_USER", "")
    db_password: str = client_config.db_password if client_config else os.getenv("DB_PASSWORD", "")

    databases: dict[str, str] = {
        "default": (
            "postgresql://{user}:{password}@192.168.60.18:5432/{db_name}".format(
                user=db_user,
                password=quote_plus(db_password),
                db_name=db_name,
            )
        ),
    }

    # Nomad Configuration
    nomad_host: str = "192.168.60.18"

    # Mlflow Configuration
    ml_flow_tracking_uri: str = f"./mlruns/{client_name}"

    # Logging Configuration
    loki_url: str = os.getenv("LOKI_URL", "")
