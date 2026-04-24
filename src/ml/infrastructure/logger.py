import logging
import os

from toolkit.configuration import configuration
from toolkit.logging import StructuredLogger


def get_logger(name: str, level: int | None = None) -> StructuredLogger:
    """Get a configured StructuredLogger instance.

    Args:
        name: Logger name (typically task name or module name)
        level: Logging level (default: INFO)

    Returns:
        Configured StructuredLogger instance with project/service context

    Configuration:
        client_name: Client name from settings (used as service tag)
        environment: ENV environment variable (default: "dev")
    """
    if level is None:
        level = logging.INFO

    # Determine format based on environment (JSON for prod, text for dev)
    env = os.getenv("ENV", "dev")
    format_type = "json" if env == "production" else "text"

    config = {
        "level": level,
        "format_type": format_type,
        "datefmt": "%Y-%m-%d %H:%M:%S",
        "format": "%(asctime)s | %(levelname)s | %(project)s | %(service)s | %(extras)s | %(message)s",
    }

    logger = StructuredLogger(
        name=name,
        project="core-ml-platform",
        service=configuration.settings.client_name,
        config=config,
    )

    return logger
