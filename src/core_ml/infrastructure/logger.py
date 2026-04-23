import logging
import os
import sys

import logging_loki
from toolkit.configuration import configuration


def get_logger(name: str, level: int | None = None) -> logging.Logger:
    """Get a configured logger instance with console and Loki handlers.

    Args:
        name: Logger name (typically __name__ of the calling module)
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance

    Configuration:
        loki_url: Loki endpoint URL from settings (e.g., http://loki:3100/loki/api/v1/push)
        client_name: Client name from settings (used as service tag)
    """
    if level is None:
        level = logging.INFO

    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(level)

        # Create formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler (always present for immediate visibility)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Loki handler (optional, for centralized log aggregation)
        loki_url = configuration.settings.loki_url
        if loki_url:
            try:
                loki_handler = logging_loki.LokiHandler(
                    url=loki_url,
                    tags={
                        "service": "service-core-ml",
                        "client": configuration.settings.client_name,
                        "environment": os.getenv("ENV", "dev"),
                    },
                    version="1",
                )
                loki_handler.setLevel(level)
                logger.addHandler(loki_handler)
            except Exception as e:
                # Log to console if Loki setup fails (fallback to console only)
                logger.warning(f"Failed to configure Loki handler: {e}")

    return logger
