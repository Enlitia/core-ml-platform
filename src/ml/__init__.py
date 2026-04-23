"""Core ML Platform - Reusable ML framework for time series forecasting."""

import os

# Set default settings module if not already set
# This needs to happen before any other imports that might use toolkit
if not os.getenv("SM_SETTINGS_MODULE"):
    os.environ["SM_SETTINGS_MODULE"] = "dev"

__version__ = "0.1.0"
