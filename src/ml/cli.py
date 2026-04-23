"""
Core ML Platform CLI - Main interface for ML operations.

This CLI runs ML tasks using configuration from environment variables.
Client-specific config should be set by the client repository.

Usage:
    # List available tasks
    python -m ml list-tasks

    # Train a model
    python -m ml train --task advanced_power_forecast

    # Predict
    python -m ml predict --task advanced_power_forecast
"""

import os
from datetime import datetime
from typing import Optional

import typer

from ml.tasks import TASK_CONFIG_REGISTRY, get_task_handler


def discover_client_name() -> Optional[str]:
    """Get client name from config.py."""
    try:
        from config import ClientConfig
        return ClientConfig().client_name
    except (ImportError, AttributeError):
        return None


def load_client_config() -> None:
    """Load client configuration and set environment variables."""
    try:
        from config import ClientConfig
        client_config = ClientConfig()
        
        # Set all config values as environment variables
        os.environ["CLIENT_NAME"] = client_config.client_name
        os.environ["DB_NAME"] = client_config.db_name
        os.environ["DB_USER"] = client_config.db_user
        os.environ["DB_PASSWORD"] = client_config.db_password
        
    except (ImportError, AttributeError):
        pass


def validate_environment() -> None:
    """Validate that required environment variables are set."""
    # Load client config first
    load_client_config()
    
    if not os.getenv("CLIENT_NAME"):
        typer.echo(f"❌ Missing CLIENT_NAME", err=True)
        typer.echo("Create a config.py with ClientConfig class containing client_name", err=True)
        raise typer.Exit(code=1)

    if not os.getenv("SM_SETTINGS_MODULE"):
        os.environ["SM_SETTINGS_MODULE"] = "dev"


def check_environment() -> None:
    """Check if we're running in production and warn user."""
    env = os.getenv("SM_SETTINGS_MODULE", "dev")
    client = os.getenv("CLIENT_NAME", "unknown")

    if env == "production":
        typer.echo(f"⚠️  Running in PRODUCTION environment for client '{client}'", err=True)


app = typer.Typer(
    help="Core ML Platform CLI - Unified interface for ML operations",
    no_args_is_help=True,
)


@app.command(name="train")
def train_task(
    task: str = typer.Option(..., help="Task name (e.g., advanced_power_forecast)"),
    model: Optional[str] = typer.Option(None, help="Model name (optional, uses task default if not provided)"),
    asset_ids: str = typer.Option("all", help="Asset IDs: 'all', single ID, or comma-separated IDs"),
) -> None:
    """Train a model for a specific task."""
    validate_environment()
    check_environment()

    if task not in TASK_CONFIG_REGISTRY:
        typer.echo(f"❌ Unknown task '{task}'. Use 'ml list-tasks' to see available tasks.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"🚀 Starting training for task '{task}' with asset_ids={asset_ids}")

    # Get and run the task's train function from registry
    try:
        train_func = get_task_handler(task, "train")
        train_func(asset_ids=asset_ids, task_name=task, model_name=model)
    except KeyError as e:
        typer.echo(f"❌ {e}", err=True)
        raise typer.Exit(code=1)


@app.command(name="predict")
def predict_task(
    task: str = typer.Option(..., help="Task name (e.g., advanced_power_forecast)"),
    model: Optional[str] = typer.Option(None, help="Model name (optional, uses task default if not provided)"),
    asset_ids: str = typer.Option("all", help="Asset IDs: 'all', single ID, or comma-separated IDs"),
    start_date: Optional[str] = typer.Option(None, help="Start date (ISO format, default: now)"),
) -> None:
    """Generate predictions for a specific task."""
    validate_environment()
    check_environment()

    if task not in TASK_CONFIG_REGISTRY:
        typer.echo(f"❌ Unknown task '{task}'. Use 'ml list-tasks' to see available tasks.", err=True)
        raise typer.Exit(code=1)

    # Parse start_date if provided
    parsed_start_date = None
    if start_date:
        try:
            parsed_start_date = datetime.fromisoformat(start_date)
        except ValueError:
            typer.echo(f"❌ Invalid date format '{start_date}'. Use ISO format (e.g., 2026-04-16T10:30:00)", err=True)
            raise typer.Exit(code=1)

    typer.echo(f"🚀 Starting predictions for task '{task}' with asset_ids={asset_ids}")

    # Get and run the task's predict function from registry
    try:
        predict_func = get_task_handler(task, "predict")
        predict_func(asset_ids=asset_ids, task_name=task, model_name=model, start_date=parsed_start_date)
    except KeyError as e:
        typer.echo(f"❌ {e}", err=True)
        raise typer.Exit(code=1)


@app.command(name="list-tasks")
def list_tasks() -> None:
    """List all available ML tasks."""
    typer.echo("Available ML tasks:\n")
    for task_name, config in TASK_CONFIG_REGISTRY.items():
        model_name = getattr(config, "default_model_name", "N/A")
        typer.echo(f"  • {task_name:<30} (model: {model_name})")
    typer.echo(f"\nTotal: {len(TASK_CONFIG_REGISTRY)} tasks")


@app.command(name="update-nomad-configs")
def update_nomad_configs() -> None:
    """Update Nomad job files from client registry.

    Generates Nomad .hcl files for all registered clients and their tasks,
    ensuring consistency between code and deployment configs.
    """
    import sys
    from importlib import import_module
    from pathlib import Path

    # Import the main function from ops/update_nomad_configs.py
    ops_path = Path(__file__).parent.parent / "ops"
    sys.path.insert(0, str(ops_path))

    try:
        update_nomad_module = import_module("update_nomad_configs")
        update_nomad_main = update_nomad_module.main
        update_nomad_main()
    except Exception as e:
        typer.echo(f"❌ Error updating Nomad configs: {e}", err=True)
        raise typer.Exit(code=1)


@app.callback()
def callback() -> None:
    """
    Service Core ML CLI

    Run ML tasks (train/predict) for different clients and environments.
    Each client has its own database and configuration.
    """
    pass


if __name__ == "__main__":
    app()
