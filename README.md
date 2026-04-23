# Service Core ML

A machine learning service that trains and predicts forecasts for multiple clients. Each client gets periodic model training (every few months) and frequent predictions (every few hours).

## Architecture Overview

### Registries: Single Source of Truth

The service uses **registry patterns** for consistent configuration:

- **`MODEL_REGISTRY`** (`src/ml/models/__init__.py`) - Available ML models
- **`TASK_CONFIG_REGISTRY`** (`src/ml/tasks/__init__.py`) - Task configurations and parameters  
- **`CLIENT_REGISTRY`** (`src/clients/__init__.py`) - Client configs with database credentials and tasks

All registries follow the same pattern: configuration classes registered in a central dict with singular getter functions (`get_model()`, `get_task_config()`, `get_client()`).

### Models vs Tasks

**Models** (`src/ml/models/`) contain the **ML logic** - how to fit, predict, and evaluate. Think of them as reusable ML algorithms (ex: positive_linear, kde3).

**Tasks** (`src/ml/tasks/`) contain the **business logic** - what data to fetch, how to preprocess it, and what to do with predictions. Each task has its own `config.py` with task-specific parameters. Tasks use models to solve specific business problems (ex: advanced_power_forecast).

One model can be used by many tasks, or you can create task-specific models.

### How It Works

1. **Training**: Runs periodically (configured in Nomad) to retrain models with latest data
2. **Prediction**: Runs frequently to generate fresh forecasts
3. **MLflow**: Tracks experiments and stores trained models
4. **Database**: Each client has a database with standard columns for storing data and predictions
5. **Nomad Configs**: Auto-generated from CLIENT_REGISTRY to ensure consistency

---

## Adding a New Client

1. **Add client configuration** in `src/clients/<client_name>.py`:
   ```python
   from clients.base import ClientConfig, NomadTaskConfig
   
   class NewClientConfig(ClientConfig):
       """Configuration for New Client."""
       
       client_name: str = "new_client"
       db_user: str = "db_user"
       db_password: str = "db_password"
       
       tasks: dict[str, NomadTaskConfig] = {
           "advanced_power_forecast": NomadTaskConfig(
               train_cron="0 4 * * *",      # Daily at 4am (None = use default)
               predict_cron="0 */4 * * *",  # Every 4 hours (None = use default)
               enabled=True,
               memory_mb=1024,              # Memory in MB (None = use default)
           ),
           "power_forecast": NomadTaskConfig(
               enabled=True,  # Use all defaults
           ),
       }
   ```

2. **Register the client** in `src/clients/__init__.py`:
   ```python
   from clients.base import ClientConfig
   from clients.erg import ErgClientConfig
   from clients.new_client import NewClientConfig
   
   CLIENT_REGISTRY: dict[str, ClientConfig] = {
       "erg": ErgClientConfig(),
       "new_client": NewClientConfig(),
   }
   ```

3. **Generate Nomad configs** automatically from the registry:
   ```bash
   poetry run python ops/update_nomad_configs.py
   ```
   This creates `ops/nomad/<client>/` with `<task>-train.hcl` and `<task>-predict.hcl` files for each enabled task.

4. **Ensure database exists** with standard column structure (check existing client schemas)

5. **Deploy Nomad jobs** from generated configs in `ops/nomad/<client>/`

**Note**: The client registry with NomadTaskConfig ensures consistency - one source of truth for all client configurations and Nomad deployments.

---

## Adding a New Model

Models define **how** to perform ML operations (fit, predict, evaluate).

### Steps

1. **Create model class** in `src/ml/models/<model_name>.py`:
   - Inherit from `BaseModel`
   - Set `self.model_name` in `__init__`
   - Implement `fit()`, `predict()`, `evaluate()`, and `get_feature_weights()`

2. **Register the model** in `src/ml/models/__init__.py`:
   ```python
   from .your_model import YourModel
   
   MODEL_REGISTRY = {
       "positive_linear": PositiveLinearModel,
       "xgboost": XGBoostModel,
       "your_model": YourModel,  # Add this line
   }
   ```

3. **Add to task's available models** in task config:
   ```python
   class YourTaskConfig(BaseTaskConfig):
       available_models: list[str] = ["positive_linear", "xgboost", "your_model"]
   ```

4. **Use with CLI**:
   ```bash
   # Use your new model
   poetry run python -m cli ml train --task your_task --client erg --model your_model
   ```

**Examples**: Check `positive_linear.py`, `xgboost_model.py`, or `random_forest_model.py` for reference implementations.

---

## Adding a New Task

Tasks define **what** business problem to solve using models.

### Steps

1. **Copy the template** from `src/ml/tasks/_template/` to `src/ml/tasks/<your_task_name>/`

2. **Create task configuration** in `src/ml/tasks/<your_task_name>/config.py`:
   ```python
   from ml.tasks.base import BaseTaskConfig
   
   class YourTaskConfig(BaseTaskConfig):
       """Configuration for Your Task."""
       
       task_name: str = "your_task_name"  # Must match folder name
       
       # Model Configuration
       default_model_name: str = "positive_linear"  # Default model
       available_models: list[str] = ["positive_linear", "xgboost", "random_forest"]
       model_params: dict[str, dict] = {}  # Optional model-specific params
       
       # Add task-specific parameters
       training_interval: str = "1 year"
       prediction_days: int = 15
   ```
   
   **Tip**: Copy from `src/ml/tasks/advanced_power_forecast/config.py` and adapt.

3. **Register the task config** in `src/ml/tasks/__init__.py`:
   ```python
   from ml.tasks.your_task_name.config import YourTaskConfig
   
   TASK_CONFIG_REGISTRY = {
       "advanced_power_forecast": AdvancedPowerForecastConfig(),
       "your_task_name": YourTaskConfig(),
   }
   
   TASK_HANDLERS = {
       "your_task_name": {
           "train": your_task_train,
           "predict": your_task_predict,
       },
   }
   ```

4. **Adapt the task code**:
   - `train.py` - training logic (accept `model_name` parameter)
   - `predict.py` - prediction logic (accept `model_name` parameter)
   - `utils/` - shared code used in both train and predict (data transformations, feature engineering, etc.)

5. **Follow the pattern** from `advanced_power_forecast`:
   - Fetch data (use `ml.common.queries`)
   - Put reusable logic in task-specific utils to avoid duplication between train/predict
   - Validate and clean data
   - Train model or make predictions
   - Save results back to database

**Examples**: Use `advanced_power_forecast` as template.

---

## Model Selection

Tasks can support multiple models. Select which model to use via the `--model` CLI flag.

### Using Different Models

```bash
# Use default model (defined in task config)
poetry run python -m cli ml train --task advanced_power_forecast --client erg

# Use XGBoost instead
poetry run python -m cli ml train --task advanced_power_forecast --client erg --model xgboost

# Use Random Forest
poetry run python -m cli ml train --task advanced_power_forecast --client erg --model random_forest

# Predict with specific model
poetry run python -m cli ml predict --task advanced_power_forecast --client erg --model xgboost
```

### Configuring Model Parameters

Override default model parameters in the task config:

```python
class YourTaskConfig(BaseTaskConfig):
    # Model Configuration
    default_model_name: str = "positive_linear"
    available_models: list[str] = ["positive_linear", "xgboost", "random_forest"]
    
    # Custom parameters per model
    model_params: dict[str, dict] = {
        "xgboost": {"n_estimators": 200, "max_depth": 8, "learning_rate": 0.05},
        "random_forest": {"n_estimators": 150, "max_depth": 10},
    }
```

---

## Quick Reference

| Component | Location | Purpose |
|-----------|----------|---------|
| Models | `src/ml/models/` | ML algorithms (fit/predict logic) |
| Tasks | `src/ml/tasks/` | Business logic (what to predict, for whom) |
| Task Configs | `src/ml/tasks/<task>/config.py` | Task configuration and parameters |
| Clients | `src/clients/` | Client database configurations |
| Nomad Jobs | `ops/nomad/` | Deployment and scheduling (auto-generated from client registry) |
| Common | `src/ml/common/` | Shared code used across different tasks/models (queries, validation, data splitting, etc.) |

## Running Tasks

**Quick Start**: Use `poetry run python -m cli ml <command>` for all operations.

### Via CLI (Recommended)

The CLI provides a unified interface for running tasks across different clients. Client configurations (DB credentials) are defined in `src/clients/*.py` and registered in the `CLIENT_REGISTRY`.

**Use Poetry to run commands** - it ensures you use the correct virtual environment:

```bash
# List available clients
poetry run python -m cli ml list-clients

# List available tasks
poetry run python -m cli ml list-tasks

# Train a model
poetry run python -m cli ml train \
  --task advanced_power_forecast \
  --client erg \
  --env dev

# Train with specific model
poetry run python -m cli ml train \
  --task advanced_power_forecast \
  --client erg \
  --model xgboost

# Train using environment variable
export CLIENT_NAME=erg
export SM_SETTINGS_MODULE=dev
poetry run python -m cli ml train --task advanced_power_forecast

# Predict for specific assets
poetry run python -m cli ml predict \
  --task advanced_power_forecast \
  --client erg \
  --asset-ids "1,2,3"

# Predict with custom start date
poetry run python -m cli ml predict \
  --task advanced_power_forecast \
  --client erg \
  --start-date "2026-04-16T10:00:00"

# Get help
poetry run python -m cli ml --help
poetry run python -m cli ml train --help
```

### Direct Task Execution (Alternative)

You can also run tasks directly without the CLI wrapper (as used in Nomad deployments):

```bash
export CLIENT_NAME=erg
export DB_USER=app_hub
export DB_PASSWORD=xxxxx
export SM_SETTINGS_MODULE=production
poetry run python src/ml/tasks/advanced_power_forecast/train.py
```

## Development

```bash
# Install dependencies
make install

# Run checks (linting, formatting, tests)
make check
```

---

**Need help?** Check existing implementations in the codebase - they're your best documentation.
