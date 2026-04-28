# Service Core ML - Architecture Documentation

## C4 Model Architecture Diagrams

The C4 model provides a hierarchical way to visualize the architecture of the ML service.

---

## Level 1: System Context

**What:** High-level view showing how the system fits into the broader environment.

```
┌──────────────┐
│   Scheduler  │ ──── Triggers train/predict jobs periodically
│    (Nomad)   │      (cron schedules defined in CLIENT_REGISTRY)
└──────────────┘
       │
       ├─────────────────────────────────────┐
       │                                     │
       ▼                                     ▼
┌─────────────────────┐            ┌──────────────────┐
│   ML Service        │◀───────────│   Developer      │
│  (service-core-ml)  │            │   (via CLI)      │
└─────────────────────┘            └──────────────────┘
       │         │
       │         └────────────────┐
       ▼                          ▼
┌──────────────┐          ┌─────────────────┐
│  PostgreSQL  │          │  MLflow Registry│
│  Databases   │          │  (Model Store)  │
│ (per client) │          └─────────────────┘
└──────────────┘
 ─ erg_db
 ─ other_client_db
```

**Key Interactions:**
- Scheduler (Nomad) triggers train/predict jobs on cron schedules
- Developers run CLI commands manually for testing/debugging
- ML Service reads/writes to client-specific databases
- ML Service stores trained models in MLflow Registry
- Each client has its own database with standardized schema

---

## Level 2: Container Diagram

**What:** Shows the major applications and data stores that make up the system.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          ML Service (Python)                             │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                       CLI Layer (Typer)                             │ |
│  │  • ml train --task <task> --client <client> --env <env>             │ │
│  │  • ml predict --task <task> --client <client>                       │ │
│  │  • ml list-tasks, ml list-clients                                   │ │
│  │  • update-nomad-configs                                             │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    Configuration Layer                              │ │
│  │  ┌────────────────┐  ┌──────────────────┐  ┌────────────────┐       │ │
│  │  │ CLIENT_REGISTRY│  │ TASK_CONFIG_     │  │ MODEL_REGISTRY │       │ │
│  │  │                │  │ REGISTRY         │  │                │       │ │
│  │  │ • erg          │  │ • adv_power_fcst │  │ • positive_    │       │ │
│  │  │ • other_client │  │ • power_forecast │  │   linear       │       │ │
│  │  └────────────────┘  └──────────────────┘  └────────────────┘       │ │
│  │  Provides: DB credentials, tasks, model algorithms                  │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    Task Execution Layer                             │ │
│  │  ┌──────────────────────────────┐  ┌──────────────────────────────┐ │ │
│  │  │ Train Workflows              │  │ Predict Workflows            │ │ │
│  │  │                              │  │                              │ │ │
│  │  │ 1. Fetch historical data     │  │ 1. Load trained model        │ │ │
│  │  │ 2. Preprocess & validate     │  │ 2. Fetch latest forecast data│ │ │
│  │  │ 3. Train model per asset     │  │ 3. Generate predictions      │ │ │
│  │  │ 4. Evaluate metrics          │  │ 4. Save to database          │ │ │
│  │  │ 5. Save to MLflow            │  │                              │ │ │
│  │  └──────────────────────────────┘  └──────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    Model Layer (scikit-learn)                       │ │
│  │  • BaseModel (abstract)                                             │ │
│  │  • PositiveLinearModel (linear regression with positive constraint) │ │
│  │  • [Future: XGBoost, KDE, etc.]                                     │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                  Infrastructure Gateways                            │ │
│  │  ┌─────────────────────┐            ┌─────────────────────────┐     │ │
│  │  │   DBGateway         │            │   MLflowGateway         │     │ │
│  │  │                     │            │                         │     │ │
│  │  │ • fetch_df()        │            │ • save_model()          │     │ │
│  │  │ • save_df()         │            │ • load_model()          │     │ │
│  │  │ • session mgmt      │            │ • set tags/metrics      │     │ │
│  │  └─────────────────────┘            └─────────────────────────┘     │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
                     │                               │
                     ▼                               ▼
        ┌─────────────────────┐          ┌──────────────────────┐
        │   PostgreSQL DBs    │          │  MLflow Registry     │
        │                     │          │                      │
        │ • erg (192.168...)  │          │ • Experiment per task│
        │ • other_client      │          │ • Model per client   │
        │                     │          │   + asset            │
        │ Tables:             │          │ • Champion alias     │
        │ - power_forecast    │          │   for production     │
        │ - power_real        │          └──────────────────────┘
        │ - asset             │
        │ - asset_type        │
        └─────────────────────┘
```

---

## Level 3: Component Diagram

**What:** Major structural components within the ML Service container.

### Configuration Components

```
CLIENT_REGISTRY (clients/__init__.py)
├── Purpose: Maps client names to database credentials and task assignments
├── Structure: dict[str, ClientConfig]
└── Example:
    {
      "erg": ClientConfig(
        db_user="...",
        db_password="...",
        tasks={"advanced_power_forecast": NomadTaskConfig(...)}
      )
    }

TASK_CONFIG_REGISTRY (ml/tasks/__init__.py)
├── Purpose: Maps task names to ML configuration parameters
├── Structure: dict[str, BaseTaskConfig]
└── Example:
    {
      "advanced_power_forecast": AdvancedPowerForecastConfig(
        model_name="positive_linear",
        training_interval="1 year",
        prediction_days=15,
        ...
      )
    }

TASK_HANDLERS (ml/tasks/__init__.py)
├── Purpose: Maps task names to train/predict function implementations
├── Structure: dict[str, dict[str, Callable]]
└── Example:
    {
      "advanced_power_forecast": {
        "train": ml.tasks.advanced_power_forecast.train.train,
        "predict": ml.tasks.advanced_power_forecast.predict.predict
      }
    }

MODEL_REGISTRY (ml/models/__init__.py)
├── Purpose: Maps model names to model class implementations
├── Structure: dict[str, Type[BaseModel]]
└── Example:
    {
      "positive_linear": PositiveLinearModel
    }
```

### Data Flow Through Components

```
┌──────────────┐
│ CLI Command  │  ml train --task advanced_power_forecast --client erg
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────┐
│ 1. Load Configs from Registries          │
│                                          │
│   client_config = CLIENT_REGISTRY["erg"] │
│   task_config = TASK_CONFIG_REGISTRY[    │
│                   "advanced_power_..."]  │
│   train_fn = TASK_HANDLERS[task]["train"]│
└──────┬───────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│ 2. Initialize Context                    │
│                                          │
│   Context(                               │
│     settings=task_config,                │
│     logger=get_logger(),                 │
│     mlflow=MLflowGateway()               │
│   )                                      │
└──────┬───────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│ 3. Execute Task Workflow                 │
│                                          │
│   For each asset:                        │
│   • Fetch data (DBGateway)               │
│   • Preprocess                           │
│   • Train model                          │
│   • Evaluate                             │
│   • Save (MLflowGateway)                 │
└──────┬───────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│ 4. Persist Results                       │
│                                          │
│   MLflow: model + metrics + params       │
│   Database: predictions (future)         │
└──────────────────────────────────────────┘
```

---

## Registry Pattern Benefits

**Single Source of Truth:**
- Adding a client → Update CLIENT_REGISTRY only
- Adding a task → Update TASK_CONFIG_REGISTRY + TASK_HANDLERS only
- Adding a model → Update MODEL_REGISTRY only

**Consistency:**
- Nomad configs generated from CLIENT_REGISTRY (no drift)
- All code references same configuration objects
- No scattered configuration files to maintain

**Type Safety:**
- Pydantic validates all config objects
- Type hints throughout for IDE support
- Runtime validation of configuration

---

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Nomad Cluster                              │
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Job: erg-advanced-power-forecast-train                    │ │
│  │  Schedule: "0 4 * * *" (daily at 4am)                      │ │
│  │  Command: ml train --task advanced_power_forecast          │ │
│  │           --client erg --env production                    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Job: erg-advanced-power-forecast-predict                  │ │
│  │  Schedule: "0 */4 * * *" (every 4 hours)                   │ │
│  │  Command: ml predict --task advanced_power_forecast        │ │
│  │           --client erg --env production                    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  [Jobs auto-generated from CLIENT_REGISTRY via:                 │
│   update-nomad-configs command]                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. **Registry Pattern Over Central Config**
- **Decision:** Use separate registries for clients, tasks, and models
- **Rationale:** Keeps components decoupled and independently extensible
- **Tradeoff:** Indirection (registry lookup) vs tight coupling

### 2. **Task = Business Logic, Model = ML Algorithm**
- **Decision:** Separate concerns between what data to process (task) and how to model it (model)
- **Rationale:** One model can serve multiple tasks; tasks own their preprocessing logic
- **Example:** `positive_linear` model used by both `advanced_power_forecast` and `power_forecast`

### 3. **Per-Client Databases**
- **Decision:** Each client gets their own PostgreSQL database
- **Rationale:** Data isolation, independent scaling, client-specific access control
- **Tradeoff:** More infrastructure vs security/isolation

### 4. **MLflow Champion Alias**
- **Decision:** Use "champion" alias instead of "Production" stage
- **Rationale:** MLflow deprecated stages in favor of aliases (more flexible)
- **Benefit:** Can have multiple aliases (champion, challenger, baseline)

### 5. **Context Object Pattern**
- **Decision:** Inject settings, logger, mlflow via Context dataclass
- **Rationale:** Explicit dependencies, easy testing, clear ownership
- **Tradeoff:** More verbose vs global state

---
