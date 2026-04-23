# Tests

## TODO

Implement test structure mirroring the src/ directory:

```
tests/
├── infrastructure/
│   ├── test_database.py
│   ├── test_logger.py
│   └── test_ml_flow.py
├── ml/
│   ├── common/
│   │   ├── test_assets.py
│   │   ├── test_dates.py
│   │   └── test_queries.py
│   ├── models/
│   │   ├── test_base.py
│   │   └── test_positive_linear.py
│   └── tasks/
│       └── advanced_power_forecast/
│           ├── test_train.py
│           ├── test_predict.py
│           └── utils/
│               └── test_preprocess.py
└── README.md
```

## Guidelines

- Use pytest as the testing framework
- Mirror the src/ structure for easy navigation
- Name test files with `test_` prefix
- Aim for unit tests of individual functions/methods
- Add integration tests for end-to-end workflows
