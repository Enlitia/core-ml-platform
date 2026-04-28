import pandas as pd


def validate_not_empty(df: pd.DataFrame, asset_id: int) -> None:
    """
    Validate that dataframe is not empty.

    Raises ValueError if empty.
    """
    if df.empty:
        raise ValueError(f"No data available for asset {asset_id}")


def validate_negative_values(df: pd.DataFrame, column: str, asset_id: int) -> None:
    """
    Check if column has negative values.

    Raises ValueError if negative values found.
    """
    if (df[column] < 0).any():
        count = (df[column] < 0).sum()
        raise ValueError(f"{column}: found {count} negative values for asset {asset_id}")


def validate_null_values(df: pd.DataFrame, column: str, asset_id: int) -> None:
    """
    Check if column has null values.

    Raises ValueError if nulls found.
    """
    if df[column].isnull().any():
        count = df[column].isnull().sum()
        raise ValueError(f"{column}: found {count} null values for asset {asset_id}")


def validate_out_of_range(df: pd.DataFrame, column: str, min_val: float, max_val: float, asset_id: int) -> None:
    """
    Check if column has values outside specified range.

    Raises ValueError if out of range values found.
    """
    out_of_range = (df[column] < min_val) | (df[column] > max_val)
    if out_of_range.any():
        count = out_of_range.sum()
        raise ValueError(f"{column}: found {count} values outside range [{min_val}, {max_val}] for asset {asset_id}")


def validate_inputs_training(X: pd.DataFrame, y: pd.Series, min_size_train: int, asset_id: int) -> None:
    """Validate model inputs."""
    if X.empty:
        raise ValueError(f"Training: Input features (X) cannot be empty for asset {asset_id}.")
    if y.empty:
        raise ValueError(f"Training: Target variable (y) cannot be empty for asset {asset_id}.")
    if len(X) != len(y):
        raise ValueError(
            f"Training: Input features (X) and target variable (y) must have the same number of samples for asset {asset_id}."
        )
    if X.isnull().any().any():
        raise ValueError(f"Training: Input features (X) contain null values for asset {asset_id}.")
    if y.isnull().any():
        raise ValueError(f"Training: Target variable (y) contains null values for asset {asset_id}.")
    if len(X.columns) == 0:
        raise ValueError(f"Training: Input features (X) must have at least one column for asset {asset_id}.")
    if len(X) < min_size_train:
        raise ValueError(
            f"Training: Input features (X) must have at least {min_size_train} samples for asset {asset_id}."
        )


def validate_inputs_prediction(X: pd.DataFrame, asset_id: int) -> None:
    """
    Validate prediction inputs.

    Raises ValueError if no column has complete data (all columns have NaNs).
    """
    if X.empty:
        raise ValueError(f"Prediction: Input features (X) cannot be empty for asset {asset_id}.")

    if len(X.columns) == 0:
        raise ValueError(f"Prediction: Input features (X) must have at least one column for asset {asset_id}.")

    # Check if at least one column has no NaNs
    cols_without_nans = X.columns[~X.isnull().any()].tolist()

    if len(cols_without_nans) == 0:
        raise ValueError(
            f"Prediction: All columns contain null values for asset {asset_id}. At least one complete column required."
        )


def validate_model_quality(metrics: dict[str, float], thresholds: dict[str, dict[str, float]], asset_id: int) -> None:
    """
    Validate model metrics against quality thresholds.

    Raises ValueError if any metric is missing or violates its threshold.
    """
    for metric_name, limits in thresholds.items():
        actual_value = metrics.get(metric_name)

        # Metric must exist
        if actual_value is None:
            raise ValueError(f"Asset {asset_id}: {metric_name} metric is missing from model evaluation")

        # Extract limits
        max_limit = limits.get("max")
        min_limit = limits.get("min")

        # Check max threshold
        if max_limit is not None and actual_value > max_limit:
            raise ValueError(f"Asset {asset_id}: {metric_name} {actual_value:.2f} exceeds maximum {max_limit:.2f}")

        # Check min threshold
        if min_limit is not None and actual_value < min_limit:
            raise ValueError(f"Asset {asset_id}: {metric_name} {actual_value:.2f} below minimum {min_limit:.2f}")
