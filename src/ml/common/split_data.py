import pandas as pd
from sklearn.model_selection import train_test_split


def split_data_by_day(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data by calendar day to prevent temporal leakage.

    Ensures that all samples from the same day are kept together in either
    train or test set. Prevents data leakage in time series forecasting.

    Args:
        X: Feature dataframe
        y: Target series with matching index
        dates: Date column or index to split by (must have same length as X)
        test_size: Proportion of the dataset to include in the test split
        random_state: Seed used by the random number generator

    Returns:
        X_train, X_test, y_train, y_test split by day
    """
    # Extract unique calendar days
    dates = pd.to_datetime(dates).dt.date
    unique_days = pd.Series(dates.values).unique()

    # Split unique days into train/test
    train_days, test_days = train_test_split(
        unique_days,
        test_size=test_size,
        random_state=random_state,
    )

    # Create boolean masks for filtering
    train_mask = pd.Series(dates).isin(train_days).to_numpy()
    test_mask = pd.Series(dates).isin(test_days).to_numpy()

    # Split data based on day membership
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    return X_train, X_test, y_train, y_test
