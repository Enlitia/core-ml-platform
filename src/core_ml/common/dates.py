from datetime import datetime, timedelta

import pandas as pd


def get_dates(start_date: datetime | None, prediction_days: int, delta_minutes: int) -> tuple[datetime, datetime, int]:
    """Calculate start and end dates for prediction."""
    if start_date is None:
        start_date = pd.Timestamp.now().floor("s")
    end_date: datetime = start_date + timedelta(days=prediction_days)
    return start_date, end_date, delta_minutes
