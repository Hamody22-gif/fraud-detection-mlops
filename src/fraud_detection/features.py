"""Feature engineering."""

import pandas as pd

from fraud_detection.config import (
    CATEGORICAL_FEATURES,
    NIGHT_HOURS,
    NUMERIC_FEATURES,
    PASSTHROUGH_FEATURES,
    TARGET,
)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, is_night, and day_of_week from the transaction timestamp.

    Returns a new DataFrame; the input is not modified.
    """
    df = df.copy()
    ts = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"] = ts.dt.hour
    df["is_night"] = ts.dt.hour.isin(NIGHT_HOURS).astype(int)
    df["day_of_week"] = ts.dt.dayofweek
    return df


def add_age(df: pd.DataFrame) -> pd.DataFrame:
    """Add cardholder age (in years) at the time of each transaction."""
    df = df.copy()
    ts = pd.to_datetime(df["trans_date_trans_time"])
    dob = pd.to_datetime(df["dob"])
    # Age = (transaction date - birth date) in days, converted to whole years
    df["age"] = ((ts - dob).dt.days / 365.25).astype(int)
    return df


def make_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Engineer features, then split into X (features) and y (target)."""
    df = add_time_features(df)
    df = add_age(df)
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + PASSTHROUGH_FEATURES
    return df[feature_cols], df[TARGET]
