"""Train/test splitting."""

import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_detection.config import TARGET


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into train/test, preserving the fraud ratio in both.

    Returns (train_df, test_df).
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[TARGET],  # keep the 0.58% fraud ratio in both halves
        random_state=random_state,
    )
    return train_df, test_df
