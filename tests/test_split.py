"""Tests for train/test splitting."""

import pandas as pd

from fraud_detection.split import stratified_split


def test_stratified_split_preserves_rows_and_ratio() -> None:
    # Arrange: 100 rows with a clear 20% fraud rate (exaggerated for an easy check)
    df = pd.DataFrame(
        {
            "amt": range(100),
            "is_fraud": [1] * 20 + [0] * 80,
        }
    )

    # Act
    train_df, test_df = stratified_split(df, test_size=0.2)

    # Assert
    assert len(train_df) + len(test_df) == 100  # no rows lost
    assert len(test_df) == 20  # test_size respected (20%)
    assert train_df["is_fraud"].mean() == 0.2  # fraud ratio preserved
    assert test_df["is_fraud"].mean() == 0.2
