"""Tests for preprocessing."""

import pandas as pd

from fraud_detection.preprocess import build_preprocessor


def test_preprocessor_output_shape() -> None:
    df = pd.DataFrame(
        {
            "amt": [10.0, 500.0, 25.0],
            "age": [30, 45, 60],
            "hour": [2, 14, 23],
            "day_of_week": [1, 5, 3],
            "time_since_last": [100.0, float("nan"), 300.0],  # NaN → imputer should fill it
            "category": ["grocery_pos", "shopping_net", "grocery_pos"],
            "gender": ["M", "F", "M"],
            "is_night": [1, 0, 1],
        }
    )
    pre = build_preprocessor()
    result = pre.fit_transform(df)
    # 5 numeric + 2 category + 2 gender + 1 passthrough = 10 columns
    assert result.shape == (3, 10)
