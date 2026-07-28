"""Tests for preprocessing."""

import pandas as pd

from fraud_detection.preprocess import build_preprocessor


def test_preprocessor_output_shape() -> None:
    # Tiny frame with all 7 feature columns; 2 categories, 2 genders
    df = pd.DataFrame(
        {
            "amt": [10.0, 500.0, 25.0],
            "age": [30, 45, 60],
            "hour": [2, 14, 23],
            "day_of_week": [1, 5, 3],
            "category": ["grocery_pos", "shopping_net", "grocery_pos"],
            "gender": ["M", "F", "M"],
            "is_night": [1, 0, 1],
        }
    )
    pre = build_preprocessor()
    result = pre.fit_transform(df)

    # 4 numeric + 2 category one-hot + 2 gender one-hot + 1 passthrough = 9 cols
    assert result.shape == (3, 9)
