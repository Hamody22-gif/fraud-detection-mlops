"""Tests for feature engineering."""

import pandas as pd

from fraud_detection.features import add_age, add_time_features


def test_add_time_features_computes_correct_values() -> None:
    # Arrange: two transactions with known timestamps
    #   2019-01-01 02:30 -> Tuesday, hour 2  (night)
    #   2019-06-15 14:00 -> Saturday, hour 14 (day)
    df = pd.DataFrame({"trans_date_trans_time": ["2019-01-01 02:30:00", "2019-06-15 14:00:00"]})

    # Act
    result = add_time_features(df)

    # Assert
    assert list(result["hour"]) == [2, 14]
    assert list(result["is_night"]) == [1, 0]  # 2am=night, 2pm=day
    assert list(result["day_of_week"]) == [1, 5]  # Tue=1, Sat=5


def test_add_time_features_does_not_mutate_input() -> None:
    df = pd.DataFrame({"trans_date_trans_time": ["2019-01-01 02:30:00"]})
    add_time_features(df)
    # The original df must NOT have gained the new columns
    assert "hour" not in df.columns


def test_add_age_computes_correct_age() -> None:
    # Born 1990-06-15, transaction on 2020-12-25 -> age 30
    df = pd.DataFrame(
        {
            "trans_date_trans_time": ["2020-12-25 00:00:00"],
            "dob": ["1990-06-15"],
        }
    )
    result = add_age(df)
    assert result["age"].iloc[0] == 30
