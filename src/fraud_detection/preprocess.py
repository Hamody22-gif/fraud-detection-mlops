"""Preprocessing: scale numeric features, one-hot encode categorical ones."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraud_detection.config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PASSTHROUGH_FEATURES,
)


def build_preprocessor() -> ColumnTransformer:
    """Build the preprocessing transformer.

    - numeric features  -> StandardScaler (level the ranges)
    - categorical       -> OneHotEncoder  (text -> 0/1 columns)
    - passthrough       -> kept as-is
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("pass", "passthrough", PASSTHROUGH_FEATURES),
        ],
    )
