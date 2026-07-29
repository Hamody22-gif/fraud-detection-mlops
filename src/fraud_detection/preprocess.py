"""Preprocessing: scale numeric features, one-hot encode categorical ones."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraud_detection.config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PASSTHROUGH_FEATURES,
)


def build_preprocessor() -> ColumnTransformer:
    """Scale numerics (with imputation), one-hot encode categoricals."""
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),  # fill NaN (e.g. first txn)
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("pass", "passthrough", PASSTHROUGH_FEATURES),
        ],
    )
