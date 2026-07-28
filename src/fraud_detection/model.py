"""Model definitions."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fraud_detection.preprocess import build_preprocessor


def build_logreg_model() -> Pipeline:
    """Baseline model: preprocessing + Logistic Regression, as one Pipeline."""
    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
