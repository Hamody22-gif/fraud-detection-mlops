"""Model definitions."""

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from fraud_detection.preprocess import build_preprocessor


def build_xgboost_model() -> Pipeline:
    """XGBoost model: preprocessing + gradient-boosted trees, as one Pipeline."""
    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            (
                "clf",
                XGBClassifier(
                    n_estimators=500,
                    max_depth=10,
                    learning_rate=0.03,
                    subsample=0.7,
                    colsample_bytree=0.7,
                    min_child_weight=1,
                    scale_pos_weight=171,
                    eval_metric="aucpr",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )
