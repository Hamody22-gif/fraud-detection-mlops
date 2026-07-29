"""Hyperparameter tuning for the XGBoost model."""

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from fraud_detection.config import TRAIN_CSV
from fraud_detection.data import load_raw_data
from fraud_detection.features import make_xy
from fraud_detection.model import build_xgboost_model


def main() -> None:
    X, y = make_xy(load_raw_data(TRAIN_CSV))

    # Subsample to keep the search fast + memory-safe on 8 GB RAM
    X_sample = X.sample(n=150_000, random_state=42)
    y_sample = y.loc[X_sample.index]

    # Settings to search over. "clf__" targets the classifier step in the Pipeline.
    param_dist = {
        "clf__n_estimators": [200, 300, 500],
        "clf__max_depth": [4, 6, 8, 10],
        "clf__learning_rate": [0.03, 0.05, 0.1, 0.2],
        "clf__subsample": [0.7, 0.85, 1.0],
        "clf__colsample_bytree": [0.7, 0.85, 1.0],
        "clf__min_child_weight": [1, 3, 5],
    }

    search = RandomizedSearchCV(
        build_xgboost_model(),
        param_distributions=param_dist,
        n_iter=12,  # try 12 random combinations
        scoring="average_precision",  # = PR-AUC, our metric
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        n_jobs=1,  # sequential — keeps memory low on 8 GB
        refit=False,
        verbose=2,
        random_state=42,
    )
    search.fit(X_sample, y_sample)

    print(f"\nBest CV PR-AUC: {search.best_score_:.4f}")
    print("Best params:")
    for key, val in search.best_params_.items():
        print(f"  {key} = {val}")


if __name__ == "__main__":
    main()
