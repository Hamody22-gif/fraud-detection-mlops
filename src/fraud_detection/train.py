"""Train and evaluate the XGBoost model."""

import joblib
from sklearn.metrics import average_precision_score, classification_report

from fraud_detection.config import MODELS_DIR, TEST_CSV, TRAIN_CSV
from fraud_detection.data import load_raw_data
from fraud_detection.features import make_xy
from fraud_detection.model import build_xgboost_model


def main() -> None:
    model = build_xgboost_model()

    # Train, then free the training data before loading the test set
    X_train, y_train = make_xy(load_raw_data(TRAIN_CSV))
    model.fit(X_train, y_train)
    del X_train, y_train

    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / "xgboost.joblib"
    joblib.dump(model, model_path)
    print(f"Saved model → {model_path}")

    # Evaluate on the held-out test set
    X_test, y_test = make_xy(load_raw_data(TEST_CSV))
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    pr_auc = average_precision_score(y_test, y_proba)
    print(f"\nPR-AUC: {pr_auc:.4f}\n")
    print(classification_report(y_test, y_pred, digits=4))


if __name__ == "__main__":
    main()
