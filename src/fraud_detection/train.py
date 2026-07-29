"""Train and evaluate the XGBoost model."""

import joblib
import mlflow
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_score,
    recall_score,
)

from fraud_detection.config import DECISION_THRESHOLD, MODELS_DIR, TEST_CSV, TRAIN_CSV
from fraud_detection.data import load_raw_data
from fraud_detection.features import make_xy
from fraud_detection.model import build_xgboost_model


def main() -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("fraud-detection")

    with mlflow.start_run(run_name="xgboost-tuned"):
        model = build_xgboost_model()

        # Log the hyperparameters we're using
        clf_params = model.named_steps["clf"].get_params()
        mlflow.log_params(
            {
                "n_estimators": clf_params["n_estimators"],
                "max_depth": clf_params["max_depth"],
                "learning_rate": clf_params["learning_rate"],
                "scale_pos_weight": clf_params["scale_pos_weight"],
                "threshold": DECISION_THRESHOLD,
            }
        )

        # Train
        X_train, y_train = make_xy(load_raw_data(TRAIN_CSV))
        model.fit(X_train, y_train)
        del X_train, y_train

        # Save locally too (the notebook/API load this)
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(model, MODELS_DIR / "xgboost.joblib")

        # Evaluate at our tuned threshold
        X_test, y_test = make_xy(load_raw_data(TEST_CSV))
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)

        pr_auc = average_precision_score(y_test, y_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # Log the metrics to MLflow
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        print(f"PR-AUC={pr_auc:.4f}  precision={precision:.4f}  recall={recall:.4f}")
        print(classification_report(y_test, y_pred, digits=4))


if __name__ == "__main__":
    main()
