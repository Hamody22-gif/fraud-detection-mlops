"""DVC stage 2 — train and evaluate the XGBoost model on the processed data."""

import json

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_score,
    recall_score,
)

from fraud_detection.config import (
    DECISION_THRESHOLD,
    METRICS_FILE,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    TARGET,
)
from fraud_detection.model import build_xgboost_model


def main() -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("fraud-detection")

    with mlflow.start_run(run_name="xgboost-tuned"):
        model = build_xgboost_model()

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

        # --- Train on the processed TRAIN table (featurize produced this) ---
        train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
        y_train = train_df[TARGET]
        X_train = train_df.drop(columns=[TARGET])
        model.fit(X_train, y_train)
        del X_train, y_train, train_df  # free memory before loading test

        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(model, MODELS_DIR / "xgboost.joblib")

        # --- Evaluate on the processed TEST table ---
        test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
        y_test = test_df[TARGET]
        X_test = test_df.drop(columns=[TARGET])
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)

        pr_auc = float(average_precision_score(y_test, y_proba))
        precision = float(precision_score(y_test, y_pred))
        recall = float(recall_score(y_test, y_pred))

        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # --- Write metrics.json for DVC to track (enables `dvc metrics show`) ---
        METRICS_FILE.write_text(
            json.dumps({"pr_auc": pr_auc, "precision": precision, "recall": recall}, indent=2)
        )

        mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name="fraud-detection-model",
            skops_trusted_types=[
                "numpy.dtype",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
            ],
        )

        print(f"PR-AUC={pr_auc:.4f}  precision={precision:.4f}  recall={recall:.4f}")
        print(classification_report(y_test, y_pred, digits=4))


if __name__ == "__main__":
    main()
