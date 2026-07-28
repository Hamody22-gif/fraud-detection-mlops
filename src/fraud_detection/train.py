"""Train and evaluate the baseline model."""

from sklearn.metrics import average_precision_score, classification_report

from fraud_detection.config import TEST_CSV, TRAIN_CSV
from fraud_detection.data import load_raw_data
from fraud_detection.features import make_xy
from fraud_detection.model import build_logreg_model


def main() -> None:
    # Prepare data: train on fraudTrain (past), test on fraudTest (future)
    X_train, y_train = make_xy(load_raw_data(TRAIN_CSV))
    X_test, y_test = make_xy(load_raw_data(TEST_CSV))

    # Train the model
    model = build_logreg_model()
    model.fit(X_train, y_train)

    # Predict fraud probabilities on the held-out test set
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Evaluate
    pr_auc = average_precision_score(y_test, y_proba)
    print(f"\nPR-AUC: {pr_auc:.4f}\n")
    print(classification_report(y_test, y_pred, digits=4))


if __name__ == "__main__":
    main()
