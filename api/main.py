"""FastAPI app serving the fraud-detection model."""

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from fraud_detection.config import (
    CATEGORICAL_FEATURES,
    DECISION_THRESHOLD,
    MODELS_DIR,
    NUMERIC_FEATURES,
    PASSTHROUGH_FEATURES,
)
from fraud_detection.features import add_age, add_time_features


class Transaction(BaseModel):
    """One transaction to score. These are the raw fields the model needs."""

    amt: float
    category: str
    gender: str
    trans_date_trans_time: str  # e.g. "2020-06-21 03:14:25"
    dob: str  # date of birth, e.g. "1968-03-19"
    time_since_last: float | None = None  # seconds since this card's previous txn (optional)


class Prediction(BaseModel):
    """What the API returns for one transaction."""

    fraud_probability: float
    is_fraud: bool
    threshold: float


app = FastAPI(title="Fraud Detection API")


# Loaded once at startup, reused for every request
model = joblib.load(MODELS_DIR / "xgboost.joblib")


@app.post("/predict", response_model=Prediction)
def predict(txn: Transaction) -> Prediction:
    # One-row DataFrame from the request
    df = pd.DataFrame([txn.model_dump()])
    # Force numeric dtype so the imputer/scaler work even when time_since_last is null
    df["time_since_last"] = df["time_since_last"].astype("float64")

    # SAME feature engineering as training → no train/serve skew
    df = add_time_features(df)  # hour, is_night, day_of_week
    df = add_age(df)  # age

    # Exactly the columns the pipeline expects
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + PASSTHROUGH_FEATURES]

    proba = float(model.predict_proba(X)[:, 1][0])
    return Prediction(
        fraud_probability=proba,
        is_fraud=proba >= DECISION_THRESHOLD,
        threshold=DECISION_THRESHOLD,
    )
