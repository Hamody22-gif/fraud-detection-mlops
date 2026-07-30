"""Shared configuration: file paths and constants."""

from pathlib import Path

# This file is at src/fraud_detection/config.py.
# .parents[2] walks up 3 levels: config.py → fraud_detection → src → PROJECT ROOT.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_CSV = RAW_DATA_DIR / "fraudTrain.csv"
TEST_CSV = RAW_DATA_DIR / "fraudTest.csv"
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
METRICS_FILE = PROJECT_ROOT / "metrics.json"


# Hours considered "night" — fraud rate was ~18-20x higher here (from EDA)
NIGHT_HOURS = [22, 23, 0, 1, 2, 3]


# The column we're predicting
TARGET = "is_fraud"
# Modeling features, grouped by how they'll be preprocessed (from EDA)
NUMERIC_FEATURES = ["amt", "age", "hour", "day_of_week", "time_since_last"]  # will be scaled
CATEGORICAL_FEATURES = ["category", "gender"]  # will be one-hot encoded
PASSTHROUGH_FEATURES = ["is_night"]  # already 0/1, leave as-is

# Chosen decision threshold (tuned on the PR curve — balances recall vs. false alarms)
DECISION_THRESHOLD = 0.95
