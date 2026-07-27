"""Shared configuration: file paths and constants."""

from pathlib import Path

# This file is at src/fraud_detection/config.py.
# .parents[2] walks up 3 levels: config.py → fraud_detection → src → PROJECT ROOT.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_CSV = RAW_DATA_DIR / "fraudTrain.csv"
TEST_CSV = RAW_DATA_DIR / "fraudTest.csv"

# Hours considered "night" — fraud rate was ~18-20x higher here (from EDA)
NIGHT_HOURS = [22, 23, 0, 1, 2, 3]


# The column we're predicting
TARGET = "is_fraud"
