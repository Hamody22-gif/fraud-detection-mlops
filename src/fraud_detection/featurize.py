"""DVC stage 1 — turn raw CSVs into processed feature tables."""

from fraud_detection.config import PROCESSED_DATA_DIR, TARGET, TEST_CSV, TRAIN_CSV
from fraud_detection.data import load_raw_data
from fraud_detection.features import make_xy


def main() -> None:
    # Make sure data/processed/ exists (parents=True in case data/ is missing too)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Do the same transform for both raw files
    for raw_path, name in [(TRAIN_CSV, "train"), (TEST_CSV, "test")]:
        X, y = make_xy(load_raw_data(raw_path))

        # Recombine features + target into one table so each processed file is self-contained
        processed = X.copy()
        processed[TARGET] = y

        # index=False: the row numbers aren't meaningful, don't write them
        processed.to_csv(PROCESSED_DATA_DIR / f"{name}.csv", index=False)
        print(f"wrote {name}.csv  rows={len(processed)}  cols={processed.shape[1]}")


if __name__ == "__main__":
    main()
