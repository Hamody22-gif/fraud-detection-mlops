"""Data loading."""

from pathlib import Path

import pandas as pd


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load a raw transactions CSV into a DataFrame.

    The first unnamed column is just a row counter, so we use it as the
    index (index_col=0) instead of treating it as a feature.
    """
    return pd.read_csv(path, index_col=0)
