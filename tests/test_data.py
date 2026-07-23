"""Tests for data loading."""

from pathlib import Path

from fraud_detection.data import load_raw_data


def test_load_raw_data_reads_csv(tmp_path: Path) -> None:
    # Arrange: write a tiny fake CSV shaped like the real file
    # (leading comma = the unnamed row-counter column)
    csv = tmp_path / "sample.csv"
    csv.write_text(",amt,is_fraud\n0,4.97,0\n1,107.23,1\n")

    # Act: load it through our function
    df = load_raw_data(csv)

    # Assert: 2 rows, and the counter column became the index (not a feature)
    assert df.shape == (2, 2)
    assert list(df.columns) == ["amt", "is_fraud"]
