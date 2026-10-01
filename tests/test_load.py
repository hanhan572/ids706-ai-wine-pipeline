"""Tests for CSV loading."""

from pathlib import Path

import pandas as pd
import pytest

from wine_pipeline.load import load_data


def test_load_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.csv"
    with pytest.raises(FileNotFoundError) as exc_info:
        load_data(missing)
    assert str(missing) in str(exc_info.value)


def test_load_reads_rows(tmp_path: Path, sample_wine_df: pd.DataFrame) -> None:
    path = tmp_path / "wine.csv"
    sample_wine_df.to_csv(path, index=False)
    loaded = load_data(path)
    assert list(loaded.columns) == list(sample_wine_df.columns)
    assert len(loaded) == len(sample_wine_df)
    pd.testing.assert_series_equal(
        loaded["alcohol"],
        sample_wine_df["alcohol"],
        check_names=False,
    )


def test_load_strips_column_whitespace(tmp_path: Path, sample_wine_df: pd.DataFrame) -> None:
    path = tmp_path / "padded.csv"
    padded = sample_wine_df.rename(columns={"alcohol": " alcohol ", "type": " type "})
    padded.to_csv(path, index=False)
    loaded = load_data(path)
    assert "alcohol" in loaded.columns
    assert "type" in loaded.columns
    assert " alcohol " not in loaded.columns
    assert " type " not in loaded.columns
