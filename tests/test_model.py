"""Tests for linear regression training and evaluation."""

import pandas as pd
import pytest

from wine_pipeline.model import train_and_evaluate
from wine_pipeline.schema import MIN_ROWS


def test_model_reproducible(sample_wine_df: pd.DataFrame) -> None:
    first = train_and_evaluate(sample_wine_df, test_size=0.2, random_state=42)
    second = train_and_evaluate(sample_wine_df, test_size=0.2, random_state=42)
    assert first["mae"] == second["mae"]
    assert first["r2"] == second["r2"]
    assert first["n_train"] == second["n_train"]
    assert first["n_test"] == second["n_test"]
    assert first["n_train"] + first["n_test"] == len(sample_wine_df)


def test_model_excludes_type(sample_wine_df: pd.DataFrame) -> None:
    altered = sample_wine_df.copy()
    altered["type"] = "not-a-real-type"
    assert train_and_evaluate(sample_wine_df) == train_and_evaluate(altered)


def test_train_rejects_small_dataset(sample_wine_df: pd.DataFrame) -> None:
    small = sample_wine_df.iloc[: MIN_ROWS - 1]
    with pytest.raises(ValueError) as exc_info:
        train_and_evaluate(small)
    message = str(exc_info.value)
    assert str(MIN_ROWS) in message
    assert str(len(small)) in message


@pytest.mark.parametrize("test_size", [0, 1, -0.1, 1.5])
def test_train_rejects_test_size_outside_open_interval(
    sample_wine_df: pd.DataFrame,
    test_size: float,
) -> None:
    with pytest.raises(ValueError):
        train_and_evaluate(sample_wine_df, test_size=test_size)
