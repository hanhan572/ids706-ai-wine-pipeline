"""Tests for exact duplicate removal."""

import pandas as pd

from wine_pipeline.clean import remove_duplicates


def test_remove_duplicates_drops_exact_copies(duplicated_wine_df: pd.DataFrame) -> None:
    original_length = len(duplicated_wine_df)
    cleaned, removed = remove_duplicates(duplicated_wine_df)
    assert removed == 1
    assert len(cleaned) == original_length - 1
    assert len(duplicated_wine_df) == original_length
    assert int(cleaned.duplicated().sum()) == 0
    assert list(cleaned.index) == list(range(len(cleaned)))
    pd.testing.assert_series_equal(
        cleaned.iloc[0],
        duplicated_wine_df.iloc[0],
        check_names=False,
    )


def test_remove_duplicates_keeps_unique_rows(sample_wine_df: pd.DataFrame) -> None:
    cleaned, removed = remove_duplicates(sample_wine_df)
    assert removed == 0
    assert len(cleaned) == len(sample_wine_df)
    pd.testing.assert_frame_equal(cleaned, sample_wine_df.reset_index(drop=True))


def test_remove_duplicates_keeps_near_duplicates(sample_wine_df: pd.DataFrame) -> None:
    near = sample_wine_df.iloc[[0]].copy()
    near["alcohol"] = near["alcohol"] + 0.1
    combined = pd.concat([sample_wine_df, near], ignore_index=True)
    cleaned, removed = remove_duplicates(combined)
    assert removed == 0
    assert len(cleaned) == len(combined)
