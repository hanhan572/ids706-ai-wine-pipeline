"""Tests for wine-type summaries."""

import pandas as pd
import pytest

from wine_pipeline.summarize import SUMMARY_COLUMNS, summarize_by_type


def test_summarize_both_types(sample_wine_df: pd.DataFrame) -> None:
    frame = sample_wine_df.copy()
    frame.loc[0, "type"] = " Red "
    before = frame.copy()
    summary = summarize_by_type(frame)

    assert list(summary.columns) == SUMMARY_COLUMNS
    assert list(summary["type"]) == ["red", "white"]
    assert list(summary["count"]) == [6, 6]
    red = summary.loc[summary["type"] == "red"].iloc[0]
    white = summary.loc[summary["type"] == "white"].iloc[0]
    assert red["high_quality_count"] == 2
    assert white["high_quality_count"] == 4
    assert red["average_quality"] == 6
    assert white["average_quality"] == 7
    assert red["median_quality"] == 6
    assert white["median_quality"] == 7
    assert red["average_alcohol"] == pytest.approx(11)
    assert white["average_alcohol"] == pytest.approx(11.4)
    pd.testing.assert_frame_equal(frame, before)


def test_summarize_single_type(sample_wine_df: pd.DataFrame) -> None:
    red_only = sample_wine_df.loc[sample_wine_df["type"] == "red"].reset_index(drop=True)
    summary = summarize_by_type(red_only)

    assert len(summary) == 1
    assert summary.loc[0, "type"] == "red"
    assert summary.loc[0, "count"] == len(red_only)
    assert summary.loc[0, "high_quality_count"] == 2
