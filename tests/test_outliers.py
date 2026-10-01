"""Tests for IQR outlier reporting."""

import pandas as pd
import pytest

from wine_pipeline.outliers import build_outlier_report
from wine_pipeline.schema import NUMERIC_COLUMNS


def test_outlier_constant_column(sample_wine_df: pd.DataFrame) -> None:
    frame = sample_wine_df.copy()
    frame["density"] = 1.0
    before = frame.copy()
    report = build_outlier_report(frame)

    assert list(report["column"]) == list(NUMERIC_COLUMNS)
    row = report.loc[report["column"] == "density"].iloc[0]
    assert row["iqr"] == pytest.approx(0)
    assert row["lower_fence"] == pytest.approx(row["q1"])
    assert row["upper_fence"] == pytest.approx(row["q3"])
    assert row["outlier_count"] == 0
    assert len(frame) == len(before)
    pd.testing.assert_frame_equal(frame, before)


def test_outlier_extreme_value_is_kept(
    sample_wine_df: pd.DataFrame,
    extreme_alcohol_wine_df: pd.DataFrame,
) -> None:
    baseline = build_outlier_report(sample_wine_df)
    before = extreme_alcohol_wine_df.copy()
    length_before = len(extreme_alcohol_wine_df)

    report = build_outlier_report(extreme_alcohol_wine_df)

    assert len(extreme_alcohol_wine_df) == length_before
    pd.testing.assert_frame_equal(extreme_alcohol_wine_df, before)
    assert extreme_alcohol_wine_df.loc[0, "alcohol"] == 100.0
    baseline_count = int(baseline.loc[baseline["column"] == "alcohol", "outlier_count"].iloc[0])
    extreme_count = int(report.loc[report["column"] == "alcohol", "outlier_count"].iloc[0])
    assert extreme_count > baseline_count


def test_outlier_report_uses_multiplier(sample_wine_df: pd.DataFrame) -> None:
    multiplier = 2.0
    report = build_outlier_report(sample_wine_df, multiplier=multiplier)
    for row in report.itertuples(index=False):
        assert row.iqr == pytest.approx(row.q3 - row.q1)
        assert row.lower_fence == pytest.approx(row.q1 - multiplier * row.iqr)
        assert row.upper_fence == pytest.approx(row.q3 + multiplier * row.iqr)


@pytest.mark.parametrize("multiplier", [0, -1.5])
def test_outlier_rejects_non_positive_multiplier(
    sample_wine_df: pd.DataFrame,
    multiplier: float,
) -> None:
    with pytest.raises(ValueError):
        build_outlier_report(sample_wine_df, multiplier=multiplier)
