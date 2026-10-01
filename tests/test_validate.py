"""Tests for schema, numeric, and missing-value validation."""

import pandas as pd
import pytest

from wine_pipeline.schema import REQUIRED_COLUMNS
from wine_pipeline.validate import (
    missing_value_report,
    validate_columns,
    validate_no_missing,
    validate_numeric,
    validate_type,
)


def test_validate_columns_accepts_required_set(sample_wine_df: pd.DataFrame) -> None:
    validate_columns(sample_wine_df)


@pytest.mark.parametrize("column_name", ["quality", "type"])
def test_validate_missing_column(sample_wine_df: pd.DataFrame, column_name: str) -> None:
    frame = sample_wine_df.drop(columns=[column_name])
    with pytest.raises(ValueError) as exc_info:
        validate_columns(frame)
    message = str(exc_info.value)
    assert message.startswith("Missing required columns:")
    assert column_name in message


def test_validate_unexpected_column(sample_wine_df: pd.DataFrame) -> None:
    frame = sample_wine_df.copy()
    frame["region"] = "napa"
    with pytest.raises(ValueError) as exc_info:
        validate_columns(frame)
    message = str(exc_info.value)
    assert message.startswith("Unexpected columns:")
    assert "region" in message


def test_validate_non_numeric_feature(sample_wine_df: pd.DataFrame) -> None:
    frame = sample_wine_df.astype({"alcohol": object})
    frame.loc[0, "alcohol"] = "high"
    with pytest.raises(ValueError) as exc_info:
        validate_numeric(frame)
    assert str(exc_info.value) == "Non-numeric values in column: alcohol"


def test_validate_numeric_coerces_numeric_strings(sample_wine_df: pd.DataFrame) -> None:
    frame = sample_wine_df.copy()
    frame["quality"] = frame["quality"].astype(str)
    before = frame.copy()
    result = validate_numeric(frame)
    assert result["quality"].isna().sum() == 0
    assert pd.api.types.is_numeric_dtype(result["quality"])
    pd.testing.assert_series_equal(
        result["quality"],
        pd.to_numeric(before["quality"]),
        check_names=False,
    )
    pd.testing.assert_frame_equal(frame, before)


def test_validate_missing_value(missing_value_wine_df: pd.DataFrame) -> None:
    coerced = validate_numeric(missing_value_wine_df)
    report = missing_value_report(coerced)
    assert list(report["column"]) == list(REQUIRED_COLUMNS)
    ph_count = int(report.loc[report["column"] == "pH", "missing_count"].iloc[0])
    assert ph_count == 1
    assert int(report["missing_count"].sum()) == 1
    with pytest.raises(ValueError) as exc_info:
        validate_no_missing(report)
    message = str(exc_info.value)
    assert message.startswith("Missing values found:")
    assert "pH=1" in message


def test_validate_no_missing_accepts_complete_report(sample_wine_df: pd.DataFrame) -> None:
    report = missing_value_report(sample_wine_df)
    assert (report["missing_count"] == 0).all()
    validate_no_missing(report)


def test_validate_type_normalizes_case_and_whitespace(sample_wine_df: pd.DataFrame) -> None:
    frame = sample_wine_df.copy()
    frame.loc[0, "type"] = " Red "
    frame.loc[1, "type"] = "WHITE"
    before = frame.copy()
    result = validate_type(frame)
    assert result.loc[0, "type"] == "red"
    assert result.loc[1, "type"] == "white"
    assert set(result["type"].unique()) == {"red", "white"}
    pd.testing.assert_frame_equal(frame, before)


def test_validate_type_rejects_unexpected_value(sample_wine_df: pd.DataFrame) -> None:
    frame = sample_wine_df.copy()
    frame.loc[0, "type"] = "rose"
    before = frame.copy()
    with pytest.raises(ValueError) as exc_info:
        validate_type(frame)
    message = str(exc_info.value)
    assert message.startswith("Invalid type values:")
    assert "rose" in message
    pd.testing.assert_frame_equal(frame, before)
