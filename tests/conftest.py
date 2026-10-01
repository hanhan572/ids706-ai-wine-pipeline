"""Synthetic wine frames for unit tests. Do not read the full dataset here."""

import pandas as pd
import pytest

from wine_pipeline.schema import FEATURE_COLUMNS, REQUIRED_COLUMNS, TARGET_COLUMN, TYPE_COLUMN


@pytest.fixture
def sample_wine_df() -> pd.DataFrame:
    """At least 10 unique, complete rows covering both wine types."""
    rows: list[dict[str, object]] = []
    for i in range(12):
        row: dict[str, object] = {
            name: float(i + 1) + offset for offset, name in enumerate(FEATURE_COLUMNS)
        }
        row["alcohol"] = 9.0 + (i * 0.4)
        row[TARGET_COLUMN] = 4 + (i % 6)
        row[TYPE_COLUMN] = "red" if i % 2 == 0 else "white"
        rows.append(row)
    return pd.DataFrame(rows, columns=list(REQUIRED_COLUMNS))


@pytest.fixture
def duplicated_wine_df(sample_wine_df: pd.DataFrame) -> pd.DataFrame:
    """The sample frame plus one exact copy of its first row."""
    extra = sample_wine_df.iloc[[0]]
    return pd.concat([sample_wine_df, extra], ignore_index=True)


@pytest.fixture
def missing_value_wine_df(sample_wine_df: pd.DataFrame) -> pd.DataFrame:
    """The sample frame with one missing pH value."""
    frame = sample_wine_df.copy()
    frame.loc[0, "pH"] = float("nan")
    return frame


@pytest.fixture
def extreme_alcohol_wine_df(sample_wine_df: pd.DataFrame) -> pd.DataFrame:
    """The sample frame with one extreme alcohol value."""
    frame = sample_wine_df.copy()
    frame.loc[0, "alcohol"] = 100.0
    return frame
