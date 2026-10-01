"""Column, numeric, and missing-value checks for the wine-quality schema."""

import pandas as pd

from wine_pipeline.schema import NUMERIC_COLUMNS, REQUIRED_COLUMNS, TYPE_COLUMN

ALLOWED_WINE_TYPES = frozenset({"red", "white"})


def validate_columns(df: pd.DataFrame) -> None:
    """Require the column set to equal the 13 required names."""
    actual = set(df.columns)
    required = set(REQUIRED_COLUMNS)
    missing = sorted(required - actual)
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"Missing required columns: {names}")
    extra = sorted(actual - required)
    if extra:
        names = ", ".join(extra)
        raise ValueError(f"Unexpected columns: {names}")


def validate_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with numeric columns coerced.

    A non-null value that cannot be coerced raises ``ValueError``. Existing
    missing values stay missing so they can be counted later.
    """
    result = df.copy()
    for name in NUMERIC_COLUMNS:
        original = result[name]
        converted = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & converted.isna()
        if invalid.any():
            raise ValueError(f"Non-numeric values in column: {name}")
        result[name] = converted
    return result


def validate_type(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with ``type`` stripped and lowercased.

    Missing values stay missing. Any other non-null value raises ``ValueError``.
    """
    result = df.copy()
    normalized = result[TYPE_COLUMN].astype("string").str.strip().str.lower()
    invalid = normalized.notna() & ~normalized.isin(ALLOWED_WINE_TYPES)
    if invalid.any():
        bad = ", ".join(sorted(normalized.loc[invalid].unique()))
        raise ValueError(f"Invalid type values: {bad}")
    result[TYPE_COLUMN] = normalized
    return result


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Count missing values for each required column, in schema order."""
    rows = [
        {"column": name, "missing_count": int(df[name].isna().sum())}
        for name in REQUIRED_COLUMNS
    ]
    return pd.DataFrame(rows, columns=["column", "missing_count"])


def validate_no_missing(report: pd.DataFrame) -> None:
    """Raise when any required column has a positive missing count."""
    affected = report.loc[report["missing_count"] > 0]
    if affected.empty:
        return
    pairs = ", ".join(
        f"{row.column}={int(row.missing_count)}"
        for row in affected.itertuples(index=False)
    )
    raise ValueError(f"Missing values found: {pairs}")
