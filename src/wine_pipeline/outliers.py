"""IQR outlier counts. Rows are reported and kept."""

import pandas as pd

from wine_pipeline.schema import NUMERIC_COLUMNS

REPORT_COLUMNS = [
    "column",
    "q1",
    "q3",
    "iqr",
    "lower_fence",
    "upper_fence",
    "outlier_count",
]


def build_outlier_report(df: pd.DataFrame, multiplier: float = 1.5) -> pd.DataFrame:
    """Count IQR outliers for each numeric column, including quality.

    The caller's frame is not filtered, clipped, or otherwise modified.
    A multiplier less than or equal to zero raises ``ValueError``.
    """
    if multiplier <= 0:
        raise ValueError(f"iqr_multiplier must be greater than 0, got {multiplier}")

    rows: list[dict[str, object]] = []
    for name in NUMERIC_COLUMNS:
        series = df[name]
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower_fence = q1 - multiplier * iqr
        upper_fence = q3 + multiplier * iqr
        outlier_count = int(((series < lower_fence) | (series > upper_fence)).sum())
        rows.append(
            {
                "column": name,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_fence": lower_fence,
                "upper_fence": upper_fence,
                "outlier_count": outlier_count,
            }
        )
    return pd.DataFrame(rows, columns=REPORT_COLUMNS)
