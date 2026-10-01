"""Wine-type summary statistics."""

import pandas as pd

from wine_pipeline.schema import HIGH_QUALITY_THRESHOLD, TARGET_COLUMN, TYPE_COLUMN

SUMMARY_COLUMNS = [
    "type",
    "count",
    "average_quality",
    "median_quality",
    "average_alcohol",
    "high_quality_count",
]


def summarize_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize quality and alcohol by normalized wine type.

    ``type`` is stripped and lowercased before grouping. Rows with
    ``quality >= 7`` contribute to ``high_quality_count``.
    """
    work = df.copy()
    work[TYPE_COLUMN] = work[TYPE_COLUMN].astype("string").str.strip().str.lower()

    records: list[dict[str, object]] = []
    for wine_type, group in work.groupby(TYPE_COLUMN, sort=True):
        quality = group[TARGET_COLUMN]
        records.append(
            {
                "type": str(wine_type),
                "count": int(len(group)),
                "average_quality": float(quality.mean()),
                "median_quality": float(quality.median()),
                "average_alcohol": float(group["alcohol"].mean()),
                "high_quality_count": int((quality >= HIGH_QUALITY_THRESHOLD).sum()),
            }
        )
    return pd.DataFrame(records, columns=SUMMARY_COLUMNS)
