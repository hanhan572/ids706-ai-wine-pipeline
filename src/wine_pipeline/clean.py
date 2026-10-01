"""Exact-duplicate removal for wine-quality observations."""

import pandas as pd


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate rows and return the cleaned frame plus the count.

    Near-duplicates are kept. The count is taken before dropping, and the
    returned frame has a reset index.
    """
    duplicate_count = int(df.duplicated().sum())
    cleaned = df.drop_duplicates().reset_index(drop=True)
    return cleaned, duplicate_count
