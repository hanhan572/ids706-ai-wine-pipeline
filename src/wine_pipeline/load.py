"""Read a wine-quality CSV without applying schema rules."""

from pathlib import Path

import pandas as pd


def load_data(path: str | Path) -> pd.DataFrame:
    """Load a CSV and strip whitespace from column names.

    Schema checks belong in ``validate.py``. A missing file raises
    ``FileNotFoundError`` and includes ``path`` in the message.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Input file not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    frame.columns = frame.columns.str.strip()
    return frame
