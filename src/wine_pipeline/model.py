"""Train and evaluate a linear model of wine quality."""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from wine_pipeline.schema import FEATURE_COLUMNS, MIN_ROWS, TARGET_COLUMN


def train_and_evaluate(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, float | int]:
    """Fit linear regression and score the test split.

    Features are the 11 physicochemical columns. ``type`` is excluded.
    ``test_size`` must lie in the open interval (0, 1). Fewer than
    ``MIN_ROWS`` rows raises ``ValueError``.
    """
    if not 0 < test_size < 1:
        raise ValueError(
            f"test_size must be in the open interval (0, 1), got {test_size}"
        )
    row_count = len(df)
    if row_count < MIN_ROWS:
        raise ValueError(
            f"Need at least {MIN_ROWS} rows to train, got {row_count}"
        )

    features = df.loc[:, list(FEATURE_COLUMNS)]
    target = df[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    predictions = regressor.predict(x_test)
    return {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
    }
