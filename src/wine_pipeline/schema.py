"""Shared column names and thresholds for the wine-quality pipeline."""

REQUIRED_COLUMNS: tuple[str, ...] = (
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality",
    "type",
)

FEATURE_COLUMNS: tuple[str, ...] = REQUIRED_COLUMNS[:11]
NUMERIC_COLUMNS: tuple[str, ...] = FEATURE_COLUMNS + ("quality",)
TARGET_COLUMN = "quality"
TYPE_COLUMN = "type"
MIN_ROWS = 10
HIGH_QUALITY_THRESHOLD = 7
