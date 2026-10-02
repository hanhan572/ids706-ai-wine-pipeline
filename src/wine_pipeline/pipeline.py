"""Orchestrate the wine-quality analysis and write artifacts after it succeeds."""

import json
from pathlib import Path

from wine_pipeline.clean import remove_duplicates
from wine_pipeline.load import load_data
from wine_pipeline.model import train_and_evaluate
from wine_pipeline.outliers import build_outlier_report
from wine_pipeline.schema import MIN_ROWS
from wine_pipeline.summarize import summarize_by_type
from wine_pipeline.validate import (
    missing_value_report,
    validate_columns,
    validate_no_missing,
    validate_numeric,
    validate_type,
)
from wine_pipeline.visualize import create_scatter_plot

METRIC_KEYS = (
    "rows_read",
    "rows_after_dedup",
    "duplicates_removed",
    "missing_values_total",
    "outlier_count_total",
    "rows_modeled",
    "test_size",
    "random_state",
    "iqr_multiplier",
    "n_train",
    "n_test",
    "mae",
    "r2",
)


def run_pipeline(
    input_path,
    output_dir,
    test_size: float = 0.2,
    random_state: int = 42,
    iqr_multiplier: float = 1.5,
) -> dict:
    """Load, validate, summarize, and model a wine CSV, then write artifacts.

    Exact duplicates are removed before ``validate_type``, so type case and
    whitespace do not change row identity. Missing type values are counted
    earlier, on the original column. Nothing is written until validation,
    duplicate removal, the minimum-row check, and modeling succeed.
    """
    frame = load_data(input_path)
    rows_read = len(frame)
    validate_columns(frame)
    frame = validate_numeric(frame)
    missing_report = missing_value_report(frame)
    validate_no_missing(missing_report)
    frame, duplicates_removed = remove_duplicates(frame)
    rows_after_dedup = len(frame)
    if rows_after_dedup < MIN_ROWS:
        raise ValueError(
            f"Need at least {MIN_ROWS} rows after duplicate removal, got {rows_after_dedup}"
        )
    frame = validate_type(frame)

    outlier_report = build_outlier_report(frame, multiplier=iqr_multiplier)
    summary = summarize_by_type(frame)
    model_metrics = train_and_evaluate(
        frame,
        test_size=test_size,
        random_state=random_state,
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    plot_path = create_scatter_plot(frame, destination / "alcohol_vs_quality.png")
    summary_path = destination / "summary.csv"
    missing_path = destination / "missing_values.csv"
    outliers_path = destination / "outliers_report.csv"
    metrics_path = destination / "metrics.json"
    summary.to_csv(summary_path, index=False)
    missing_report.to_csv(missing_path, index=False)
    outlier_report.to_csv(outliers_path, index=False)

    metrics = {
        "rows_read": int(rows_read),
        "rows_after_dedup": int(rows_after_dedup),
        "duplicates_removed": int(duplicates_removed),
        "missing_values_total": int(missing_report["missing_count"].sum()),
        "outlier_count_total": int(outlier_report["outlier_count"].sum()),
        "rows_modeled": int(rows_after_dedup),
        "test_size": float(test_size),
        "random_state": int(random_state),
        "iqr_multiplier": float(iqr_multiplier),
        "n_train": int(model_metrics["n_train"]),
        "n_test": int(model_metrics["n_test"]),
        "mae": float(model_metrics["mae"]),
        "r2": float(model_metrics["r2"]),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return {
        **metrics,
        "output_dir": str(destination),
        "summary_path": str(summary_path),
        "missing_values_path": str(missing_path),
        "outliers_report_path": str(outliers_path),
        "metrics_path": str(metrics_path),
        "plot_path": str(plot_path),
    }
