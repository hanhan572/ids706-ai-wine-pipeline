"""Command-line entry point for the wine-quality pipeline."""

import argparse
import sys

from wine_pipeline.pipeline import run_pipeline


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the pipeline, and return a process exit code."""
    parser = argparse.ArgumentParser(description="Wine quality analysis pipeline")
    parser.add_argument("--input", required=True, help="Path to the wine-quality CSV")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to create and write analysis artifacts",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows in the test split, in the open interval (0, 1)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the train/test split",
    )
    parser.add_argument(
        "--iqr-multiplier",
        type=float,
        default=1.5,
        help="Positive multiplier for IQR outlier fences",
    )
    args = parser.parse_args(argv)
    try:
        result = run_pipeline(
            input_path=args.input,
            output_dir=args.output_dir,
            test_size=args.test_size,
            random_state=args.random_state,
            iqr_multiplier=args.iqr_multiplier,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    print(
        f"rows_read={result['rows_read']} "
        f"duplicates_removed={result['duplicates_removed']} "
        f"mae={result['mae']} "
        f"r2={result['r2']} "
        f"output_dir={result['output_dir']}"
    )
    return 0
