"""Integration tests for pipeline orchestration and the command-line interface."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from wine_pipeline.cli import main
from wine_pipeline.pipeline import METRIC_KEYS, run_pipeline
from wine_pipeline.schema import NUMERIC_COLUMNS, REQUIRED_COLUMNS


def _write_frame(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False)
    return path


def _metrics_path(output_dir: Path) -> Path:
    return output_dir / "metrics.json"


def test_pipeline_missing_file(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    with pytest.raises(FileNotFoundError):
        run_pipeline(tmp_path / "missing.csv", output_dir)
    assert not _metrics_path(output_dir).exists()
    assert not output_dir.exists()


def test_pipeline_missing_column(tmp_path: Path, sample_wine_df: pd.DataFrame) -> None:
    source = _write_frame(sample_wine_df.drop(columns=["type"]), tmp_path / "wine.csv")
    output_dir = tmp_path / "output"
    with pytest.raises(ValueError) as exc_info:
        run_pipeline(source, output_dir)
    message = str(exc_info.value)
    assert message.startswith("Missing required columns:")
    assert "type" in message
    assert not _metrics_path(output_dir).exists()
    assert not output_dir.exists()


def test_pipeline_missing_value(tmp_path: Path, missing_value_wine_df: pd.DataFrame) -> None:
    source = _write_frame(missing_value_wine_df, tmp_path / "wine.csv")
    output_dir = tmp_path / "output"
    with pytest.raises(ValueError) as exc_info:
        run_pipeline(source, output_dir)
    assert str(exc_info.value).startswith("Missing values found:")
    assert not _metrics_path(output_dir).exists()
    assert not output_dir.exists()


def test_pipeline_rejects_invalid_type(tmp_path: Path, sample_wine_df: pd.DataFrame) -> None:
    frame = sample_wine_df.copy()
    frame.loc[0, "type"] = "rose"
    source = _write_frame(frame, tmp_path / "wine.csv")
    output_dir = tmp_path / "output"
    with pytest.raises(ValueError) as exc_info:
        run_pipeline(source, output_dir)
    assert str(exc_info.value).startswith("Invalid type values:")
    assert not _metrics_path(output_dir).exists()
    assert not output_dir.exists()


def test_pipeline_records_duplicate_count(
    tmp_path: Path,
    duplicated_wine_df: pd.DataFrame,
) -> None:
    source = _write_frame(duplicated_wine_df, tmp_path / "wine.csv")
    output_dir = tmp_path / "output"
    result = run_pipeline(source, output_dir)
    metrics = json.loads(_metrics_path(output_dir).read_text())
    assert metrics["duplicates_removed"] == 1
    assert metrics["rows_modeled"] == metrics["rows_after_dedup"]
    assert result["rows_modeled"] == result["rows_after_dedup"]
    assert metrics["rows_after_dedup"] >= 10


def test_pipeline_empty_csv(tmp_path: Path) -> None:
    source = tmp_path / "empty.csv"
    pd.DataFrame(columns=list(REQUIRED_COLUMNS)).to_csv(source, index=False)
    output_dir = tmp_path / "output"
    with pytest.raises(ValueError):
        run_pipeline(source, output_dir)
    assert not _metrics_path(output_dir).exists()
    assert not output_dir.exists()


def test_pipeline_dedup_below_minimum_writes_nothing(
    tmp_path: Path,
    sample_wine_df: pd.DataFrame,
) -> None:
    unique = sample_wine_df.iloc[:9]
    repeated = pd.concat([unique.iloc[[0]]] * 3, ignore_index=True)
    frame = pd.concat([unique, repeated], ignore_index=True)
    assert len(frame) >= 10
    source = _write_frame(frame, tmp_path / "wine.csv")
    output_dir = tmp_path / "output"
    with pytest.raises(ValueError) as exc_info:
        run_pipeline(source, output_dir)
    message = str(exc_info.value)
    assert "10" in message
    assert "9" in message
    assert not _metrics_path(output_dir).exists()
    assert not output_dir.exists()


def test_pipeline_small_dataset_writes_nothing(
    tmp_path: Path,
    sample_wine_df: pd.DataFrame,
) -> None:
    source = _write_frame(sample_wine_df.iloc[:9], tmp_path / "wine.csv")
    output_dir = tmp_path / "output"
    with pytest.raises(ValueError) as exc_info:
        run_pipeline(source, output_dir)
    message = str(exc_info.value)
    assert "10" in message
    assert "9" in message
    assert not _metrics_path(output_dir).exists()
    assert not output_dir.exists()


def test_pipeline_writes_all_artifacts(tmp_path: Path, sample_wine_df: pd.DataFrame) -> None:
    source = _write_frame(sample_wine_df, tmp_path / "wine.csv")
    output_dir = tmp_path / "nested" / "output"
    assert not output_dir.exists()

    result = run_pipeline(source, output_dir)
    artifact_names = [
        "summary.csv",
        "missing_values.csv",
        "outliers_report.csv",
        "metrics.json",
        "alcohol_vs_quality.png",
    ]
    for name in artifact_names:
        artifact = output_dir / name
        assert artifact.is_file()
        assert artifact.stat().st_size > 0
    assert sorted(path.name for path in output_dir.iterdir()) == sorted(artifact_names)

    summary = pd.read_csv(output_dir / "summary.csv")
    assert list(summary.columns) == [
        "type",
        "count",
        "average_quality",
        "median_quality",
        "average_alcohol",
        "high_quality_count",
    ]
    outliers = pd.read_csv(output_dir / "outliers_report.csv")
    assert len(outliers) == len(NUMERIC_COLUMNS)
    assert list(outliers["column"]) == list(NUMERIC_COLUMNS)

    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert list(metrics.keys()) == list(METRIC_KEYS)
    assert metrics["rows_modeled"] == metrics["rows_after_dedup"]
    assert metrics["rows_modeled"] == result["rows_modeled"]
    assert metrics["missing_values_total"] == 0


def test_cli_success_prints_summary(
    tmp_path: Path,
    sample_wine_df: pd.DataFrame,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _write_frame(sample_wine_df, tmp_path / "wine.csv")
    output_dir = tmp_path / "output"
    code = main(["--input", str(source), "--output-dir", str(output_dir)])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    assert f"rows_read={len(sample_wine_df)}" in captured.out
    assert "duplicates_removed=0" in captured.out
    assert "mae=" in captured.out
    assert "r2=" in captured.out
    assert f"output_dir={output_dir}" in captured.out


def test_cli_missing_file_prints_stderr(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "output"
    missing = tmp_path / "missing.csv"
    code = main(["--input", str(missing), "--output-dir", str(output_dir)])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert str(missing) in captured.err
    assert not _metrics_path(output_dir).exists()


@pytest.mark.parametrize("test_size", ["0", "1", "-0.1", "1.5"])
def test_cli_rejects_bad_test_size(
    tmp_path: Path,
    sample_wine_df: pd.DataFrame,
    capsys: pytest.CaptureFixture[str],
    test_size: str,
) -> None:
    source = _write_frame(sample_wine_df, tmp_path / "wine.csv")
    output_dir = tmp_path / "output"
    code = main(
        [
            "--input",
            str(source),
            "--output-dir",
            str(output_dir),
            "--test-size",
            test_size,
        ]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "test_size" in captured.err
    assert not _metrics_path(output_dir).exists()
    assert not output_dir.exists()


@pytest.mark.parametrize("multiplier", ["0", "-1.5"])
def test_cli_rejects_non_positive_iqr_multiplier(
    tmp_path: Path,
    sample_wine_df: pd.DataFrame,
    capsys: pytest.CaptureFixture[str],
    multiplier: str,
) -> None:
    source = _write_frame(sample_wine_df, tmp_path / "wine.csv")
    output_dir = tmp_path / "output"
    code = main(
        [
            "--input",
            str(source),
            "--output-dir",
            str(output_dir),
            "--iqr-multiplier",
            multiplier,
        ]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "iqr_multiplier" in captured.err
    assert not _metrics_path(output_dir).exists()
    assert not output_dir.exists()


def test_module_and_script_help() -> None:
    module = subprocess.run(
        [sys.executable, "-m", "wine_pipeline", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert module.returncode == 0
    assert "--input" in module.stdout
    assert "--output-dir" in module.stdout

    script = Path(sys.executable).parent / "wine-pipeline"
    direct = subprocess.run(
        [str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert direct.returncode == 0
    assert "--input" in direct.stdout
    assert "--iqr-multiplier" in direct.stdout
