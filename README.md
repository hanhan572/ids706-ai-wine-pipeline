# Wine quality command-line pipeline

This repository is Option 2 for IDS 706: a previous wine-quality analysis rebuilt as a new, modular, reproducible command-line pipeline. The application is one Python process. It reads a merged red and white wine CSV and writes summary, outlier, model, and plot artifacts.

Two design choices are fixed:

- The program runs as one command-line process in one container. It does not use Docker Compose, a database, a web server, or a second container.
- IQR outliers are counted and retained because extreme physicochemical measurements may be legitimate wines. The pipeline does not drop, clip, impute, or down-weight those rows.

## Dataset

The file `data/wine_quality_merged.csv` is the merged red and white wine quality table previously used from the Kaggle dataset [Red and White Wine Quality](https://www.kaggle.com/datasets/amirmohamadrezaie/red-and-white-wine-quality).

The CSV must contain exactly these 13 columns:

- `fixed acidity`
- `volatile acidity`
- `citric acid`
- `residual sugar`
- `chlorides`
- `free sulfur dioxide`
- `total sulfur dioxide`
- `density`
- `pH`
- `sulphates`
- `alcohol`
- `quality`
- `type` (`red` or `white` after stripping whitespace and lowercasing)

A missing column, an unexpected column, a non-numeric feature or `quality` value, an invalid `type`, or any missing required value stops the run. The pipeline does not impute. Exact duplicate rows are removed. Fewer than 10 rows after that removal also stops the run. Nothing is written until validation and modeling succeed.

## Installation

Use Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Command-line usage

`wine-pipeline` and `python -m wine_pipeline` call the same program.

```bash
wine-pipeline \
  --input data/wine_quality_merged.csv \
  --output-dir output \
  --test-size 0.2 \
  --random-state 42 \
  --iqr-multiplier 1.5
```

| Argument | Required | Default | Rule |
| --- | --- | --- | --- |
| `--input` | yes | none | Path to the CSV |
| `--output-dir` | yes | none | Directory to create and write |
| `--test-size` | no | `0.2` | Float in `(0, 1)` |
| `--random-state` | no | `42` | Integer |
| `--iqr-multiplier` | no | `1.5` | Float greater than 0 |

A successful run prints one line to stdout with the rows read, duplicates removed, MAE, R-squared, and output directory, then exits `0`. A missing file or a validation error is printed to stderr and the process exits `1`.

The model uses the 11 physicochemical columns to predict `quality`. `type` is not a model feature. Linear regression treats the integer quality score as continuous.

## Output artifacts

A successful run writes these five files in `--output-dir`:

- `summary.csv`: one row per wine type, with `count`, `average_quality`, `median_quality`, `average_alcohol`, and `high_quality_count` for `quality >= 7`.
- `missing_values.csv`: missing-value count for each required column. A count above zero stops the run before this file is written.
- `outliers_report.csv`: IQR fences and outlier counts for the 11 features and `quality`. Rows outside the fences stay in the summary, the plot, and the model.
- `metrics.json`: row counts, duplicate count, outlier total, split settings, training and test sizes, MAE, and R-squared. `rows_modeled` equals `rows_after_dedup`.
- `alcohol_vs_quality.png`: scatter plot of alcohol versus quality, labeled "Alcohol Content" and "Wine Quality", titled "Alcohol Content vs Wine Quality".

## Testing

From the repository root, after the editable install:

```bash
python -m pytest
```

Tests use synthetic frames under `tmp_path`. They do not read `data/wine_quality_merged.csv` and they do not use the network.

GitHub Actions runs the same pytest suite on pushes to `main` and on pull requests. That workflow does not build the Docker image.

## Docker

The image is one container and one process. Build it from the repository root:

```bash
docker build -t wine-pipeline .
```

The image contains the installed package. It does not contain `data/`, `tests/`, `output/`, or `output-docker/`. Mount the dataset read-only and mount a host directory for the artifacts:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$(pwd)/data:/data:ro" \
  -v "$(pwd)/output-docker:/output" \
  wine-pipeline \
  --input /data/wine_quality_merged.csv \
  --output-dir /output
```

The `--user` flag keeps the written files owned by the host user. `MPLCONFIGDIR=/tmp` lets Matplotlib run when that user has no home directory in the container.

## Reproducibility

The default split is `test_size=0.2` and `random_state=42`. Both values are stored in `metrics.json`. The same settings produce the same MAE, R-squared, and split sizes. Changing `test_size` or `random_state` changes those results. Duplicate removal compares entire rows for exact equality.

## Manual smoke test

**Status: PASS.** These checks were run manually against the current tree. The counts below are sanity checks for `data/wine_quality_merged.csv`, not unit-test oracles.

- `python -m pytest -q` passed: 51 tests.
- `wine-pipeline` and `python -m wine_pipeline` both start.
- `python -m wine_pipeline --input data/wine_quality_merged.csv --output-dir output` completed successfully.
- The run wrote exactly these five files: `summary.csv`, `missing_values.csv`, `outliers_report.csv`, `metrics.json`, and `alcohol_vs_quality.png`.
- `metrics.json` recorded `rows_read` 6497, `duplicates_removed` 1177, `rows_after_dedup` 5320, `rows_modeled` 5320, and `missing_values_total` 0.
- `python -m wine_pipeline --input data/does-not-exist.csv --output-dir output` returned exit code 1.
- `docker build -t wine-pipeline .` succeeded.
- The container ran with the dataset mounted read-only and the output directory mounted from the host.
- All five Docker-generated artifacts appeared on the host, and those files are owned by the host user.

## AI-assisted development reflection

This repository used three separate AI-assisted roles: Architect, Builder, and Tester.

The Architect helped redesign the previous single-script wine analysis as a modular command-line pipeline and produced the implementation plan in `docs/plan.md`.

The Builder implemented the package structure, validation, duplicate handling, outlier reporting, summary analysis, linear regression, tests, Docker configuration, GitHub Actions workflow, and documentation. I independently ran the verification commands after each stage and reported failures back to the Builder.

The Tester independently compared the implementation against `docs/plan.md`. I accepted its finding that type normalization occurred too early and could change exact-duplicate identity. The implementation was corrected so exact duplicate removal occurs before type normalization.

I modified one part of the Tester's recommendation rather than accepting it directly. The Tester initially recommended removing type validation entirely. Because the input contract specifies normalized red and white wine types, I kept type validation but moved it until after duplicate removal.

I also accepted the recommendation to reject non-finite IQR multipliers such as NaN.

Independent verification included the full pytest suite, the full wine dataset, CLI failure behavior, Docker image creation, container execution with mounted input and output directories, and inspection of the generated artifacts. The final independent Tester review found no remaining blocker.