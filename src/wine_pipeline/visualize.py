"""Alcohol versus quality scatter plot."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wine_pipeline.schema import TARGET_COLUMN


def create_scatter_plot(df, output_path: str | Path) -> Path:
    """Save a scatter plot of alcohol versus quality and close the figure."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots()
    axis.scatter(df["alcohol"], df[TARGET_COLUMN])
    axis.set_xlabel("Alcohol Content")
    axis.set_ylabel("Wine Quality")
    axis.set_title("Alcohol Content vs Wine Quality")
    figure.savefig(destination)
    plt.close(figure)
    return destination
