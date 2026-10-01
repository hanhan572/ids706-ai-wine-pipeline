"""Tests for the alcohol versus quality scatter plot."""

from pathlib import Path

import pandas as pd
import pytest
from matplotlib.figure import Figure

from wine_pipeline.visualize import create_scatter_plot


def test_scatter_plot_creates_file(
    tmp_path: Path,
    sample_wine_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    original_savefig = Figure.savefig

    def spy(self: Figure, *args: object, **kwargs: object) -> object:
        axis = self.axes[0]
        captured["xlabel"] = axis.get_xlabel()
        captured["ylabel"] = axis.get_ylabel()
        captured["title"] = axis.get_title()
        captured["n_points"] = len(axis.collections[0].get_offsets())
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", spy)
    output = tmp_path / "figures" / "alcohol_vs_quality.png"
    result = create_scatter_plot(sample_wine_df, output)

    assert result == output
    assert output.is_file()
    assert output.stat().st_size > 0
    assert captured["xlabel"] == "Alcohol Content"
    assert captured["ylabel"] == "Wine Quality"
    assert captured["title"] == "Alcohol Content vs Wine Quality"
    assert captured["n_points"] > 1
