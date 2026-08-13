"""Render Fig. 2f: BATC performance across simulated trajectories.

The input table is transcribed from the final legacy plotting script on the
41702 server.  Each value is the BATC used for the corresponding bar.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = [
    "GRAVITY",
    "CellDancer",
    "scVelo",
    "TFvelo",
    "RegVelo",
    "GRAVITY_naive",
]
DATASETS = ["Linear", "Bifurcating", "Trifurcating", "Quadrifurcating", "Average"]
COLORS = ["#7c9895", "#92a5d1", "#d9b9d4", "#EC6E66", "#fbbf45", "#B5CE4E"]


def read_scores(path: Path) -> dict[tuple[str, str], float]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return {(row["dataset"], row["method"]): float(row["batc"]) for row in reader}


def plot(scores: dict[tuple[str, str], float], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(16, 6), dpi=600)
    positions = np.arange(len(DATASETS))
    width = 0.8 / len(METHODS)

    for index, (method, color) in enumerate(zip(METHODS, COLORS)):
        offset = (index - (len(METHODS) - 1) / 2) * width
        values = [scores[(dataset, method)] for dataset in DATASETS]
        axis.bar(positions + offset, values, width=width, color=color, label=method)

    axis.set_xticks(positions, DATASETS, fontsize=30)
    axis.set_ylabel("BATC", fontsize=30)
    axis.set_title("Performance on Simulation Datasets", fontsize=34)
    axis.tick_params(axis="y", labelsize=30)
    # Fix the legacy display range and major ticks across Matplotlib versions.
    axis.set_ylim(-0.73, 0.84)
    axis.set_yticks(np.arange(-0.6, 0.81, 0.2))
    axis.grid(axis="y", linestyle="--", alpha=0.6)
    for spine in axis.spines.values():
        spine.set_linewidth(2)

    axis.legend(
        fontsize=18,
        ncol=len(METHODS),
        loc="lower right",
        bbox_to_anchor=(1.0, 0.0),
        frameon=False,
        borderaxespad=0.3,
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)


def main() -> None:
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=folder / "inputs" / "batc_summary.csv")
    parser.add_argument("--output", type=Path, default=folder / "outputs" / "fig2f.png")
    arguments = parser.parse_args()
    plot(read_scores(arguments.input), arguments.output)


if __name__ == "__main__":
    main()
