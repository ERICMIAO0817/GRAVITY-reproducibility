"""Render Fig. 2g: BATC robustness to observation noise.

The input table contains all ten BATC replicates at each noise rate.  Error
bars intentionally use the population standard deviation, matching the final
legacy plotting script on the 41702 server.  To rederive the table from Stage-2
outputs, run ``compute_batc.py`` and pass its CSV with ``--input``.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SERIES = [
    ("Linear", "#92a5d1"),
    ("Bifurcating", "#EC6E66"),
    ("Trifurcating", "#fbbf45"),
    ("Quadrifurcating", "#B5CE4E"),
]


def read_scores(path: Path) -> dict[str, dict[float, list[float]]]:
    scores: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            scores[row["topology"]][float(row["noise_rate"])].append(float(row["batc"]))
    return scores


def plot(scores: dict[str, dict[float, list[float]]], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(16, 6), dpi=600)

    for topology, color in SERIES:
        noise_rates = sorted(scores[topology])
        means = np.array([np.mean(scores[topology][rate]) for rate in noise_rates])
        standard_deviations = np.array([np.std(scores[topology][rate]) for rate in noise_rates])
        axis.errorbar(
            noise_rates,
            means,
            yerr=standard_deviations,
            marker="o",
            markersize=12,
            linewidth=2.5,
            capsize=6,
            elinewidth=2,
            color=color,
            linestyle="-",
            label=topology,
        )

    axis.set_xlabel("Noise rate", fontsize=30)
    axis.set_ylabel("BATC", fontsize=30)
    axis.set_title("BATC vs Noise Rate", fontsize=34)
    axis.tick_params(axis="both", labelsize=30)
    axis.set_ylim(0, 1.05)
    axis.set_yticks(np.arange(0, 1.01, 0.2))
    axis.grid(axis="y", linestyle="--", alpha=0.6)
    for spine in axis.spines.values():
        spine.set_linewidth(2)

    axis.legend(
        fontsize=18,
        ncol=len(SERIES),
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        frameon=False,
        borderaxespad=0.3,
        handlelength=1.2,
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
    parser.add_argument(
        "--input", type=Path, default=folder / "inputs" / "batc_noise_replicates.csv"
    )
    parser.add_argument("--output", type=Path, default=folder / "outputs" / "fig2g.png")
    arguments = parser.parse_args()
    plot(read_scores(arguments.input), arguments.output)


if __name__ == "__main__":
    main()
