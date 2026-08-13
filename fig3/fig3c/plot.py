#!/usr/bin/env python3
"""Reproduce Fig. 3c from the bundled scEU-seq kinetic-rate table."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


GENES = (
    ("Cooperative", "PTX3"),
    ("Neutral", "CPEB2"),
    ("Destabilizing", "H2AFZ"),
)
KAPPA_COLOR = "#EC6E66"
GAMMA_COLOR = "#92A5D1"


def ordered_rate_columns(table: pd.DataFrame, prefix: str) -> list[str]:
    columns = [column for column in table.columns if column.startswith(prefix)]
    return sorted(columns, key=lambda column: int(re.search(r"\d+", column).group()))


def draw_gene(axis: plt.Axes, table: pd.DataFrame, category: str, gene: str, show_xlabel: bool) -> None:
    row = table.loc[table["gene_symbol"].eq(gene)]
    if len(row) != 1:
        raise ValueError(f"Expected one row for {gene}, found {len(row)}")
    row = row.iloc[0]
    kappa_columns = ordered_rate_columns(table, "norm_kappa_")
    gamma_columns = ordered_rate_columns(table, "norm_gamma_")
    time = np.array([int(re.search(r"\d+", column).group()) for column in kappa_columns])
    display_time = np.linspace(time.min(), time.max(), 20)
    kappa = interp1d(time, row[kappa_columns].to_numpy(dtype=float), kind="cubic")(display_time)
    gamma = interp1d(time, row[gamma_columns].to_numpy(dtype=float), kind="cubic")(display_time)
    cosine = float(row["cosine_similarity"])

    axis.plot(display_time, kappa, color=KAPPA_COLOR, linewidth=2.7, label="Kappa (Synthesis Rate)")
    axis.plot(display_time, gamma, color=GAMMA_COLOR, linewidth=2.7, label="Gamma (Degradation Rate)")
    axis.set_title(f"{category}: {gene} (cosine = {cosine:.2f})", fontsize=11)
    axis.set_ylabel("Normalized Rate", fontsize=9)
    if show_xlabel:
        axis.set_xlabel("Process / Time Point", fontsize=9)
    else:
        axis.tick_params(labelbottom=False)
    axis.tick_params(labelsize=8)
    axis.legend(frameon=True, fontsize=7, loc="upper right")
    axis.spines[["top", "right"]].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser()
    panel_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--input",
        type=Path,
        default=panel_dir / "inputs" / "aax3072_table-s1.csv",
        help="Published scEU-seq kinetic-rate table.",
    )
    parser.add_argument("--output", type=Path, default=panel_dir / "outputs" / "fig3c.png")
    args = parser.parse_args()

    table = pd.read_csv(args.input, skiprows=1)

    figure = plt.figure(figsize=(12, 7), dpi=300)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.05, 1.65), wspace=0.35, hspace=0.38)
    figure.add_subplot(grid[0, 0]).set_axis_off()
    draw_gene(figure.add_subplot(grid[0, 1]), table, "Neutral", "CPEB2", show_xlabel=False)
    draw_gene(figure.add_subplot(grid[1, 0]), table, "Cooperative", "PTX3", show_xlabel=True)
    draw_gene(figure.add_subplot(grid[1, 1]), table, "Destabilizing", "H2AFZ", show_xlabel=True)
    figure.subplots_adjust(left=0.04, right=0.98, bottom=0.08, top=0.95)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=600, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
