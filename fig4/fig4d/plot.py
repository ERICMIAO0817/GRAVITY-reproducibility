#!/usr/bin/env python3
"""Reproduce Fig. 4d, dynamic gene-importance scores in pancreas.

The default mode writes the published panel. ``--mode recompute`` renders the
tracked final-UMAP2 per-cell-type summary table: dot colour is the mean
min-max-normalised gene score in that cell type, and dot area encodes the
percent of cells with a score above 0.10. No model is loaded.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "inputs" / "dynamic_gene_importance_umap2.csv"
OUTPUTS = ROOT / "outputs"
PUBLISHED_PANEL = ROOT / "inputs" / "published_panel.png"
GENES = ("FOXP2", "NEUROG3", "HMGN3", "PYY", "IAPP", "CPE", "GCG", "SLC38A5", "PDX1", "INS2", "SST", "HHEX", "GHRL")
GENE_LABELS = {
    "FOXP2": "Foxp2", "NEUROG3": "Neurog3", "HMGN3": "Hmgn3", "PYY": "Pyy", "IAPP": "Iapp",
    "CPE": "Cpe", "GCG": "Gcg", "SLC38A5": "Slc38a5", "PDX1": "Pdx1", "INS2": "Ins2",
    "SST": "Sst", "HHEX": "Hhex", "GHRL": "Ghrl",
}
CELL_TYPES = ("Ductal", "Ngn3Low", "Ngn3High", "Pre-endocrine", "Alpha", "Beta", "Delta", "Epsilon")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("published", "recompute"), default="published")
    parser.add_argument("--output-dir", type=Path, help="output directory (defaults to outputs/ or outputs/recomputed/ by mode)")
    args = parser.parse_args()

    default_output = OUTPUTS if args.mode == "published" else OUTPUTS / "recomputed"
    output_dir = (args.output_dir or default_output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "published":
        if not PUBLISHED_PANEL.is_file():
            raise FileNotFoundError(PUBLISHED_PANEL)
        shutil.copyfile(PUBLISHED_PANEL, output_dir / "fig4d.png")
        print(f"Wrote published Fig. 4d to {output_dir}")
        return

    summary = pd.read_csv(INPUT)
    required = {"clusterid", "gene", "mean_score", "fraction", "x", "y"}
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"Input is missing columns: {missing}")
    if len(summary) != len(GENES) * len(CELL_TYPES):
        raise ValueError("Input must contain the complete cell-type by gene grid.")

    figure = plt.figure(figsize=(8.8, 4.9), dpi=600)
    axis = figure.add_axes((0.08, 0.15, 0.57, 0.74))
    scatter = axis.scatter(
        summary["x"],
        summary["y"],
        s=8.0 + summary["fraction"] * 4.2,
        c=summary["mean_score"],
        cmap="GnBu",
        vmin=0.0,
        vmax=1.0,
        edgecolor="#5E6A6A",
        linewidth=0.45,
    )
    axis.set_title("Dynamic Gene Importance Score", fontsize=14, weight="bold", pad=5)
    axis.set_xticks(range(len(GENES)))
    axis.set_xticklabels([GENE_LABELS[gene] for gene in GENES], rotation=90, fontsize=9)
    axis.set_yticks(range(len(CELL_TYPES)))
    axis.set_yticklabels(CELL_TYPES, fontsize=9)
    axis.set_xlim(-0.8, len(GENES) - 0.2)
    axis.set_ylim(-0.8, len(CELL_TYPES) - 0.2)
    axis.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)

    legend_axis = figure.add_axes((0.69, 0.35, 0.28, 0.48))
    legend_axis.set_axis_off()
    levels = (20, 40, 60, 80, 100)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color="#6D6D6D", markersize=(8.0 + level * 4.2) ** 0.5)
        for level in levels
    ]
    legend_axis.legend(
        handles,
        [str(level) for level in levels],
        title="Fraction of cells\nin group (%)",
        loc="upper left",
        frameon=False,
        fontsize=9,
        title_fontsize=10,
        labelspacing=0.85,
        handletextpad=0.6,
    )
    color_axis = figure.add_axes((0.83, 0.19, 0.07, 0.25))
    colorbar = figure.colorbar(ScalarMappable(norm=Normalize(0.0, 1.0), cmap="GnBu"), cax=color_axis)
    colorbar.set_label("Gene importance\nin group", fontsize=10)
    colorbar.ax.tick_params(labelsize=8)
    figure.savefig(output_dir / "fig4d.png", dpi=600, bbox_inches="tight", pad_inches=0.01, facecolor="white")
    plt.close(figure)
    print(f"Wrote re-rendered Fig. 4d to {output_dir}")


if __name__ == "__main__":
    main()
