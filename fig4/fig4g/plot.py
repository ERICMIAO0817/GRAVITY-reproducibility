#!/usr/bin/env python3
"""Reproduce Fig. 4g, the UMAP2 Pdx1 beta-cell regulatory module.

The default mode writes the published panel. ``--mode recompute`` uses
``inputs/pdx1_module.csv``, the saved ranking of the 25 strongest Pdx1 target
genes in beta cells. Its evidence fields follow the Figure 4g curation: orange
indicates ChIP-seq support, green beta-cell/developmental literature support,
and grey uncurated evidence. This script only redraws the network.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "inputs" / "pdx1_module.csv"
OUTPUTS = ROOT / "outputs"
PUBLISHED_PANEL = ROOT / "inputs" / "published_panel.png"
CATEGORY_COLORS = {
    "ChIP-seq database support": "#FFAA20",
    "Literature support": "#8EDBAD",
    "No curated support": "#D9D9D9",
}


def _display_gene(gene: str) -> str:
    return gene[:1].upper() + gene[1:].lower()


def _positions(count: int) -> list[tuple[float, float]]:
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2.0 * np.pi, count, endpoint=False)
    return [(2.9 * np.cos(angle), 2.2 * np.sin(angle)) for angle in angles]


def _edge_widths(weights: np.ndarray) -> np.ndarray:
    span = float(weights.max() - weights.min())
    return np.full(len(weights), 1.6) if span == 0 else 0.85 + 2.0 * (weights - weights.min()) / span


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
        shutil.copyfile(PUBLISHED_PANEL, output_dir / "fig4g.png")
        print(f"Wrote published Fig. 4g to {output_dir}")
        return

    table = pd.read_csv(INPUT).sort_values("rank", kind="stable").reset_index(drop=True)
    required = {"rank", "target_gene", "attention_weight", "evidence_category"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Input is missing columns: {missing}")
    if len(table) != 25 or table["rank"].tolist() != list(range(1, 26)):
        raise ValueError("The input must contain the ranked Pdx1 top-25 module.")
    unknown = sorted(set(table["evidence_category"]).difference(CATEGORY_COLORS))
    if unknown:
        raise ValueError(f"Unknown evidence categories: {unknown}")

    figure, axis = plt.subplots(figsize=(8.2, 6.6), dpi=600)
    positions = _positions(len(table))
    widths = _edge_widths(table["attention_weight"].to_numpy(dtype=float))
    for row, (x, y), width in zip(table.itertuples(index=False), positions, widths):
        color = CATEGORY_COLORS[row.evidence_category]
        axis.plot([0.0, x], [0.0, y], color=color, linewidth=float(width), alpha=0.82, zorder=1)
        axis.scatter(x, y, s=520, color=color, edgecolor="white", linewidth=1.0, zorder=3)
        axis.text(x, y, _display_gene(str(row.target_gene)), ha="center", va="center", fontsize=7.4, zorder=4)
    center = FancyBboxPatch(
        (-0.48, -0.25), 0.96, 0.50, boxstyle="round,pad=0.055,rounding_size=0.09",
        facecolor="#C78AD6", edgecolor="white", linewidth=1.2, zorder=5,
    )
    axis.add_patch(center)
    axis.text(0.0, 0.0, "Pdx1", ha="center", va="center", fontsize=11, weight="bold", zorder=6)
    axis.set_xlim(-3.85, 3.85)
    axis.set_ylim(-3.05, 3.05)
    axis.set_aspect("equal")
    axis.set_axis_off()
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color="white", markerfacecolor=color,
               markeredgecolor="white", markersize=10, label=label)
        for category, label, color in (
            ("ChIP-seq database support", "ChIP-seq", CATEGORY_COLORS["ChIP-seq database support"]),
            ("Literature support", "Literature", CATEGORY_COLORS["Literature support"]),
            ("No curated support", "Others", CATEGORY_COLORS["No curated support"]),
        )
    ]
    axis.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.055), fontsize=10)
    figure.savefig(output_dir / "fig4g.png", dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(figure)
    print(f"Wrote re-rendered Fig. 4g to {output_dir}")


if __name__ == "__main__":
    main()
