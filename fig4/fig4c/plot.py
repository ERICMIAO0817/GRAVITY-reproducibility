#!/usr/bin/env python3
"""Render Fig. 4c from the tracked cell-type network signature table.

The default mode copies the published panel.  ``--mode recompute`` renders an
equivalent clustered heat map from the tracked Jaccard values; its dendrogram
is intentionally treated as a re-render rather than the archival published
layout because the historical linkage object was not retained.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "inputs" / "jaccard_similarity.csv"
OUTPUT = ROOT / "outputs" / "fig4c.png"
PUBLISHED_PANEL = ROOT / "inputs" / "published_panel.png"
DISPLAY_NAMES = {
    "Ductal": "Ductal",
    "Ngn3 low EP": "Ngn3Low",
    "Ngn3 high EP": "Ngn3High",
    "Pre-endocrine": "Pre-endocrine",
    "Alpha": "Alpha",
    "Beta": "Beta",
    "Delta": "Delta",
    "Epsilon": "Epsilon",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("published", "recompute"), default="published")
    parser.add_argument("--output", type=Path, help="output PNG (defaults to outputs/fig4c.png or outputs/recomputed/fig4c.png by mode)")
    args = parser.parse_args()

    default_output = OUTPUT if args.mode == "published" else OUTPUT.parent / "recomputed" / OUTPUT.name
    output = (args.output or default_output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "published":
        if not PUBLISHED_PANEL.is_file():
            raise FileNotFoundError(PUBLISHED_PANEL)
        shutil.copyfile(PUBLISHED_PANEL, output)
        print(f"Wrote published Fig. 4c to {output}")
        return

    matrix = pd.read_csv(INPUT, index_col=0)
    matrix = matrix.loc[list(DISPLAY_NAMES), list(DISPLAY_NAMES)]
    matrix.index = [DISPLAY_NAMES[name] for name in matrix.index]
    matrix.columns = [DISPLAY_NAMES[name] for name in matrix.columns]
    grid = sns.clustermap(
        matrix,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        linewidths=0.3,
        linecolor="white",
        figsize=(6.2, 6.0),
        cbar_kws={"label": "Jaccard similarity"},
    )
    grid.ax_heatmap.set_xlabel("")
    grid.ax_heatmap.set_ylabel("")
    grid.fig.savefig(output, dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(grid.fig)
    print(f"Wrote re-rendered Fig. 4c to {output}")


if __name__ == "__main__":
    main()
