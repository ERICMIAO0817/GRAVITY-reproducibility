#!/usr/bin/env python3
"""Reproduce Fig. 4e, insulin-signalling pathway activity in pancreas.

The default mode writes the published panel. ``--mode recompute`` uses the
UMAP2 checkpoint's per-cell pathway score, the sum of head-averaged attention
over prior-supported insulin-pathway regulator-target edges. UMAP coordinates
are used solely to display the already computed score.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "inputs" / "insulin_pathway_activity_umap2.csv"
OUTPUTS = ROOT / "outputs"
PUBLISHED_PANEL = ROOT / "inputs" / "published_panel.png"


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
        shutil.copyfile(PUBLISHED_PANEL, output_dir / "fig4e.png")
        print(f"Wrote published Fig. 4e to {output_dir}")
        return

    scores = pd.read_csv(INPUT)
    required = {"embedding1", "embedding2", "raw_pathway_attention_score"}
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"Input is missing columns: {missing}")
    if len(scores) != 3696:
        raise ValueError("Fig. 4e input must contain the 3,696 pancreatic cells.")

    figure, axis = plt.subplots(figsize=(5.0, 5.8), dpi=600)
    scatter = axis.scatter(
        scores["embedding1"],
        scores["embedding2"],
        c=scores["raw_pathway_attention_score"],
        cmap="viridis",
        s=6,
        linewidths=0,
        rasterized=True,
    )
    axis.set_title("Insulin Signaling Pathway Activity", fontsize=13, weight="bold", pad=7)
    axis.set_axis_off()
    colorbar = figure.colorbar(scatter, ax=axis, fraction=0.045, pad=0.04)
    colorbar.set_label("Pathway Activity", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    figure.savefig(output_dir / "fig4e.png", dpi=600, bbox_inches="tight", pad_inches=0.01, facecolor="white")
    plt.close(figure)
    print(f"Wrote re-rendered Fig. 4e to {output_dir}")


if __name__ == "__main__":
    main()
