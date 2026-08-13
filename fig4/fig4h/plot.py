#!/usr/bin/env python3
"""Reproduce Fig. 4h from saved UMAP2 learned embedding features.

The default mode writes the published panel.  The ``recompute`` mode uses the
tracked ``combined_features.npz``: it stores the learned splice/unsplice
combined feature for every pancreatic cell after stage 1.  A display UMAP is
fit with fixed diagnostic display parameters (15 neighbours, min_dist 0.25,
Euclidean metric and random_state 42); this operation does not reload a model
or rerun GRAVITY inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from umap import UMAP


ROOT = Path(__file__).resolve().parent
FEATURES = ROOT / "inputs" / "combined_features.npz"
METADATA = ROOT / "inputs" / "metadata.csv"
OUTPUTS = ROOT / "outputs"
PUBLISHED_PANEL = ROOT / "inputs" / "published_panel.png"
CELL_TYPES = (
    "Ductal",
    "Ngn3 low EP",
    "Ngn3 high EP",
    "Pre-endocrine",
    "Alpha",
    "Beta",
    "Delta",
    "Epsilon",
)
PALETTE = {
    "Ductal": "#7c9895",
    "Ngn3 low EP": "#92a5d1",
    "Ngn3 high EP": "#d9b9d4",
    "Pre-endocrine": "#EC6E66",
    "Alpha": "#fbbf45",
    "Beta": "#B5CE4E",
    "Delta": "#bd7795",
    "Epsilon": "#DAA87C",
}
TARGET_PAIRS = (("Alpha", "Epsilon", "#d62728"), ("Beta", "Delta", "#1f77b4"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("published", "recompute"),
        default="published",
        help="write the canonical published panel (default) or redraw the retained learned embedding",
    )
    parser.add_argument("--output-dir", type=Path, help="output directory (defaults to outputs/ or outputs/recomputed/ by mode)")
    args = parser.parse_args()

    default_output = OUTPUTS if args.mode == "published" else OUTPUTS / "recomputed"
    output_dir = (args.output_dir or default_output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "published":
        if not PUBLISHED_PANEL.is_file():
            raise FileNotFoundError(PUBLISHED_PANEL)
        shutil.copyfile(PUBLISHED_PANEL, output_dir / "fig4h.png")
        print(f"Wrote published Fig. 4h to {output_dir}")
        return

    stored = np.load(FEATURES)
    if "combined" not in stored:
        raise ValueError("combined_features.npz must contain the 'combined' learned feature.")
    features = np.asarray(stored["combined"], dtype=np.float32)
    metadata = pd.read_csv(METADATA)
    if "cell_type" not in metadata:
        raise ValueError("metadata.csv must contain cell_type.")
    labels = metadata["cell_type"].astype(str).to_numpy()
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError("Learned feature and metadata dimensions are inconsistent.")
    absent = [cell_type for cell_type in CELL_TYPES if not np.any(labels == cell_type)]
    if absent:
        raise ValueError(f"Missing pancreatic cell types: {absent}")

    coordinates = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.25,
        metric="euclidean",
        random_state=42,
    ).fit_transform(features)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"embedding1": coordinates[:, 0], "embedding2": coordinates[:, 1], "cell_type": labels}).to_csv(
        output_dir / "fig4h_display_coordinates.csv", index=False
    )

    figure, axis = plt.subplots(figsize=(7.0, 5.4), dpi=600)
    centroids: dict[str, np.ndarray] = {}
    for cell_type in CELL_TYPES:
        subset = coordinates[labels == cell_type]
        centroids[cell_type] = subset.mean(axis=0)
        axis.scatter(subset[:, 0], subset[:, 1], s=7, alpha=0.78, color=PALETTE[cell_type], edgecolor="none", rasterized=True)
    for first, second, color in TARGET_PAIRS:
        source, target = centroids[first], centroids[second]
        axis.plot([source[0], target[0]], [source[1], target[1]], linestyle="--", linewidth=1.8, color=color, zorder=4)
    for cell_type in CELL_TYPES:
        center = centroids[cell_type]
        axis.text(center[0], center[1], f" {cell_type}", fontsize=7.6, weight="bold", zorder=5)
    axis.set_aspect("auto")
    axis.set_axis_off()
    handles = [
        Line2D([0], [0], color=color, linewidth=1.8, linestyle="--", label=f"{first} - {second}")
        for first, second, color in TARGET_PAIRS
    ]
    axis.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06), fontsize=9)
    figure.savefig(output_dir / "fig4h.png", dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(figure)
    print(f"Wrote re-analysis Fig. 4h to {output_dir}")


if __name__ == "__main__":
    main()
