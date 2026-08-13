#!/usr/bin/env python3
"""Export the archived Fig. 3a forebrain benchmark panels and a compact layout.

The repository intentionally stores only the rendered benchmark panels, not the
large AnnData objects used by the five external methods.  The original source
locations and drawing definitions are recorded below for traceability:

* cellular panels: 41702:/home/sda1/miaozy/cellDancer-main/src/celldancer
* GRAVITY genes: newnewnewqiannaogene.png
* CellDancer genes: gene_bench_0stage2.png
* scVelo genes: scvelo_dyn.h5ad plus fb1000celldancer.csv positions;
  archived local ``d_s``/``d_u`` arrows from the original run
* TFvelo genes: TFvelo_all.h5ad, ``TFvelo.pl.velocity(..., layers=[])``
* RegVelo genes: 112qnqnqnRV_genes.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
OUTPUTS = ROOT / "outputs"

PALETTE = [
    "#7c9895",
    "#92a5d1",
    "#d9b9d4",
    "#EC6E66",
    "#fbbf45",
    "#B5CE4E",
    "#bd7795",
]
CELL_TYPES = [
    "Radial Glia 1",
    "Radial Glia 2",
    "Neuroblast 1",
    "Neuroblast 2",
    "Immature Neuron 1",
    "Immature Neuron 2",
    "Neuron",
]
PANELS = [
    ("GRAVITY", "gravity", "0.737"),
    ("cellDancer", "celldancer", "-0.138"),
    ("scVelo", "scvelo", "0.047"),
    ("TFvelo", "tfvelo", "0.706"),
    ("RegVelo", "regvelo", "0.251"),
]
SCVELO_GENES = ("CNTNAP2", "GNAO1")


def trim_white_margin(image: np.ndarray, right_limit: float | None = None) -> np.ndarray:
    """Crop export margins while preserving the plotted data extent."""
    if right_limit is not None:
        image = image[:, : int(image.shape[1] * right_limit)]

    rgb = image[..., :3]
    content = np.any(rgb < 0.985, axis=-1)
    rows = np.flatnonzero(content.any(axis=1))
    cols = np.flatnonzero(content.any(axis=0))
    if not len(rows) or not len(cols):
        return image

    pad_y = max(4, int((rows[-1] - rows[0] + 1) * 0.03))
    pad_x = max(4, int((cols[-1] - cols[0] + 1) * 0.03))
    y0, y1 = max(0, rows[0] - pad_y), min(image.shape[0], rows[-1] + pad_y + 1)
    x0, x1 = max(0, cols[0] - pad_x), min(image.shape[1], cols[-1] + pad_x + 1)
    return image[y0:y1, x0:x1]


def load_panel(kind: str, method: str) -> np.ndarray:
    image = plt.imread(INPUTS / f"{kind}_{method}.png")
    # The archived cellular plots have a numeric legend far to the right; the
    # manuscript uses the shared labelled legend instead.
    return trim_white_margin(image, right_limit=0.88 if kind == "cell_velocity" else None)


def _as_dense(values: object) -> np.ndarray:
    """Return a one-dimensional dense vector from an AnnData layer slice."""
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values, dtype=float).reshape(-1)


def sample_phase_neighbors(
    phase: np.ndarray,
    step: tuple[int, int] = (15, 15),
    percentile: float = 15,
) -> np.ndarray:
    """Select representative dense phase-space positions as in CellDancer plots."""
    from scipy.spatial import cKDTree

    minima = phase.min(axis=0)
    maxima = phase.max(axis=0)
    padding = (maxima - minima) * 0.025
    minima -= padding
    maxima += padding
    x_grid, y_grid = np.meshgrid(
        np.linspace(minima[0], maxima[0], step[0]),
        np.linspace(minima[1], maxima[1], step[1]),
    )
    grid = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    grid += np.random.default_rng(10).normal(
        scale=np.maximum(maxima - minima, np.finfo(float).eps) * 0.15,
        size=grid.shape,
    )

    tree = cKDTree(phase)
    selected = np.unique(tree.query(grid, k=1)[1])
    distances, _ = tree.query(phase[selected], k=min(20, len(phase)))
    density = np.exp(-(distances**2) / (2 * 0.5**2)).sum(axis=1)
    return selected[density > np.percentile(density, percentile)]


def render_scvelo_gene_panel(adata_path: Path, positions_path: Path) -> None:
    """Recompute scVelo phase portraits with per-cell kinetic arrows.

    Positions use the normalised splice/unsplice values from the archived
    CellDancer input, matching the original plotting script. Arrow components
    are the archived scVelo dynamical ``d_s`` and ``d_u`` estimates at the
    same cells. Missing values are zeroed, matching the historic script.
    """
    import pandas as pd
    import scanpy as sc

    adata = sc.read_h5ad(adata_path)

    positions = pd.read_csv(positions_path)
    required_columns = {"cellIndex", "gene_name", "unsplice", "splice", "clusters"}
    missing_columns = required_columns.difference(positions.columns)
    if missing_columns:
        raise ValueError(f"scVelo position table is missing columns: {sorted(missing_columns)}")

    fig, axes = plt.subplots(1, len(SCVELO_GENES), figsize=(20, 4), dpi=600)

    for axis, gene in zip(np.ravel(axes), SCVELO_GENES):
        gene_positions = positions.loc[positions["gene_name"] == gene].sort_values("cellIndex")
        cell_indices = gene_positions["cellIndex"].to_numpy(dtype=int)
        if len(gene_positions) != adata.n_obs or not np.array_equal(cell_indices, np.arange(adata.n_obs)):
            raise ValueError(f"{gene} positions do not align one-to-one with AnnData cells")

        gene_index = adata.var_names.get_loc(gene)
        unspliced = gene_positions["unsplice"].to_numpy(dtype=float)
        spliced = gene_positions["splice"].to_numpy(dtype=float)
        velocity_s = np.nan_to_num(
            _as_dense(adata.layers["dynamical_velocity"][:, gene_index]),
            nan=0.0,
        )
        velocity_u = np.nan_to_num(
            _as_dense(adata.layers["dynamical_velocity_u"][:, gene_index]),
            nan=0.0,
        )
        colors = np.asarray(PALETTE, dtype=object)[gene_positions["clusters"].to_numpy(dtype=int)]
        sampled = sample_phase_neighbors(np.column_stack((unspliced, spliced)))

        axis.scatter(spliced, unspliced, c=colors, s=40, alpha=0.5, edgecolors="none")
        axis.scatter(
            spliced[sampled],
            unspliced[sampled],
            s=40,
            facecolors="none",
            edgecolors="black",
            linewidths=0.8,
        )
        if np.any(velocity_s[sampled]) or np.any(velocity_u[sampled]):
            axis.quiver(
                spliced[sampled],
                unspliced[sampled],
                velocity_s[sampled],
                velocity_u[sampled],
                angles="xy",
                color="black",
                width=0.002,
            )
        axis.set_title(gene)
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(INPUTS / "gene_dynamics_scvelo.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def export_archived_panels() -> None:
    for _, method, _ in PANELS:
        for kind in ("cell_velocity", "gene_dynamics"):
            copyfile(INPUTS / f"{kind}_{method}.png", OUTPUTS / f"{kind}_{method}.png")


def draw_panel(ax_cell: plt.Axes, ax_gene: plt.Axes, name: str, method: str, batc: str) -> None:
    ax_cell.imshow(load_panel("cell_velocity", method))
    ax_cell.set_title(f"{name}\nBATC = {batc}", fontsize=10, fontweight="bold", pad=2)
    ax_cell.axis("off")

    ax_gene.imshow(load_panel("gene_dynamics", method))
    ax_gene.axis("off")


def build_composite() -> None:
    fig = plt.figure(figsize=(13.5, 9.0), dpi=600)
    grid = fig.add_gridspec(2, 3, hspace=0.14, wspace=0.08)
    fig.text(0.01, 0.995, "a", ha="left", va="top", fontsize=22, fontweight="bold")

    for index, (name, method, batc) in enumerate(PANELS):
        row, column = divmod(index, 3)
        subgrid = grid[row, column].subgridspec(2, 1, height_ratios=(2.0, 1.15), hspace=0.01)
        draw_panel(fig.add_subplot(subgrid[0]), fig.add_subplot(subgrid[1]), name, method, batc)

    legend_axis = fig.add_subplot(grid[1, 2])
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="none", label=label, markersize=8)
        for label, color in zip(CELL_TYPES, PALETTE)
    ]
    legend_axis.legend(handles=handles, loc="center", frameon=False, fontsize=11, labelspacing=1.25, handletextpad=0.5)
    legend_axis.axis("off")

    fig.savefig(OUTPUTS / "fig3a.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--render-scvelo-genes",
        type=Path,
        metavar="H5AD",
        help="render the scVelo CNTNAP2/GNAO1 panel from the archived velocity AnnData file",
    )
    parser.add_argument(
        "--scvelo-positions",
        type=Path,
        metavar="CSV",
        help="normalised CellDancer splice/unsplice values used as scVelo phase coordinates",
    )
    args = parser.parse_args()

    OUTPUTS.mkdir(exist_ok=True)
    if args.render_scvelo_genes:
        if args.scvelo_positions is None:
            parser.error("--render-scvelo-genes requires --scvelo-positions")
        render_scvelo_gene_panel(args.render_scvelo_genes, args.scvelo_positions)
    export_archived_panels()
    build_composite()
    print(f"Wrote Fig. 3a panels to {OUTPUTS}")


if __name__ == "__main__":
    main()
