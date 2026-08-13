#!/usr/bin/env python3
"""Reproduce Fig. 4b, pancreatic gene-phase portraits.

The default ``published`` mode writes the rendered manuscript panel.  The
``recompute`` mode draws the four historical columns: GRAVITY stage 1,
GRAVITY after gene-wise training (stage 2), scVelo dynamical and CellDancer.
Each tracked input contains only BICC1, RBFOX3 and RFX6 for the 3,696
pancreatic cells.  All arrow components are saved method-specific rates; this
script does no model fitting or velocity inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
OUTPUTS = ROOT / "outputs"
PUBLISHED_PANEL = INPUTS / "published_panel.png"
GENES = ("BICC1", "RBFOX3", "RFX6")
GENE_LABELS = {"BICC1": "Bicc1", "RBFOX3": "Rbfox3", "RFX6": "Rfx6"}
METHODS = (
    ("GRAVITY stage 1", "gravity_stage1_three_genes.csv"),
    ("GRAVITY", "gravity_stage2_three_genes.csv"),
    ("scVelo", "scvelo_dynamical_three_genes.csv"),
    ("CellDancer", "celldancer_three_genes.csv"),
)
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


def _sample_phase_space(phase: np.ndarray, *, grid: tuple[int, int] = (15, 15), percentile: float = 5.0) -> np.ndarray:
    """Replicate the historic grid-neighbour sampling used for Fig. 4b."""

    low = phase.min(axis=0)
    high = phase.max(axis=0)
    padding = (high - low) * 0.025
    x_grid, y_grid = np.meshgrid(
        np.linspace(low[0] - padding[0], high[0] + padding[0], grid[0]),
        np.linspace(low[1] - padding[1], high[1] + padding[1], grid[1]),
    )
    anchors = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    anchors += np.random.default_rng(10).normal(scale=0.15, size=anchors.shape)
    tree = cKDTree(phase)
    selected = np.unique(tree.query(anchors, k=1)[1])
    if len(selected) < 2:
        return selected
    distances, _ = tree.query(phase[selected], k=min(20, len(phase)))
    density = np.exp(-(distances**2) / (2.0 * 0.5**2)).sum(axis=1)
    return selected[density > np.percentile(density, percentile)]


def _load(path: Path) -> dict[str, pd.DataFrame]:
    frame = pd.read_csv(path)
    if len(frame.columns) and str(frame.columns[0]).startswith("Unnamed:"):
        frame = frame.iloc[:, 1:].copy()
    required = {
        "cellID",
        "cellIndex",
        "gene_name",
        "unsplice",
        "splice",
        "unsplice_velocity",
        "splice_velocity",
        "clusters",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    frame["gene_key"] = frame["gene_name"].astype(str).str.upper()
    result: dict[str, pd.DataFrame] = {}
    for gene in GENES:
        rows = frame.loc[frame["gene_key"].eq(gene)].drop(columns="gene_key").copy()
        rows = rows.sort_values("cellIndex", kind="stable").reset_index(drop=True)
        if len(rows) != 3696 or rows["cellIndex"].duplicated().any():
            raise ValueError(f"{path.name}: {gene} must contain one row for each of 3,696 cells.")
        result[gene] = rows
    return result


def _draw(axis: plt.Axes, rows: pd.DataFrame, *, gene: str) -> None:
    for cell_type in CELL_TYPES:
        mask = rows["clusters"].astype(str).eq(cell_type)
        if mask.any():
            axis.scatter(
                rows.loc[mask, "splice"],
                rows.loc[mask, "unsplice"],
                s=12,
                alpha=0.5,
                color=PALETTE[cell_type],
                edgecolor="none",
                rasterized=True,
                zorder=1,
            )

    values = rows[["unsplice", "splice", "unsplice_velocity", "splice_velocity"]].to_numpy(dtype=float)
    valid = np.isfinite(values).all(axis=1)
    values = values[valid]
    # Sampling operates in (unspliced, spliced); the rendered x/y axes are
    # (spliced, unspliced), exactly as the original CellDancer helper did.
    selected = _sample_phase_space(values[:, :2])
    sampled = values[selected]
    arrow = sampled[:, 2:4][:, ::-1]
    magnitude = np.linalg.norm(arrow, axis=1)
    nonzero = magnitude > 0
    arrow[nonzero] /= magnitude[nonzero, None]
    arrow[~nonzero] = 0.0
    axis.scatter(sampled[:, 1], sampled[:, 0], s=12, facecolors="none", edgecolors="black", linewidths=0.45, zorder=3)
    axis.quiver(
        sampled[:, 1],
        sampled[:, 0],
        arrow[:, 0],
        arrow[:, 1],
        angles="xy",
        color="black",
        width=0.0025,
        scale=12.0,
        zorder=4,
    )
    axis.set_title(GENE_LABELS[gene], fontsize=9, weight="bold", pad=1.5)
    axis.set_axis_off()


def render(data: dict[str, dict[str, pd.DataFrame]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(len(GENES), len(METHODS), figsize=(9.4, 7.6), dpi=600)
    for column, (method, _) in enumerate(METHODS):
        axes[0, column].text(0.5, 1.11, method, transform=axes[0, column].transAxes, ha="center", va="bottom", fontsize=10, weight="bold")
    for row, gene in enumerate(GENES):
        for column, (method, _) in enumerate(METHODS):
            _draw(axes[row, column], data[method][gene], gene=gene)
    figure.subplots_adjust(left=0.015, right=0.995, bottom=0.01, top=0.95, hspace=0.12, wspace=0.08)
    figure.savefig(output / "fig4b.png", dpi=600, bbox_inches="tight", pad_inches=0.01, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("published", "recompute"),
        default="published",
        help="write the canonical published panel (default) or redraw from cached phase portraits",
    )
    parser.add_argument("--output-dir", type=Path, help="output directory (defaults to outputs/ or outputs/recomputed/ by mode)")
    args = parser.parse_args()

    default_output = OUTPUTS if args.mode == "published" else OUTPUTS / "recomputed"
    output_dir = (args.output_dir or default_output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "published":
        if not PUBLISHED_PANEL.is_file():
            raise FileNotFoundError(PUBLISHED_PANEL)
        shutil.copyfile(PUBLISHED_PANEL, output_dir / "fig4b.png")
        print(f"Wrote published Fig. 4b to {output_dir}")
        return

    data = {method: _load(INPUTS / filename) for method, filename in METHODS}
    render(data, output_dir)
    print(f"Wrote re-analysis Fig. 4b to {output_dir}")


if __name__ == "__main__":
    main()
