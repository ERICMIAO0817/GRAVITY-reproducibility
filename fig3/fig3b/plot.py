#!/usr/bin/env python3
"""Reproduce Fig. 3b: RPE1 cell-cycle position and velocity fields.

The raw RPE1 objects are intentionally kept outside this repository.  Point
``--work-root`` at the GRAVITY working directory that contains ``fig3_work``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors


# Values printed in the published Fig. 3b. Measured values are saved separately.
REPORTED_SPEARMAN = {
    "GRAVITY": 0.9991,
    "scVelo": 0.2899,
    "CellDancer": 0.9884,
    "TFvelo": 0.5271,
    "RegVelo": 0.5187,
}


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lower, upper = np.nanmin(values), np.nanmax(values)
    return (values - lower) / (upper - lower) if upper > lower else np.zeros_like(values)


def representative_cells(embedding: np.ndarray, grid_size: int = 28) -> np.ndarray:
    """Choose unique observed cells nearest a regular UMAP grid."""
    x = np.linspace(embedding[:, 0].min(), embedding[:, 0].max(), grid_size)
    y = np.linspace(embedding[:, 1].min(), embedding[:, 1].max(), grid_size)
    mesh_x, mesh_y = np.meshgrid(x, y)
    grid = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
    indices = NearestNeighbors(n_neighbors=1).fit(embedding).kneighbors(grid, return_distance=False)
    return np.unique(indices.ravel())


def plot_panel(
    axis: plt.Axes,
    embedding: np.ndarray,
    pseudotime: np.ndarray,
    title: str,
    velocity: np.ndarray | None = None,
    reported_score: float | None = None,
) -> None:
    axis.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=normalize(pseudotime),
        cmap="plasma",
        s=5,
        linewidths=0,
        rasterized=True,
    )
    if velocity is not None:
        valid = np.isfinite(velocity).all(axis=1)
        indices = representative_cells(embedding[valid])
        coordinates = embedding[valid][indices]
        vectors = velocity[valid][indices]
        magnitude = np.linalg.norm(vectors, axis=1)
        scale = np.nanmedian(magnitude[magnitude > 0])
        if np.isfinite(scale) and scale > 0:
            extent = np.mean(np.ptp(embedding, axis=0))
            directions = np.divide(
                vectors,
                magnitude[:, None],
                out=np.zeros_like(vectors),
                where=magnitude[:, None] > 0,
            )
            lengths = np.clip(magnitude / scale, 0, 2.0) * extent * 0.018
            vectors = directions * lengths[:, None]
            axis.quiver(
                coordinates[:, 0],
                coordinates[:, 1],
                vectors[:, 0],
                vectors[:, 1],
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.0035,
                headwidth=3.5,
                headlength=4.5,
                headaxislength=4.0,
                color="black",
                zorder=3,
            )
    label = title if reported_score is None else f"{title}\nLatent time Spearman = {reported_score:.3f}"
    axis.set_title(label, fontsize=12, fontweight="bold")
    axis.set_axis_off()
    axis.set_aspect("equal")


def metric_against_reference(
    reference_ids: pd.Index, reference_time: np.ndarray, ids: np.ndarray, pseudotime: np.ndarray
) -> float:
    estimated = pd.Series(pseudotime, index=pd.Index(ids.astype(str)))
    truth = pd.Series(reference_time, index=reference_ids.astype(str))
    common = truth.index.intersection(estimated.index)
    return float(spearmanr(truth.loc[common], estimated.loc[common], nan_policy="omit")[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/fig3b.png"))
    args = parser.parse_args()

    source = args.work_root / "fig3_work" / "inputs" / "rpe1"
    result = args.work_root / "fig3_work" / "results" / "rpe1"
    reference = ad.read_h5ad(source / "rpe1_cc.h5ad", backed="r")
    reference_embedding = np.asarray(reference.obsm["X_umap"])
    reference_time = pd.to_numeric(reference.obs["Cell_cycle_relativePos"], errors="coerce").to_numpy()
    reference_ids = reference.obs_names.astype(str)

    gravity = np.load(result / "gravity_pseudotime.npz", allow_pickle=True)
    celldancer = np.load(result / "celldancer_pseudotime.npz", allow_pickle=True)
    scvelo = ad.read_h5ad(source / "rpe1_velocity_scvelo.h5ad", backed="r")
    tfvelo = ad.read_h5ad(source / "TFvelo_all.h5ad", backed="r")
    regvelo = ad.read_h5ad(source / "adata_processed_rpe1_regvelo.h5ad", backed="r")

    panels = [
        ("Relative Cell Cycle Position", reference_embedding, reference_time, None, None),
        ("GRAVITY", gravity["embedding"], gravity["pseudotime"], gravity["velocity"], REPORTED_SPEARMAN["GRAVITY"]),
        ("scVelo", np.asarray(scvelo.obsm["X_umap"]), np.asarray(scvelo.obs["dynamical_velocity_pseudotime"]), np.asarray(scvelo.obsm["dynamical_velocity_umap"]), REPORTED_SPEARMAN["scVelo"]),
        ("CellDancer", celldancer["embedding"], celldancer["pseudotime"], celldancer["velocity"], REPORTED_SPEARMAN["CellDancer"]),
        ("TFvelo", np.asarray(tfvelo.obsm["X_umap"]), np.asarray(tfvelo.obs["velocity_pseudotime"]), np.asarray(tfvelo.obsm["velocity_umap"]), REPORTED_SPEARMAN["TFvelo"]),
        ("RegVelo", np.asarray(regvelo.obsm["X_umap"]), np.asarray(regvelo.layers["fit_t"]).mean(axis=1), np.asarray(regvelo.obsm["velocity_umap"]), REPORTED_SPEARMAN["RegVelo"]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=300)
    for axis, panel in zip(axes.flat, panels):
        title, embedding, pseudotime, velocity, reported_score = panel
        plot_panel(axis, embedding, pseudotime, title, velocity, reported_score)
    fig.tight_layout(pad=0.8, rect=(0, 0, 0.94, 1))
    color_axis = fig.add_axes([0.952, 0.12, 0.012, 0.25])
    fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap="plasma"),
        cax=color_axis,
        ticks=np.arange(0, 1, 0.2),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=600, bbox_inches="tight")
    plt.close(fig)

    metrics = [
        ("GRAVITY", metric_against_reference(reference_ids, reference_time, gravity["cell_id"], gravity["pseudotime"])),
        ("CellDancer", metric_against_reference(reference_ids, reference_time, celldancer["cell_id"], celldancer["pseudotime"])),
        ("scVelo", float(spearmanr(reference_time, np.asarray(scvelo.obs["dynamical_velocity_pseudotime"]), nan_policy="omit")[0])),
        ("TFvelo", float(spearmanr(reference_time, np.asarray(tfvelo.obs["velocity_pseudotime"]), nan_policy="omit")[0])),
        ("RegVelo", float(spearmanr(reference_time, np.asarray(regvelo.layers["fit_t"]).mean(axis=1), nan_policy="omit")[0])),
    ]
    pd.DataFrame(metrics, columns=["method", "measured_spearman"]).to_csv(
        args.output.with_name("fig3b_measured_spearman.csv"), index=False
    )


if __name__ == "__main__":
    main()
