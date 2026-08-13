"""Shared plotting implementation for the four GRAVITY Fig. 2 trajectory panels.

Each panel wrapper supplies only its letter and topology.  The input files are
existing per-method GRAVITY-style stage CSVs (or one-row-per-cell CSVs that
already contain ``velocity1`` and ``velocity2``).  If a stage CSV has not yet
been projected, this module computes the 2-D velocity with GRAVITY's own
``compute_cell_velocity_`` utility; it never imports CellDancer.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np
import pandas as pd

# The reproducibility repository is intentionally separate from the GRAVITY
# source repository. Prefer an installed package, but allow a sibling checkout
# or GRAVITY_SOURCE for direct server use without copying code.
for _candidate in (
    os.environ.get("GRAVITY_SOURCE"),
    str(Path(__file__).resolve().parents[2]),
    str(Path(__file__).resolve().parents[2] / "GRAVITY"),
):
    if _candidate and (Path(_candidate) / "gravity").is_dir():
        sys.path.insert(0, _candidate)
        break

from gravity.plotting.velocity import _grid_curve_arrows
from gravity.velocity import compute_cell_velocity_


VECTOR_CANDIDATES = (
    ("velocity1", "velocity2"),
    ("velocity_umap1", "velocity_umap2"),
    ("velocity_x", "velocity_y"),
)
PAPER_PALETTE = {
    "0": "#7c9895",
    "1": "#92a5d1",
    "2": "#d9b9d4",
    "3": "#EC6E66",
    "4": "#fbbf45",
    "5": "#B5CE4E",
    "6": "#bd7795",
}


def _parse_method(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--method must use NAME=PATH.")
    name, raw_path = value.split("=", maxsplit=1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--method must use a non-empty NAME=PATH.")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Method input does not exist: {path}")
    return name.strip(), path


def _drop_export_index(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame.columns) and str(frame.columns[0]).startswith("Unnamed:"):
        return frame.iloc[:, 1:].copy()
    return frame


def _vector_columns(frame: pd.DataFrame) -> tuple[str, str] | None:
    for columns in VECTOR_CANDIDATES:
        if set(columns).issubset(frame.columns):
            return columns
    return None


def _load_h5ad_velocity(path: Path) -> pd.DataFrame:
    """Load a baseline H5AD with a 2-D UMAP velocity field.

    The published baselines store their plotted coordinates and vectors in
    ``obsm['X_umap']`` and ``obsm['velocity_umap']``.  Reading these cached
    results avoids rerunning any baseline method in the reproduction scripts.
    """
    try:
        from anndata import read_h5ad
    except ImportError as error:  # pragma: no cover - environment specific
        raise RuntimeError("Reading baseline H5AD files requires anndata.") from error

    adata = read_h5ad(path)
    if "X_umap" not in adata.obsm or "velocity_umap" not in adata.obsm:
        raise ValueError(
            f"Baseline H5AD must contain obsm['X_umap'] and obsm['velocity_umap']: {path}"
        )
    embedding = np.asarray(adata.obsm["X_umap"])
    velocity = np.asarray(adata.obsm["velocity_umap"])
    if embedding.ndim != 2 or velocity.ndim != 2 or embedding.shape[1] < 2 or velocity.shape[1] < 2:
        raise ValueError(f"Baseline H5AD needs at least two embedding and velocity dimensions: {path}")
    if embedding.shape[0] != velocity.shape[0]:
        raise ValueError(f"Baseline H5AD has unmatched embedding and velocity cell counts: {path}")

    cluster_column = next((name for name in ("clusters", "cell_type") if name in adata.obs), None)
    clusters = adata.obs[cluster_column].astype(str).to_numpy() if cluster_column else np.repeat("cells", adata.n_obs)
    return pd.DataFrame(
        {
            "cellID": adata.obs_names.astype(str),
            "embedding1": embedding[:, 0],
            "embedding2": embedding[:, 1],
            "velocity1": velocity[:, 0],
            "velocity2": velocity[:, 1],
            "clusters": clusters,
        }
    )


def _load_method_table(path: Path) -> pd.DataFrame:
    """Load either a GRAVITY-style CSV result or a cached baseline H5AD."""
    if path.suffix.lower() == ".h5ad":
        return _load_h5ad_velocity(path)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Unsupported method result type: {path}")
    return _drop_export_index(pd.read_csv(path))


def _project_if_needed(frame: pd.DataFrame, *, projection_step: int | None) -> pd.DataFrame:
    """Return a stage frame containing GRAVITY-projected 2-D cell velocities."""
    columns = _vector_columns(frame)
    if columns is not None and frame.loc[:, list(columns)].notna().any(axis=None):
        if columns != ("velocity1", "velocity2"):
            frame = frame.rename(columns={columns[0]: "velocity1", columns[1]: "velocity2"})
        return frame

    required = {"cellID", "cellIndex", "gene_name", "alpha", "beta", "splice", "unsplice", "embedding1", "embedding2"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "Input has no projected vectors and cannot be projected by GRAVITY. "
            f"Missing columns: {missing}"
        )
    projected, _ = compute_cell_velocity_(
        frame,
        # The published simulation panels projected every cell.  The earlier
        # 20-by-20 preview downsampling changes the field substantially.
        speed_up=None if projection_step is None else (projection_step, projection_step),
        expression_scale="power10",
        projection_neighbor_size=200,
        projection_neighbor_choice="embedding",
    )
    return projected


def _per_cell_frame(frame: pd.DataFrame, *, fallback_clusters: pd.Series | None = None) -> pd.DataFrame:
    required = {"embedding1", "embedding2", "velocity1", "velocity2"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Velocity table is missing required columns: {missing}")
    identity = "cellID" if "cellID" in frame.columns else "cellIndex"
    if identity not in frame.columns:
        raise ValueError("Velocity table needs either cellID or cellIndex.")

    columns = [identity, "embedding1", "embedding2", "velocity1", "velocity2"]
    if "clusters" in frame.columns:
        columns.append("clusters")
    # groupby.first keeps the first finite velocity after a stage output was
    # expanded from one cell to one cell-gene row.
    cells = frame.loc[:, columns].groupby(identity, sort=False, as_index=False).first()
    cells = cells.dropna(subset=["embedding1", "embedding2", "velocity1", "velocity2"])
    if "clusters" not in cells.columns:
        if fallback_clusters is None:
            cells["clusters"] = "cells"
        else:
            cells["clusters"] = cells[identity].map(fallback_clusters).fillna("cells")
    cells["clusters"] = cells["clusters"].astype(str)
    return cells


def _palette(categories: Iterable[str]) -> dict[str, str]:
    labels = list(dict.fromkeys(map(str, categories)))
    cmap = plt.get_cmap("tab20", max(len(labels), 1))
    return {
        label: PAPER_PALETTE.get(label, to_hex(cmap(index)))
        for index, label in enumerate(labels)
    }


def _draw_velocity(
    axis: plt.Axes,
    cells: pd.DataFrame,
    palette: dict[str, str],
    *,
    arrow_grid: tuple[int, int],
    min_mass: float,
    title: str,
) -> None:
    for cluster, group in cells.groupby("clusters", sort=False):
        axis.scatter(
            group["embedding1"],
            group["embedding2"],
            s=150,
            alpha=0.5,
            color=palette.get(str(cluster), "#999999"),
            edgecolor="none",
            rasterized=True,
        )
    _grid_curve_arrows(
        axis,
        cells.loc[:, ["embedding1", "embedding2"]].to_numpy(dtype=float),
        cells.loc[:, ["velocity1", "velocity2"]].to_numpy(dtype=float),
        arrow_grid=arrow_grid,
        min_mass=min_mass,
    )
    axis.set_axis_off()
    axis.set_title(title, fontsize=12, weight="bold", pad=5)


def _overlay_truth(axis: plt.Axes, truth: pd.DataFrame, *, arrow_grid: tuple[int, int], min_mass: float) -> None:
    """Overlay red truth arrows without modifying GRAVITY's black field renderer."""
    values = truth.loc[:, ["embedding1", "embedding2", "velocity1", "velocity2"]].to_numpy(dtype=float)
    target_count = max(1, arrow_grid[0] * arrow_grid[1])
    if len(values) > target_count:
        take = np.linspace(0, len(values) - 1, target_count, dtype=int)
        values = values[take]
    axis.quiver(
        values[:, 0],
        values[:, 1],
        values[:, 2],
        values[:, 3],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="#D62728",
        width=0.0035,
        alpha=0.9,
    )


def _parse_grid(value: str) -> tuple[int, int]:
    try:
        first, second = (int(part.strip()) for part in value.split(",", maxsplit=1))
    except ValueError as error:
        raise argparse.ArgumentTypeError("--arrow-grid must be N,N.") from error
    if first < 2 or second < 2:
        raise argparse.ArgumentTypeError("--arrow-grid values must be at least 2.")
    return first, second


def main(
    *,
    panel: str,
    topology: str,
    default_methods: list[tuple[str, Path]] | None = None,
    default_output: Path | None = None,
) -> None:
    parser = argparse.ArgumentParser(
        description=f"Render Fig. 2{panel}: the {topology} simulated trajectory benchmark."
    )
    parser.add_argument(
        "--method",
        action="append",
        type=_parse_method,
        required=default_methods is None,
        metavar="NAME=PATH",
        help="Method display name and its stage/result CSV. Repeat in paper panel order.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        help="Optional CSV with embedding1, embedding2, velocity1 and velocity2. Red arrows are overlaid on GRAVITY.",
    )
    parser.add_argument(
        "--ground-truth-panel",
        default="GRAVITY",
        help="Display name of the method receiving the red ground-truth overlay (default: GRAVITY).",
    )
    parser.add_argument("--output", type=Path, default=default_output or Path(f"fig2{panel}.png"))
    parser.add_argument("--arrow-grid", type=_parse_grid, default=(20, 20))
    parser.add_argument(
        "--projection-step",
        type=int,
        default=None,
        help="Optional embedding-grid downsampling step. Omit to reproduce the published full-cell projection.",
    )
    parser.add_argument("--min-mass", type=float, default=0.5)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    if args.projection_step is not None and args.projection_step < 2:
        parser.error("--projection-step must be at least 2.")
    if not 0.0 < args.min_mass <= 1.0:
        parser.error("--min-mass must be in (0, 1].")

    method_specs = args.method
    if method_specs is None:
        assert default_methods is not None
        method_specs = [(name, path.resolve()) for name, path in default_methods]

    methods: list[tuple[str, pd.DataFrame]] = []
    for name, path in method_specs:
        stage = _load_method_table(path)
        projected = _project_if_needed(stage, projection_step=args.projection_step)
        methods.append((name, _per_cell_frame(projected)))

    all_clusters = [cluster for _, cells in methods for cluster in cells["clusters"]]
    palette = _palette(all_clusters)
    fallback_clusters = methods[0][1].set_index("cellID")["clusters"] if "cellID" in methods[0][1] else None

    truth: pd.DataFrame | None = None
    if args.ground_truth is not None:
        truth_path = args.ground_truth.expanduser().resolve()
        if not truth_path.is_file():
            parser.error(f"--ground-truth does not exist: {truth_path}")
        truth = _per_cell_frame(_drop_export_index(pd.read_csv(truth_path)), fallback_clusters=fallback_clusters)

    # Five near-square panels reproduce the paper's horizontal method layout.
    figure, axes = plt.subplots(1, len(methods), figsize=(3 * len(methods), 3), squeeze=False)
    for axis, (name, cells) in zip(axes[0], methods):
        _draw_velocity(
            axis,
            cells,
            palette,
            arrow_grid=args.arrow_grid,
            min_mass=args.min_mass,
            title=name,
        )
        if truth is not None and name == args.ground_truth_panel:
            _overlay_truth(axis, truth, arrow_grid=args.arrow_grid, min_mass=args.min_mass)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main(panel="?", topology="simulated")
