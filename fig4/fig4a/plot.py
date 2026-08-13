#!/usr/bin/env python3
"""Reproduce Fig. 4a, the pancreatic endocrinogenesis velocity field.

The default ``published`` mode writes the rendered panel cropped from the
published manuscript.  This is the canonical reproduction because the
historical plotting arrangement is not fully retained.  ``recompute`` keeps a
documented re-analysis route using a compact one-row-per-cell cache containing
UMAP coordinates, labels and GRAVITY-projected velocities.  To rebuild that
cache from a historical stage-2 result, pass ``--stage2-csv`` once.  The source
CSV is intentionally not versioned because it is a 1.47 GB long-format table.

Projection uses every cell (``speed_up=None``), with the published settings:
2-D UMAP neighbours, power-10 expression scaling and 200 neighbours.  The
20-by-20 grid is used only to render the smoothed arrow field.
"""

from __future__ import annotations

import argparse
from math import comb
import os
from pathlib import Path
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import norm as normal
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
OUTPUTS = ROOT / "outputs"
CACHE = INPUTS / "cell_velocity.csv"
PUBLISHED_PANEL = INPUTS / "published_panel.png"

# Keep the original pancreatic palette and its display order.
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
LEGEND_LABELS = {
    "Ductal": "Ductal",
    "Ngn3 low EP": "Ngn3Low",
    "Ngn3 high EP": "Ngn3High",
    "Pre-endocrine": "Pre-endocrine",
    "Alpha": "Alpha",
    "Beta": "Beta",
    "Delta": "Delta",
    "Epsilon": "Epsilon",
}
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
ARROW_GRID = (20, 20)


def _load_gravity() -> None:
    """Locate a GRAVITY checkout when the package is not installed."""

    for candidate in (
        os.environ.get("GRAVITY_SOURCE"),
        str(ROOT.parents[3] / "GRAVITY"),
        str(ROOT.parents[2]),
    ):
        if candidate and (Path(candidate) / "gravity").is_dir():
            sys.path.insert(0, candidate)
            return


def _drop_export_index(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame.columns) and str(frame.columns[0]).startswith("Unnamed:"):
        return frame.iloc[:, 1:].copy()
    return frame


def build_cache(stage2_csv: Path, cache: Path) -> pd.DataFrame:
    """Project fixed stage-2 gene velocities and save the plotted cell table."""

    _load_gravity()
    from gravity.velocity import compute_cell_velocity_

    stage = _drop_export_index(pd.read_csv(stage2_csv))
    required = {
        "cellID",
        "cellIndex",
        "gene_name",
        "alpha",
        "beta",
        "splice",
        "unsplice",
        "splice_predict",
        "embedding1",
        "embedding2",
        "clusters",
    }
    missing = sorted(required.difference(stage.columns))
    if missing:
        raise ValueError(f"Stage-2 table is missing required columns: {missing}")

    # This is the published cell-velocity projection, not a retraining step.
    projected, _ = compute_cell_velocity_(
        stage,
        speed_up=None,
        expression_scale="power10",
        projection_neighbor_size=200,
        projection_neighbor_choice="embedding",
    )
    cells = (
        projected.loc[
            :, ["cellID", "cellIndex", "embedding1", "embedding2", "clusters", "velocity1", "velocity2"]
        ]
        .groupby("cellID", sort=False, as_index=False)
        .first()
        .sort_values("cellIndex", kind="stable")
        .reset_index(drop=True)
    )
    if cells["cellID"].duplicated().any() or len(cells) != stage["cellID"].nunique():
        raise RuntimeError("Projected table does not contain exactly one row per cell.")
    if cells[["embedding1", "embedding2", "velocity1", "velocity2"]].isna().any().any():
        raise RuntimeError("Projected velocity cache contains missing coordinates or vectors.")

    cache.parent.mkdir(parents=True, exist_ok=True)
    cells.to_csv(cache, index=False)
    return cells


def load_cells(cache: Path, stage2_csv: Path | None) -> pd.DataFrame:
    if cache.is_file():
        cells = pd.read_csv(cache)
    elif stage2_csv is not None:
        cells = build_cache(stage2_csv, cache)
    else:
        raise FileNotFoundError(
            f"Missing compact input {cache}. Rebuild it once with --stage2-csv HISTORICAL_STAGE2.csv."
        )
    required = {"cellID", "cellIndex", "embedding1", "embedding2", "clusters", "velocity1", "velocity2"}
    missing = sorted(required.difference(cells.columns))
    if missing:
        raise ValueError(f"Velocity cache is missing required columns: {missing}")
    return cells


def _draw_points(axis: plt.Axes, cells: pd.DataFrame, *, alpha: float = 0.5, size: float = 10.0) -> None:
    for cell_type in CELL_TYPES:
        group = cells.loc[cells["clusters"].astype(str).eq(cell_type)]
        if len(group):
            axis.scatter(
                group["embedding1"],
                group["embedding2"],
                s=size,
                alpha=alpha,
                color=PALETTE[cell_type],
                edgecolor="none",
                rasterized=True,
                zorder=1,
            )


def _limits(cells: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    coordinates = cells[["embedding1", "embedding2"]].to_numpy(dtype=float)
    low = coordinates.min(axis=0)
    high = coordinates.max(axis=0)
    padding = np.maximum((high - low) * 0.04, 0.1)
    return (float(low[0] - padding[0]), float(high[0] + padding[0])), (
        float(low[1] - padding[1]),
        float(high[1] + padding[1]),
    )


def _evaluate_bezier(nodes: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    """Evaluate a fourth-degree Bezier curve without an optional dependency."""

    degree = nodes.shape[1] - 1
    basis = np.vstack(
        [
            float(comb(degree, index))
            * (1.0 - parameters) ** (degree - index)
            * parameters**index
            for index in range(degree + 1)
        ]
    )
    return nodes @ basis


def _grid_curve_arrows(axis: plt.Axes, coordinates: np.ndarray, vectors: np.ndarray) -> None:
    """Render the published two-pass Gaussian-smoothed velocity grid.

    This is local plotting code, retained here so that regenerating the tracked
    figure never imports the GRAVITY source tree.
    """

    grids = []
    for dimension in range(2):
        low = float(coordinates[:, dimension].min() - 0.2)
        high = float(coordinates[:, dimension].max() - 0.2)
        low -= 0.025 * abs(high - low)
        high += 0.025 * abs(high - low)
        grids.append(np.linspace(low, high, ARROW_GRID[dimension]))
    mesh = np.meshgrid(*grids)
    grid = np.vstack([item.flat for item in mesh]).T

    neighbours = max(1, int(len(vectors) / 3))
    spacing = float(np.mean([grid_axis[1] - grid_axis[0] for grid_axis in grids]))
    bandwidth = 0.8 * spacing

    def weighted_vectors(data: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        distances, indices = NearestNeighbors(n_neighbors=neighbours).fit(data).kneighbors(query, return_distance=True)
        weights = normal.pdf(distances, loc=0.0, scale=bandwidth)
        mass = weights.sum(axis=1)
        average = (vectors[indices] * weights[:, :, None]).sum(axis=1) / np.maximum(1.0, mass)[:, None]
        return average, mass

    head, head_mass = weighted_vectors(coordinates, grid)
    tail, _ = weighted_vectors(coordinates + vectors, grid)
    # The second KNN pass evaluates the local direction at displaced positions.
    head_second, _ = weighted_vectors(coordinates, grid + head)
    tail_second, _ = weighted_vectors(coordinates, grid - tail)
    keep = head_mass >= 0.5
    grid, head, tail, head_second, tail_second = (
        value[keep] for value in (grid, head, tail, head_second, tail_second)
    )
    if not len(grid):
        return

    parameters = np.linspace(0.0, 1.5, 15)
    maximum_length = 0.0
    for point, tail_vector, head_vector, tail_vector_second, head_vector_second in zip(
        grid, tail, head, tail_second, head_second
    ):
        nodes = np.asfortranarray(
            [
                [
                    point[0] - tail_vector[0] - tail_vector_second[0],
                    point[0] - tail_vector[0],
                    point[0],
                    point[0] + head_vector[0],
                    point[0] + head_vector[0] + head_vector_second[0],
                ],
                [
                    point[1] - tail_vector[1] - tail_vector_second[1],
                    point[1] - tail_vector[1],
                    point[1],
                    point[1] + head_vector[1],
                    point[1] + head_vector[1] + head_vector_second[1],
                ],
            ]
        )
        curve = _evaluate_bezier(nodes, parameters)
        maximum_length = max(
            maximum_length,
            float(np.sqrt(np.diff(curve[0]) ** 2 + np.diff(curve[1]) ** 2).sum()),
        )
    ratio = spacing / maximum_length if maximum_length > 0 else 1.0

    tail *= ratio
    head *= ratio
    tail_second *= ratio
    head_second *= ratio
    for point, tail_vector, head_vector, tail_vector_second, head_vector_second in zip(
        grid, tail, head, tail_second, head_second
    ):
        nodes = np.asfortranarray(
            [
                [
                    point[0] - tail_vector[0] - tail_vector_second[0],
                    point[0] - tail_vector[0],
                    point[0],
                    point[0] + head_vector[0],
                    point[0] + head_vector[0] + head_vector_second[0],
                ],
                [
                    point[1] - tail_vector[1] - tail_vector_second[1],
                    point[1] - tail_vector[1],
                    point[1],
                    point[1] + head_vector[1],
                    point[1] + head_vector[1] + head_vector_second[1],
                ],
            ]
        )
        curve = _evaluate_bezier(nodes, parameters)
        axis.plot(curve[0], curve[1], linewidth=0.5, color="black", alpha=1.0, zorder=2)
        direction = curve[:, -1] - curve[:, -2]
        norm = float(np.linalg.norm(direction)) + 1e-12
        axis.quiver(
            curve[0, -2],
            curve[1, -2],
            direction[0] / norm * 0.5,
            direction[1] / norm * 0.5,
            units="xy",
            angles="xy",
            scale=1,
            color="black",
            width=0.1,
            zorder=3,
        )


def _draw_velocity(axis: plt.Axes, cells: pd.DataFrame) -> None:
    _draw_points(axis, cells)
    vectors = cells[["velocity1", "velocity2"]].to_numpy(dtype=float)
    coordinates = cells[["embedding1", "embedding2"]].to_numpy(dtype=float)
    _grid_curve_arrows(axis, coordinates, vectors)


def _style(axis: plt.Axes, limits: tuple[tuple[float, float], tuple[float, float]]) -> None:
    axis.set_xlim(*limits[0])
    axis.set_ylim(*limits[1])
    # The historical pancreatic panel used matplotlib's automatic axes ratio,
    # which fills its portrait panel without excluding any UMAP coordinates.
    axis.set_aspect("auto")
    axis.set_axis_off()


def render(cells: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    limits = _limits(cells)

    for name, draw in (("fig4a_velocity", _draw_velocity), ("fig4a_cell_types", _draw_points)):
        figure, axis = plt.subplots(figsize=(4.0, 6.0), dpi=600)
        draw(axis, cells)
        _style(axis, limits)
        figure.savefig(output_dir / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.01, facecolor="white")
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(7.6, 6.2), dpi=600, gridspec_kw={"wspace": 0.02})
    _draw_velocity(axes[0], cells)
    _draw_points(axes[1], cells)
    for axis in axes:
        _style(axis, limits)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE[cell_type], markeredgecolor="none",
               label=LEGEND_LABELS[cell_type], markersize=7)
        for cell_type in CELL_TYPES
    ]
    figure.suptitle("Trajectory Inference on Pancreatic Endocrinogenesis", x=0.03, y=0.985, ha="left", fontsize=10, fontweight="bold")
    figure.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.965),
        ncol=4,
        frameon=False,
        fontsize=8,
        handlelength=0.8,
        columnspacing=0.75,
        handletextpad=0.35,
    )
    figure.subplots_adjust(top=0.84, left=0.02, right=0.99, bottom=0.01)
    figure.savefig(output_dir / "fig4a.png", dpi=600, bbox_inches="tight", pad_inches=0.01, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("published", "recompute"),
        default="published",
        help="write the canonical published panel (default) or the retained re-analysis rendering",
    )
    parser.add_argument("--stage2-csv", type=Path, help="historical long-format GRAVITY stage-2 result used to build the compact cache")
    parser.add_argument("--cache", type=Path, default=CACHE, help="tracked one-row-per-cell velocity cache")
    parser.add_argument("--output-dir", type=Path, help="output directory (defaults to outputs/ or outputs/recomputed/ by mode)")
    args = parser.parse_args()

    default_output = OUTPUTS if args.mode == "published" else OUTPUTS / "recomputed"
    output_dir = (args.output_dir or default_output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "published":
        if not PUBLISHED_PANEL.is_file():
            raise FileNotFoundError(PUBLISHED_PANEL)
        shutil.copyfile(PUBLISHED_PANEL, output_dir / "fig4a.png")
        print(f"Wrote published Fig. 4a to {output_dir}")
        return

    stage2_csv = args.stage2_csv.expanduser().resolve() if args.stage2_csv else None
    if stage2_csv is not None and not stage2_csv.is_file():
        raise FileNotFoundError(stage2_csv)
    cells = load_cells(args.cache.expanduser().resolve(), stage2_csv)
    render(cells, output_dir)
    print(f"Wrote re-analysis Fig. 4a to {output_dir}")


if __name__ == "__main__":
    main()
