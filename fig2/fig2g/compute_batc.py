"""Recompute Fig. 2g BATC values from robustness Stage-2 H5AD outputs.

The noise is applied during the robustness training runs, not in this script.
For every noise-rate/seed pair this program loads the corresponding Stage-2
prediction, restores cluster labels from the matching scVelo reference, and
uses GRAVITY's PCHIP BATC implementation on the known simulated lineage graph.

Place the raw references and predictions under ``inputs/raw`` following the
templates in ``inputs/batc_recompute_manifest.json``.  The output CSV has the
same columns as ``inputs/batc_noise_replicates.csv`` and can be rendered with:

    python plot.py --input outputs/batc_noise_recomputed.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import anndata as ad


FOLDER = Path(__file__).resolve().parent
PROJECT_ROOT = FOLDER.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gravity.analysis.batc import compute_batc


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        manifest = json.load(handle)
    required = {"noise_rates", "seeds", "batc_parameters", "topologies"}
    missing = required.difference(manifest)
    if missing:
        raise ValueError(f"Manifest is missing required keys: {sorted(missing)}")
    return manifest


def noise_rate_token(value: float) -> str:
    """Use the historical one-decimal filenames, such as ``1.0`` and ``1.5``."""

    return f"{value:.1f}"


def resolve_input(input_root: Path, relative_path: str) -> Path:
    path = input_root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Required Fig. 2g input is missing: {path}")
    return path


def compute_topology(
    topology: dict[str, Any],
    input_root: Path,
    noise_rates: list[float],
    seeds: list[int],
    parameters: dict[str, int],
) -> list[dict[str, float | int | str]]:
    reference = ad.read_h5ad(resolve_input(input_root, topology["reference"]))
    cluster_key = topology["reference_cluster_key"]
    if cluster_key not in reference.obs:
        raise KeyError(f"{topology['name']}: reference lacks obs[{cluster_key!r}]")
    cluster_labels = reference.obs[cluster_key].astype(str)
    edges = [(str(source), str(target)) for source, target in topology["edges"]]
    rows: list[dict[str, float | int | str]] = []

    for noise_rate in noise_rates:
        for seed in seeds:
            relative_prediction = topology["prediction_template"].format(
                noise_rate=noise_rate_token(noise_rate), seed=seed
            )
            prediction = ad.read_h5ad(resolve_input(input_root, relative_prediction))
            if not prediction.obs_names.equals(reference.obs_names):
                raise ValueError(
                    f"{topology['name']} noise={noise_rate} seed={seed}: "
                    "prediction and reference cell order differ."
                )

            # The historical H5AD files store X_cdr/velocity_cdr but not labels.
            prediction.obs["_batc_cluster"] = cluster_labels.to_numpy()
            score = compute_batc(
                prediction,
                edges,
                cluster_key="_batc_cluster",
                embedding_key="cdr",
                velocity_key="velocity",
                n_bins=parameters["n_bins"],
                min_per_bin=parameters["min_per_bin"],
                n_samples=parameters["n_samples"],
                store_in_adata=False,
            )
            rows.append(
                {
                    "topology": topology["name"],
                    "noise_rate": noise_rate,
                    # Historical source CSV indexes seed 0 through 9 as replicate 1 through 10.
                    "replicate": seed + 1,
                    "batc": score,
                }
            )
            print(f"{topology['name']} noise={noise_rate:.1f} seed={seed}: BATC={score:.12f}")
    return rows


def write_rows(rows: list[dict[str, float | int | str]], output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["topology", "noise_rate", "replicate", "batc"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=FOLDER / "inputs" / "batc_recompute_manifest.json",
    )
    parser.add_argument("--input-root", type=Path, default=FOLDER / "inputs")
    parser.add_argument(
        "--output",
        type=Path,
        default=FOLDER / "outputs" / "batc_noise_recomputed.csv",
    )
    parser.add_argument(
        "--topology",
        action="append",
        help="Run only this topology; repeat the option for multiple topologies.",
    )
    parser.add_argument(
        "--noise-rates",
        type=float,
        nargs="+",
        help="Override the manifest noise rates, for example: --noise-rates 1.0 1.5",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Override the manifest seed list, for example: --seeds 0 1",
    )
    arguments = parser.parse_args()

    manifest = load_manifest(arguments.manifest)
    selected = set(arguments.topology or [])
    topologies = [
        topology for topology in manifest["topologies"] if not selected or topology["name"] in selected
    ]
    unknown = selected.difference({topology["name"] for topology in manifest["topologies"]})
    if unknown:
        raise ValueError(f"Unknown topology requested: {sorted(unknown)}")

    noise_rates = arguments.noise_rates or [float(value) for value in manifest["noise_rates"]]
    seeds = arguments.seeds or [int(value) for value in manifest["seeds"]]
    parameters = {name: int(value) for name, value in manifest["batc_parameters"].items()}

    rows: list[dict[str, float | int | str]] = []
    for topology in topologies:
        rows.extend(compute_topology(topology, arguments.input_root, noise_rates, seeds, parameters))
    write_rows(rows, arguments.output)


if __name__ == "__main__":
    main()
