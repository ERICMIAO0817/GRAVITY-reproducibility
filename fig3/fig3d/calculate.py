#!/usr/bin/env python3
"""Compute Fig. 3d rate-cosine labels from archived scEU-seq model outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


LABELS = ("Cooperative", "Neutral", "Destabilizing")


def as_dense(value) -> np.ndarray:
    return value.toarray() if hasattr(value, "toarray") else np.asarray(value)


def classify(values: np.ndarray) -> np.ndarray:
    return np.select(
        [values > 0.5, values < -0.5],
        ["Cooperative", "Destabilizing"],
        default="Neutral",
    )


def zscore_rows(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=1, keepdims=True)
    scale = values.std(axis=1, ddof=1, keepdims=True)
    return np.divide(centered, scale, out=np.zeros_like(centered), where=scale > 0)


def process_mean(values: np.ndarray, process: np.ndarray) -> np.ndarray:
    ordered = np.unique(process)
    return np.vstack([values[process == point].mean(axis=0) for point in ordered]).T


def rate_cosine(adata: ad.AnnData, cells: pd.Index, genes: list[str], process: np.ndarray) -> np.ndarray:
    subset = adata[cells, genes]
    alpha = process_mean(as_dense(subset.layers["alpha"]), process)
    gamma = process_mean(as_dense(subset.layers["gamma"]), process)
    alpha_z = zscore_rows(alpha)
    gamma_z = zscore_rows(gamma)
    return np.divide(
        np.einsum("ij,ij->i", alpha_z, gamma_z),
        np.linalg.norm(alpha_z, axis=1) * np.linalg.norm(gamma_z, axis=1),
        out=np.zeros(alpha_z.shape[0]),
        where=(np.linalg.norm(alpha_z, axis=1) * np.linalg.norm(gamma_z, axis=1)) > 0,
    )


def match_reference_range(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Use the legacy linear range match before threshold-based label calls."""
    lower, upper = values.min(), values.max()
    if upper <= lower:
        return np.full_like(values, reference.mean())
    return (values - lower) / (upper - lower) * (reference.max() - reference.min()) + reference.min()


def top_dynamic_genes(adata: ad.AnnData, cells: pd.Index, genes: list[str], process: np.ndarray) -> list[str]:
    expression = as_dense(adata[cells, genes].X)
    scores = np.array([abs(spearmanr(expression[:, index], process).statistic) for index in range(expression.shape[1])])
    selected = np.argsort(scores)[::-1][:100]
    return [genes[index] for index in selected]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    source = args.work_root / "fig3_work" / "inputs"
    table = pd.read_csv(source / "sceu" / "aax3072_table-s1.csv", skiprows=1)
    table = table.dropna(subset=["gene_symbol", "cosine_similarity"]).drop_duplicates("gene_symbol").set_index("gene_symbol")
    gravity = ad.read_h5ad(source / "sceu" / "cdr_sceu_stage2.h5ad")
    celldancer = ad.read_h5ad(source / "sceu" / "cdr_sceu_celldancer.h5ad")
    reference = ad.read_h5ad(source / "rpe1" / "rpe1_cc.h5ad", backed="r")
    scvelo = ad.read_h5ad(source / "rpe1" / "rpe1_velocity_scvelo.h5ad", backed="r")

    cells = gravity.obs_names.intersection(celldancer.obs_names).intersection(reference.obs_names)
    process = pd.to_numeric(reference.obs.loc[cells, "Cell_cycle_relativePos"], errors="coerce").to_numpy()
    valid = np.isfinite(process)
    cells = cells[valid]
    process = process[valid]
    common_genes = sorted(set(gravity.var_names) & set(celldancer.var_names) & set(scvelo.var_names) & set(table.index))
    genes = top_dynamic_genes(celldancer, cells, common_genes, process)

    truth_cosine = table.loc[genes, "cosine_similarity"].to_numpy(dtype=float)
    truth_label = classify(truth_cosine)
    records = []
    metrics = []
    for method, adata in (("GRAVITY", gravity), ("CellDancer", celldancer)):
        raw_cosine = rate_cosine(adata, cells, genes, process)
        normalized_cosine = match_reference_range(raw_cosine, truth_cosine)
        predicted_label = classify(normalized_cosine)
        records.append(
            pd.DataFrame(
                {
                    "method": method,
                    "gene": genes,
                    "truth_cosine": truth_cosine,
                    "predicted_cosine": raw_cosine,
                    "predicted_cosine_range_matched": normalized_cosine,
                    "truth_label": truth_label,
                    "predicted_label": predicted_label,
                }
            )
        )
        metrics.append(
            {
                "method": method,
                "label_agreement_count": int((truth_label == predicted_label).sum()),
                "cosine_spearman": float(spearmanr(truth_cosine, raw_cosine).statistic),
            }
        )

    scores = pd.concat(records, ignore_index=True)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_dir / "kinetic_cosine_scores.csv", index=False)
    pd.DataFrame(metrics).to_csv(output_dir / "fig3d_summary.csv", index=False)
    for method in ("GRAVITY", "CellDancer"):
        subset = scores.loc[scores["method"].eq(method)]
        confusion = pd.crosstab(subset["truth_label"], subset["predicted_label"]).reindex(index=LABELS, columns=LABELS, fill_value=0)
        confusion.to_csv(output_dir / f"{method.lower()}_label_confusion.csv")
    (output_dir / "calculation_manifest.json").write_text(
        json.dumps({"n_cells": int(len(cells)), "n_genes": int(len(genes)), "gene_selection": "top 100 absolute expression-process Spearman scores in CellDancer"}, indent=2)
    )


if __name__ == "__main__":
    main()
