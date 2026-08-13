#!/usr/bin/env python3
"""Regenerate the Fig. 4c signature and Jaccard tables from saved attention.

The original analysis wrote an average stage-1 attention matrix for each
pancreatic cell type.  This script consumes those eight historical sparse NPZ
matrices and the matching ordered ``genes.txt`` file.  It preserves the
original calculation: retain the top 1% attention entries, rank genes by the
column-wise retained mean, select the top 30 genes, then calculate pairwise
Jaccard similarity of these signatures.

The historical mean-attention files were not retained in the local archive,
so the tracked ``jaccard_similarity.csv`` is the source table transcribed from
the published panel.  When those NPZ files are recovered, this script can
recreate both tracked tables without model fitting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import load_npz


CELL_TYPES = (
    "Ductal",
    "Ngn3 low EP",
    "Ngn3 high EP",
    "Pre-endocrine",
    "Epsilon",
    "Delta",
    "Alpha",
    "Beta",
)
TOP_FRACTION = 0.01
TOP_GENES = 30


def _signature(mean_attention: np.ndarray, genes: list[str]) -> list[str]:
    matrix = torch.as_tensor(mean_attention, dtype=torch.float32)
    threshold = torch.quantile(matrix, 1.0 - TOP_FRACTION)
    retained = matrix * (matrix >= threshold)
    order = torch.argsort(retained.mean(dim=0), descending=True).numpy()
    return [genes[index] for index in order[:TOP_GENES]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mean-attention-dir", type=Path, required=True)
    parser.add_argument("--genes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "inputs")
    args = parser.parse_args()

    genes = [line.strip().upper() for line in args.genes.read_text().splitlines() if line.strip()]
    signatures: dict[str, set[str]] = {}
    rows = []
    for cell_type in CELL_TYPES:
        safe_name = cell_type.replace(" ", "_")
        matrix_path = args.mean_attention_dir / f"{safe_name}_mean_attention.npz"
        if not matrix_path.is_file():
            raise FileNotFoundError(matrix_path)
        matrix = load_npz(matrix_path).toarray()
        if matrix.shape != (len(genes), len(genes)):
            raise ValueError(f"{matrix_path.name}: expected {len(genes)} square matrix, found {matrix.shape}.")
        ranked = _signature(matrix, genes)
        signatures[cell_type] = set(ranked)
        rows.extend({"cell_type": cell_type, "rank": rank, "gene": gene} for rank, gene in enumerate(ranked, start=1))

    similarity = np.eye(len(CELL_TYPES), dtype=float)
    for row, first in enumerate(CELL_TYPES):
        for column, second in enumerate(CELL_TYPES):
            if row != column:
                similarity[row, column] = len(signatures[first] & signatures[second]) / len(signatures[first] | signatures[second])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "cell_type_top30_genes.csv", index=False)
    pd.DataFrame(similarity, index=CELL_TYPES, columns=CELL_TYPES).to_csv(args.output_dir / "jaccard_similarity.csv")


if __name__ == "__main__":
    main()
