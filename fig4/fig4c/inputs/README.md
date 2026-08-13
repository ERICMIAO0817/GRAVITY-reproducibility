The published Fig. 4c Jaccard values are stored in `jaccard_similarity.csv`.

The original eight cell-type mean-attention matrices and the derived top-30
gene sets were not retained in the local historical archive. Consequently the
published similarity table is the tracked plotting input. `calculate.py`
recreates the table from the recovered historical `*_mean_attention.npz` files
and matching `genes.txt` when those inputs are available; it does not fit a
model.
