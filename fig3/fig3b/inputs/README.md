# Fig. 3b archival inputs

The published rendered panel and its measured Spearman statistics are retained
in `../outputs/`. The historical plotting code also required two derived files:

- `gravity_pseudotime.npz`
- `celldancer_pseudotime.npz`

They contained the 2-D cell positions, projected velocity vectors, cell IDs,
and pseudotime after the original RPE1 post-processing. Those intermediate
files are not present in the archived analysis locations. The raw RPE1 H5AD
objects alone cannot recover the exact published GRAVITY and CellDancer
pseudotime arrays without rerunning that historical post-processing workflow.

Accordingly, this folder intentionally records the provenance gap instead of
shipping an approximate replacement. The `plot.py` script remains available for
regeneration when the two archived `.npz` files are recovered and supplied via
`--work-root`.
