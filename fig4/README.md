# Figure 4: mouse pancreatic development

Run all panels from the repository root:

```bash
python fig4/fig4a/plot.py
python fig4/fig4b/plot.py
python fig4/fig4c/plot.py
python fig4/fig4d/plot.py
python fig4/fig4e/plot.py
python fig4/fig4f/plot.py
python fig4/fig4g/plot.py
python fig4/fig4h/plot.py
python fig4/fig4i/plot.py
```

Every default script writes the canonical published panel kept in its
`inputs/` directory, so the recorded output exactly matches the manuscript.
Panels `d`, `e` and `g` additionally retain their tracked pancreatic result
tables. Panel `c` retains the published Jaccard table and a calculation script
that regenerates it when the original mean-attention matrices are provided.
The archival panels are crops of the original Figure 4 manuscript page; they
preserve the published composition rather than reformatting individual panels.

Panels `a`, `b`, `c`, `d`, `e`, `g` and `h` retain documented re-analysis
renderers. Invoke them explicitly with `--mode recompute`; they are never
substituted for the published figures. `fig4b --mode recompute` additionally
requires `inputs/scvelo_dynamical_three_genes.csv`, which can be regenerated
from the original dynamical scVelo object when it is recovered.

The Fig. 4h re-analysis UMAP is a diagnostic projection from saved learned
features; its exact historical display coordinates are not retained, so it is
not claimed to be a pixel-identical reconstruction of the published panel.

Unless an explicit output location is supplied, recomputed plots are written
under `outputs/recomputed/`, leaving the canonical published output unchanged.
