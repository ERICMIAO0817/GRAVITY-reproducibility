# GRAVITY reproducibility

Lightweight scripts and rendered outputs for reproducing selected figures from
the GRAVITY study.

## Layout

- `fig2/`: simulation benchmark panels and BATC robustness analysis.
- `fig3/`: embryonic brain and cell-cycle case-study panels.
- `fig4/`: mouse pancreatic development, dynamic gene importance and
  perturbation panels.

Each subdirectory contains a `plot.py` script that writes its panel to an
`outputs/` directory. The corresponding `inputs/` directory contains the
tables, cached method results, or image assets needed by that panel. Panels
requiring a calculation step also include a separate `calculate.py` or
`compute_batc.py` script.

`fig2b`--`fig2e` contain the full per-method result files used to render the
simulation velocity fields. `fig2f`, `fig2g`, `fig3c`, and `fig3d` contain the
complete plotted score tables or calculation cache. `fig3a` contains the
method-specific image assets used for composition. `fig3b` contains its
published panel and measured score table, but its two original intermediate
files (`gravity_pseudotime.npz` and `celldancer_pseudotime.npz`) were not
retained in the historical analysis directory; it is therefore supplied as a
rendered archival panel rather than claimed as a standalone rerun.

## Usage

Run a directly reproducible panel from the repository root, for example:

```bash
python fig2/fig2b/plot.py
python fig3/fig3a/plot.py
python fig3/fig3c/plot.py
python fig3/fig3d/calculate.py
python fig3/fig3d/plot.py
python fig4/fig4d/plot.py
python fig4/fig4e/plot.py
python fig4/fig4g/plot.py
```

PNG figure assets are versioned directly in Git.

## Repository preparation

Codex assisted with organizing the reproducibility directory and its figure
rendering scripts. All analyses, source data and scientific results originate
from the GRAVITY study.
