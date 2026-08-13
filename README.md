# GRAVITY reproducibility

Lightweight scripts and rendered outputs for reproducing selected figures from
the GRAVITY study.

## Layout

- `fig2/`: simulation benchmark panels and BATC robustness analysis.
- `fig3/`: embryonic brain and cell-cycle case-study panels.

Each subdirectory contains a `plot.py` script that writes its panel to an
`outputs/` directory. Panels requiring a calculation step also include a
separate `calculate.py` or `compute_batc.py` script. Large raw datasets are not
stored in this repository; scripts that need them expose an explicit input or
work-root argument.

## Usage

Run a panel from its directory, for example:

```bash
python fig2/fig2b/plot.py
python fig3/fig3a/plot.py
```

PNG figure assets are versioned with Git LFS. Install and initialize Git LFS
before cloning or contributing:

```bash
git lfs install
```
