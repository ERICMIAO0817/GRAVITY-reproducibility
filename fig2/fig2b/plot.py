"""Render Fig. 2b: linear simulated trajectory benchmark.

Example:
    python plot.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _plot_panel import main


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent
    main(
        panel="b",
        topology="linear",
        default_methods=[
            ("GRAVITY", folder / "inputs" / "Pre4temp_da_stage2.csv"),
            ("scVelo", folder / "inputs" / "scvelo.h5ad"),
            ("CellDancer", folder / "inputs" / "celldancer.csv"),
            ("TFvelo", folder / "inputs" / "tfvelo.h5ad"),
            ("RegVelo", folder / "inputs" / "regvelo.h5ad"),
        ],
        default_output=folder / "outputs" / "fig2b.png",
    )
