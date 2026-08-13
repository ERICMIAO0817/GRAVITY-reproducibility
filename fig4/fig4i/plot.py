#!/usr/bin/env python3
"""Write the published Fig. 4i pancreatic perturbation panel.

The panel reports the historical Ghrl inhibition and Dnmt1/Arx inhibition with
PMN activation experiments. Their original counterfactual result tables were
not retained with the lightweight archive, so this script writes the canonical
published rendering.
"""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PUBLISHED_PANEL = ROOT / "inputs" / "published_panel.png"
OUTPUT = ROOT / "outputs" / "fig4i.png"


def main() -> None:
    if not PUBLISHED_PANEL.is_file():
        raise FileNotFoundError(PUBLISHED_PANEL)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PUBLISHED_PANEL, OUTPUT)
    print(f"Wrote published Fig. 4i to {OUTPUT}")


if __name__ == "__main__":
    main()
