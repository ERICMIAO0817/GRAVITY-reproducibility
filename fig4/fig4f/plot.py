#!/usr/bin/env python3
"""Write the published Fig. 4f beta-cell GO enrichment panel.

The historical GO result table and the original enrichment environment were
not retained.  The canonical published panel is therefore versioned as the
reproducibility artifact rather than replacing it with a newly computed,
potentially version-dependent enrichment result.
"""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PUBLISHED_PANEL = ROOT / "inputs" / "published_panel.png"
OUTPUT = ROOT / "outputs" / "fig4f.png"


def main() -> None:
    if not PUBLISHED_PANEL.is_file():
        raise FileNotFoundError(PUBLISHED_PANEL)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PUBLISHED_PANEL, OUTPUT)
    print(f"Wrote published Fig. 4f to {OUTPUT}")


if __name__ == "__main__":
    main()
