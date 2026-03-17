#!/usr/bin/env python3
"""Convert clinical methods comparison notebook from 37M to 147M variant."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "bulkformer_dx_clinical_methods_comparison.ipynb"
DST = ROOT / "notebooks" / "bulkformer_dx_clinical_methods_comparison_147M.ipynb"


def replace_in_cell(cell: dict) -> None:
    """Apply 37M→147M replacements in cell source."""
    if cell.get("cell_type") not in ("markdown", "code"):
        return
    source = cell.get("source")
    if not source:
        return
    if isinstance(source, list):
        joined = "".join(source)
    else:
        joined = source

    # Order matters: do more specific replacements first
    replacements = [
        # Paths and identifiers
        ("clinical_methods_37M", "clinical_methods_147M"),
        ('"runs" / "clinical_methods_37M"', '"runs" / "clinical_methods_147M"'),
        ('"figures" / "clinical_methods_comparison"', '"figures" / "clinical_methods_comparison_147M"'),
        # Model
        ("BulkFormer_37M.pt", "BulkFormer_147M.pt"),
        ('--variant "37M"', '--variant "147M"'),
        ('"37M"', '"147M"'),
        # Titles and descriptions
        ("(37M)", "(147M)"),
        ("**37M model**", "**147M model**"),
        ("## 2. Anomaly score (37M)", "## 2. Anomaly score (147M)"),
        ("typical 0.7–0.9 for 37M", "typical 0.7–0.9 for 147M"),
        ("128-dim for 37M", "640-dim for 147M"),
        ("variant 37M", "variant 147M"),
        ("131 dims for 37M", "641 dims for 147M"),
        ("146 samples × 131 dims", "146 samples × 641 dims"),
        ("(146 samples x 131 dims)", "(146 samples x 641 dims)"),
        # Batch size for 147M (larger model, more VRAM)
        ('"--batch-size", "16"', '"--batch-size", "8"'),
        ("--batch-size 16", "--batch-size 8"),
    ]

    for old, new in replacements:
        joined = joined.replace(old, new)

    # Add Linux GPU note to requirements
    if "CUDA on Linux/GPU" in joined and "**Linux GPU**" not in joined:
        joined = joined.replace(
            "- If you get `'numpy.ufunc'",
            "- **Linux GPU**: Set `CUDA_VISIBLE_DEVICES` to select GPUs; batch size 8 for 147M VRAM\n"
            "- If you get `'numpy.ufunc'",
        )

    # Add Linux GPU note to intro
    if "**147M model**" in joined and "Optimized for Linux" not in joined:
        joined = joined.replace(
            "and produces comparison figures.",
            "and produces comparison figures. **Optimized for Linux GPU servers**: uses CUDA when available, reduced batch size for 147M VRAM.",
        )

    # Architecture description for 147M
    joined = joined.replace(
        "The 37M variant has 1 block, 128-dim hidden; 147M has 12 blocks, 640-dim.",
        "The **147M variant** has 12 blocks, 640-dim hidden.",
    )

    if isinstance(source, list):
        lines = joined.split("\n")
        cell["source"] = [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    else:
        cell["source"] = joined


def main() -> int:
    with open(SRC) as f:
        nb = json.load(f)

    for cell in nb.get("cells", []):
        replace_in_cell(cell)
        cell["outputs"] = []
        cell["execution_count"] = None

    with open(DST, "w") as f:
        json.dump(nb, f, indent=2)

    print(f"Wrote {DST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
