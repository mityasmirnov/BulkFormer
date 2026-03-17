#!/usr/bin/env python3
"""Apply causal gene filter to clinical methods comparison notebooks."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_GENES_BLOCK = '''ALLOWED_CAUSAL_GENES = [
    "VPS33B", "TIMMDC1", "ELAC2", "SLC25A4", "AARS2", "ETFDH", "LIG3",
    "MORC2", "MRPL38", "NFU1", "DARS2", "MRPL44", "MRPS25",
]
# Build long-format data for causal gene swarm: sample_id, gene_symbol, method, metric, is_causal
causal_genes = [g for g in known["KNOWN_MUTATION"].dropna().unique().tolist() if g in ALLOWED_CAUSAL_GENES]
'''

SWARM_OLD = '''# Build long-format data for causal gene swarm: sample_id, gene_symbol, method, metric, is_causal
causal_genes = known["KNOWN_MUTATION"].dropna().unique().tolist()'''

VOLCANO_OLD = 'samples_with_causal = known["SAMPLE_ID"].astype(str).unique().tolist()'
VOLCANO_NEW = 'known_filtered = known[known["KNOWN_MUTATION"].notna() & known["KNOWN_MUTATION"].isin(ALLOWED_CAUSAL_GENES)]\nsamples_with_causal = known_filtered["SAMPLE_ID"].astype(str).unique().tolist()'

CAUSAL_SYM_OLD = 'causal_sym = known[known["SAMPLE_ID"].astype(str) == sid]["KNOWN_MUTATION"].iloc[0]'
CAUSAL_SYM_NEW = 'causal_sym = known_filtered[known_filtered["SAMPLE_ID"].astype(str) == sid]["KNOWN_MUTATION"].iloc[0]'


def apply_to_cell(cell: dict) -> bool:
    """Apply edits to a cell. Returns True if any edit was made."""
    if cell.get("cell_type") != "code":
        return False
    source = cell.get("source")
    if not source:
        return False
    joined = "".join(source) if isinstance(source, list) else source
    changed = False

    if SWARM_OLD in joined:
        joined = joined.replace(SWARM_OLD, ALLOWED_GENES_BLOCK.strip())
        changed = True

    if VOLCANO_OLD in joined and "known_filtered" not in joined:
        joined = joined.replace(VOLCANO_OLD, VOLCANO_NEW)
        changed = True
    if CAUSAL_SYM_OLD in joined and "known_filtered" in joined:
        joined = joined.replace(CAUSAL_SYM_OLD, CAUSAL_SYM_NEW)
        changed = True

    if changed:
        lines = joined.split("\n")
        cell["source"] = [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return changed


def main() -> int:
    for name in ["bulkformer_dx_clinical_methods_comparison.ipynb", "bulkformer_dx_clinical_methods_comparison_147M.ipynb"]:
        path = ROOT / "notebooks" / name
        with open(path) as f:
            nb = json.load(f)
        for cell in nb.get("cells", []):
            apply_to_cell(cell)
        with open(path, "w") as f:
            json.dump(nb, f, indent=2)
        print(f"Updated {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
