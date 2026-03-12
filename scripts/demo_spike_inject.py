#!/usr/bin/env python3
"""Inject controlled outliers into demo aligned_log1p_tpm for spike recovery validation.

Uses bulkformer_dx.benchmark.inject.inject_outliers_log1p to create spiked expression
matrix and saves to runs/demo_spike_37M/ for downstream anomaly scoring and calibration.

Usage:
  python scripts/demo_spike_inject.py
  python scripts/demo_spike_inject.py --preprocess-dir runs/demo_preprocess_37M --output-dir runs/demo_spike_37M
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inject controlled outliers into demo aligned_log1p_tpm."
    )
    parser.add_argument(
        "--preprocess-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "demo_preprocess_37M",
        help="Preprocess output directory with aligned_log1p_tpm.tsv and valid_gene_mask.tsv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "demo_spike_37M",
        help="Output directory for spiked matrix and metadata.",
    )
    parser.add_argument(
        "--n-inject",
        type=int,
        default=40,
        help="Number of (sample, gene) pairs to perturb. Default 40.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=3.0,
        help="Perturbation magnitude in residual units. Default 3.0.",
    )
    parser.add_argument(
        "--direction",
        choices=("up", "down", "both"),
        default="both",
        help="Direction of perturbation. Default both.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed. Default 0.",
    )
    args = parser.parse_args()

    preprocess_dir = args.preprocess_dir
    output_dir = args.output_dir

    expr_path = preprocess_dir / "aligned_log1p_tpm.tsv"
    mask_path = preprocess_dir / "valid_gene_mask.tsv"
    if not expr_path.exists():
        print(
            f"Error: {expr_path} not found. Run preprocess first.",
            file=sys.stderr,
        )
        return 1
    if not mask_path.exists():
        print(
            f"Error: {mask_path} not found. Run preprocess first.",
            file=sys.stderr,
        )
        return 1

    expression = pd.read_csv(expr_path, sep="\t", index_col=0)
    valid_gene_mask = pd.read_csv(mask_path, sep="\t")
    if "ensg_id" not in valid_gene_mask.columns or "is_valid" not in valid_gene_mask.columns:
        print(
            "Error: valid_gene_mask must have ensg_id and is_valid columns.",
            file=sys.stderr,
        )
        return 1

    gene_mask_indexed = valid_gene_mask.drop_duplicates(subset=["ensg_id"]).set_index("ensg_id")
    resolved = gene_mask_indexed.reindex(expression.columns)
    if resolved["is_valid"].isna().any():
        print(
            "Error: valid_gene_mask does not cover all expression genes.",
            file=sys.stderr,
        )
        return 1
    valid_gene_flags = resolved["is_valid"].to_numpy(dtype=bool)
    n_samples, n_genes = expression.shape
    valid_mask = np.broadcast_to(
        valid_gene_flags[np.newaxis, :],
        (n_samples, n_genes),
    ).copy()

    from bulkformer_dx.benchmark.inject import inject_outliers_log1p

    result = inject_outliers_log1p(
        expression.to_numpy(dtype=np.float32),
        valid_mask,
        n_inject=args.n_inject,
        scale=args.scale,
        direction=args.direction,
        seed=args.seed,
    )

    sample_ids = list(expression.index)
    gene_ids = list(expression.columns)

    spiked_df = pd.DataFrame(
        result.expression_perturbed,
        index=sample_ids,
        columns=gene_ids,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    spiked_df.to_csv(output_dir / "aligned_log1p_tpm_spiked.tsv", sep="\t")

    injected_pairs = [
        {
            "sample_id": sample_ids[si],
            "gene_id": gene_ids[gi],
            "sample_idx": si,
            "gene_idx": gi,
            "direction": result.directions[i],
        }
        for i, (si, gi) in enumerate(
            zip(result.injected_sample_idx, result.injected_gene_idx, strict=True)
        )
    ]

    metadata = {
        "preprocess_dir": str(preprocess_dir),
        "n_inject": args.n_inject,
        "scale": args.scale,
        "direction": args.direction,
        "seed": args.seed,
        "n_samples": n_samples,
        "n_genes": n_genes,
        "injected_sample_idx": result.injected_sample_idx,
        "injected_gene_idx": result.injected_gene_idx,
        "directions": result.directions,
        "injected_pairs": injected_pairs,
        "ground_truth_mask_shape": list(result.ground_truth_mask.shape),
    }
    with (output_dir / "spike_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    np.save(
        output_dir / "ground_truth_mask.npy",
        result.ground_truth_mask,
        allow_pickle=False,
    )

    print(
        f"Injected {len(injected_pairs)} outliers. Output: {output_dir}/aligned_log1p_tpm_spiked.tsv"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
