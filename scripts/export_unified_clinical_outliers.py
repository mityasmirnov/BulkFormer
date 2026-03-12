#!/usr/bin/env python3
"""Export unified OUTRIDER-style long-format table from clinical methods calibration results.

Merges absolute_outliers.tsv from all calibration methods, adds gene and sample
annotations, and writes a single TSV suitable for R (e.g. read.delim()).

Usage:
  python scripts/export_unified_clinical_outliers.py
  python scripts/export_unified_clinical_outliers.py \\
    --runs-dir runs/clinical_methods_37M \\
    --gene-annotation data/clinical_rnaseq/gene_annotation_v29.tsv \\
    --sample-annotation data/clinical_rnaseq/sample_annotation.tsv \\
    --output runs/clinical_methods_37M/unified_outliers.tsv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Set, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = REPO_ROOT / "runs" / "clinical_methods_37M"
DEFAULT_GENE_ANNOTATION = REPO_ROOT / "data" / "clinical_rnaseq" / "gene_annotation_v29.tsv"
DEFAULT_SAMPLE_ANNOTATION = REPO_ROOT / "data" / "clinical_rnaseq" / "sample_annotation.tsv"

METHODS = ["none", "student_t", "nb_approx", "nb_outrider", "knn_local", "nll"]
METHOD_COLUMNS = [
    "expected_mu",
    "expected_sigma",
    "z_score",
    "raw_p_value",
    "by_adj_p_value",
    "is_significant",
]
SAMPLE_ANNOTATION_COLUMNS = ["SAMPLE_ID", "KNOWN_MUTATION", "CATEGORY", "gender", "TISSUE"]


def load_gene_annotation(path: Path) -> Dict[str, str]:
    """Build gene_id -> gene_name map. Handles duplicates by taking first."""
    df = pd.read_csv(path, sep="\t")
    if "gene_id" not in df.columns or "gene_name" not in df.columns:
        raise ValueError(f"gene_annotation must have gene_id and gene_name columns, got {list(df.columns)}")
    dedup = df[["gene_id", "gene_name"]].drop_duplicates(subset=["gene_id"], keep="first")
    return dict(zip(dedup["gene_id"].astype(str), dedup["gene_name"].astype(str), strict=True))


def load_sample_annotation(path: Path) -> pd.DataFrame:
    """Load sample annotation, keeping required columns."""
    df = pd.read_csv(path, sep="\t")
    available = [c for c in SAMPLE_ANNOTATION_COLUMNS if c in df.columns]
    return df[available].copy()


def load_method_outliers(runs_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load absolute_outliers.tsv for each method that exists."""
    result: Dict[str, pd.DataFrame] = {}
    for method in METHODS:
        path = runs_dir / f"calibrated_{method}" / "absolute_outliers.tsv"
        if path.exists():
            df = pd.read_csv(path, sep="\t")
            required = {"sample_id", "gene", "observed_log1p_tpm"} | set(METHOD_COLUMNS)
            missing = required - set(df.columns)
            if missing:
                continue
            result[method] = df
    return result


def merge_methods(method_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Full outer join on (sample_id, gene) across all methods."""
    if not method_dfs:
        return pd.DataFrame()

    all_keys: Set[Tuple[str, str]] = set()
    for df in method_dfs.values():
        keys = set(zip(df["sample_id"].astype(str), df["gene"].astype(str), strict=True))
        all_keys.update(keys)

    keys_df = pd.DataFrame(
        list(all_keys),
        columns=["sample_id", "gene"],
    ).drop_duplicates()

    # observed_log1p_tpm: same across methods, take from first
    first_df = next(iter(method_dfs.values()))
    obs = first_df[["sample_id", "gene", "observed_log1p_tpm"]].copy()
    obs["sample_id"] = obs["sample_id"].astype(str)
    obs["gene"] = obs["gene"].astype(str)
    merged = keys_df.merge(obs, on=["sample_id", "gene"], how="left")

    for method, df in method_dfs.items():
        df = df.copy()
        df["sample_id"] = df["sample_id"].astype(str)
        df["gene"] = df["gene"].astype(str)
        cols = ["sample_id", "gene"] + METHOD_COLUMNS
        df = df[cols].rename(columns={c: f"{method}_{c}" for c in METHOD_COLUMNS})
        merged = merged.merge(df, on=["sample_id", "gene"], how="left")

    return merged


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export unified OUTRIDER-style table from clinical methods results.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS,
        help="Directory containing calibrated_* subdirs",
    )
    parser.add_argument(
        "--gene-annotation",
        type=Path,
        default=DEFAULT_GENE_ANNOTATION,
        help="Gene annotation TSV path",
    )
    parser.add_argument(
        "--sample-annotation",
        type=Path,
        default=DEFAULT_SAMPLE_ANNOTATION,
        help="Sample annotation TSV path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output TSV path (default: runs-dir/unified_outliers.tsv)",
    )
    args = parser.parse_args()

    output_path = args.output or (args.runs_dir / "unified_outliers.tsv")

    if not args.runs_dir.exists():
        print(f"Error: runs-dir not found: {args.runs_dir}", file=sys.stderr)
        return 1
    if not args.gene_annotation.exists():
        print(f"Error: gene-annotation not found: {args.gene_annotation}", file=sys.stderr)
        return 1
    if not args.sample_annotation.exists():
        print(f"Error: sample-annotation not found: {args.sample_annotation}", file=sys.stderr)
        return 1

    gene_map = load_gene_annotation(args.gene_annotation)
    sample_anno = load_sample_annotation(args.sample_annotation)
    method_dfs = load_method_outliers(args.runs_dir)

    if not method_dfs:
        print("Error: no valid method outputs found in runs-dir", file=sys.stderr)
        return 1

    merged = merge_methods(method_dfs)

    # Add Gene_Name
    merged["Gene_Name"] = merged["gene"].map(gene_map)

    # Add sample annotations
    rename_map = {
        "SAMPLE_ID": "sample_id",
        "KNOWN_MUTATION": "known_causal_gene",
        "CATEGORY": "sample_category",
        "gender": "sample_gender",
        "TISSUE": "sample_tissue",
    }
    sample_anno = sample_anno.rename(columns={k: v for k, v in rename_map.items() if k in sample_anno.columns})
    merge_cols = ["sample_id"] + [c for c in ["known_causal_gene", "sample_category", "sample_gender", "sample_tissue"] if c in sample_anno.columns]
    merged = merged.merge(sample_anno[merge_cols], on="sample_id", how="left")
    merged["is_known_causal_gene"] = (
        merged["Gene_Name"].astype(str) == merged["known_causal_gene"].astype(str)
    ) & merged["known_causal_gene"].notna()
    merged["is_known_causal_gene"] = merged["is_known_causal_gene"].fillna(False)

    # Rename and reorder columns for output
    merged = merged.rename(columns={
        "sample_id": "SampleID",
        "gene": "GeneID",
        "CATEGORY": "sample_category",
        "gender": "sample_gender",
        "TISSUE": "sample_tissue",
    })

    output_cols = [
        "SampleID",
        "GeneID",
        "Gene_Name",
        "known_causal_gene",
        "is_known_causal_gene",
        "sample_category",
        "sample_gender",
        "sample_tissue",
        "observed_log1p_tpm",
    ]
    for method in METHODS:
        for c in METHOD_COLUMNS:
            output_cols.append(f"{method}_{c}")
    merged = merged[[c for c in output_cols if c in merged.columns]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, sep="\t", index=False)
    print(f"Wrote {output_path} ({len(merged):,} rows, {len(merged.columns)} columns)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
