#!/usr/bin/env python3
"""Generate step_two_preprocess_counts QC report from preprocess outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    preprocess_dir = REPO_ROOT / "runs" / "demo_preprocess_37M"
    reports_dir = REPO_ROOT / "bulkformer_dx" / "reports"
    figures_dir = reports_dir / "figures"

    from bulkformer_dx.preprocess import preprocess_counts, PreprocessResult
    from bulkformer_dx.benchmark.plots import (
        build_preprocess_sanity_table,
        generate_preprocess_qc_plots,
    )

    counts_path = REPO_ROOT / "data" / "demo_count_data.csv"
    annotation_path = REPO_ROOT / "data" / "gene_length_df.csv"
    if preprocess_dir.exists():
        print("Loading preprocess outputs from disk.", file=sys.stderr)
        result = _load_result_from_disk(preprocess_dir)
    elif counts_path.exists() and annotation_path.exists():
        print("Running preprocess on demo data.", file=sys.stderr)
        result = preprocess_counts(
            counts_path=counts_path,
            annotation_path=annotation_path,
            counts_orientation="samples-by-genes",
        )
    else:
        print(
            f"Neither preprocess dir ({preprocess_dir}) nor demo data found. "
            "Run: python -m bulkformer_dx.cli preprocess --counts data/demo_count_data.csv "
            "--annotation data/gene_length_df.csv --output-dir runs/demo_preprocess_37M "
            "--counts-orientation samples-by-genes",
            file=sys.stderr,
        )
        return 1

    figures_dir.mkdir(parents=True, exist_ok=True)
    saved = generate_preprocess_qc_plots(result, figures_dir)
    sanity_table = build_preprocess_sanity_table(result, n_genes=5)

    report_path = reports_dir / "step_two_preprocess_counts.md"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "# Step Two: Preprocess Counts QC Report",
        "",
        "This report documents the preprocessing outputs required for NB tests: "
        "aligned counts, gene lengths, sample scaling, and QC diagnostics.",
        "",
        "## Artifacts",
        "",
        "| Artifact | Description |",
        "|----------|-------------|",
        "| `aligned_counts.tsv` | Samples × BulkFormer gene panel; missing genes = 0 (flagged invalid) |",
        "| `aligned_tpm.tsv` | TPM aligned to BulkFormer panel |",
        "| `gene_lengths_aligned.tsv` | BulkFormer panel with length_kb and has_length |",
        "| `sample_scaling.tsv` | S_j = Σ K_{jh}/L^{kb}_h per sample |",
        "| `valid_gene_mask.tsv` | is_valid per gene (NB tests use is_valid==1 only) |",
        "",
        "## QC Plots",
        "",
    ]
    for p in saved:
        rel = p.relative_to(reports_dir)
        report_lines.append(f"![{p.stem}]({rel})")
        report_lines.append("")

    report_lines.extend([
        "## Sanity Check Table (5 random valid genes, first sample)",
        "",
        "| ensg_id | counts | TPM | log1p_TPM | length_kb | S_j | expected_count_mapping |",
        "|---------|--------|-----|-----------|-----------|-----|------------------------|",
    ])
    for _, row in sanity_table.iterrows():
        report_lines.append(
            f"| {row['ensg_id']} | {row['counts']:.1f} | {row['TPM']:.2f} | "
            f"{row['log1p_TPM']:.4f} | {row['length_kb']:.3f} | {row['S_j']:.2f} | "
            f"{row['expected_count_mapping']:.2f} |"
        )
    report_lines.append("")
    report_lines.append(
        "The `expected_count_mapping` column shows TPM × S_j/10⁶ × L_kb, "
        "the formula for mapping predicted TPM to expected counts in NB tests."
    )

    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Wrote {report_path}")
    return 0


def _load_result_from_disk(preprocess_dir: Path) -> "PreprocessResult":
    """Load PreprocessResult from preprocess output files."""
    from bulkformer_dx.preprocess import PreprocessResult

    aligned_counts = pd.read_csv(preprocess_dir / "aligned_counts.tsv", sep="\t", index_col=0)
    aligned_tpm = pd.read_csv(preprocess_dir / "aligned_tpm.tsv", sep="\t", index_col=0)
    aligned_log1p_tpm = pd.read_csv(
        preprocess_dir / "aligned_log1p_tpm.tsv", sep="\t", index_col=0
    )
    gene_lengths_aligned = pd.read_csv(
        preprocess_dir / "gene_lengths_aligned.tsv", sep="\t"
    )
    sample_scaling = pd.read_csv(
        preprocess_dir / "sample_scaling.tsv", sep="\t", index_col=0
    )
    valid_gene_mask = pd.read_csv(preprocess_dir / "valid_gene_mask.tsv", sep="\t")
    tpm = pd.read_csv(preprocess_dir / "tpm.tsv", sep="\t", index_col=0)
    log1p_tpm = np.log1p(tpm)
    with open(preprocess_dir / "preprocess_report.json") as f:
        report = json.load(f)

    return PreprocessResult(
        counts=aligned_counts,
        tpm=aligned_tpm,
        log1p_tpm=log1p_tpm,
        aligned_log1p_tpm=aligned_log1p_tpm,
        aligned_counts=aligned_counts,
        aligned_tpm=aligned_tpm,
        gene_lengths_aligned=gene_lengths_aligned,
        sample_scaling=sample_scaling,
        valid_gene_mask=valid_gene_mask,
        report=report,
    )


if __name__ == "__main__":
    raise SystemExit(main())
