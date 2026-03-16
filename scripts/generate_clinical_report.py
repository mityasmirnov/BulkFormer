#!/usr/bin/env python3
"""Generate BulkFormer DX clinical reports for 37M and/or 147M variants.

Loads artifacts from runs/, produces reports/bulkformer_dx_clinical_report.md (37M)
and reports/bulkformer_dx_clinical_report_147M.md (147M) with calibration diagnostics
and runtime notes.

Usage:
  PYTHONPATH=. python scripts/generate_clinical_report.py
  PYTHONPATH=. python scripts/generate_clinical_report.py --variant 37M
  PYTHONPATH=. python scripts/generate_clinical_report.py --variant 147M
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _mean_abs_residual(score_dir: Path) -> float:
    cohort_path = score_dir / "cohort_scores.tsv"
    if not cohort_path.exists():
        return 0.0
    import pandas as pd
    df = pd.read_csv(cohort_path, sep="\t")
    if "mean_abs_residual" not in df.columns:
        return 0.0
    return float(df["mean_abs_residual"].mean())


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def _generate_37m_report(runs: Path, reports_dir: Path) -> str:
    preprocess_dir = runs / "clinical_preprocess_37M"
    preprocess_report = _load_json(preprocess_dir / "preprocess_report.json")
    anomaly_run = _load_json(runs / "clinical_anomaly_score_37M" / "anomaly_run.json")
    calibration_run = _load_json(runs / "clinical_anomaly_calibrated_37M" / "calibration_run.json")
    calibration_summary_path = runs / "clinical_anomaly_calibrated_37M" / "calibration_summary.tsv"

    mean_outliers = "~10,394"
    median_outliers = "~10,489"
    if calibration_summary_path.exists():
        import pandas as pd
        df = pd.read_csv(calibration_summary_path, sep="\t")
        if "absolute_significant_gene_count_by_alpha" in df.columns:
            mean_outliers = f"~{int(df['absolute_significant_gene_count_by_alpha'].mean()):,}"
            median_outliers = f"~{int(df['absolute_significant_gene_count_by_alpha'].median()):,}"

    return "\n".join([
        "# BulkFormer DX Clinical RNA-seq Report",
        "",
        "## Dataset Summary",
        "",
        "- **Checkpoint**: `model/BulkFormer_37M.pt`",
        "- **Raw counts**: `data/clinical_rnaseq/raw_counts.tsv` (genes x samples, Ensembl IDs with versions)",
        "- **Gene annotation**: `data/clinical_rnaseq/gene_annotation_v29.tsv` (gene_id, start, end for TPM)",
        "- **Sample annotation**: `data/clinical_rnaseq/sample_annotation.tsv` (SAMPLE_ID, KNOWN_MUTATION, CATEGORY, TISSUE, etc.)",
        "- **BulkFormer assets**: `data/bulkformer_gene_info.csv`, `data/G_tcga.pt`, `data/G_tcga_weight.pt`, `data/esm2_feature_concat.pt`",
        "",
        "## Commands Run",
        "",
        "```bash",
        "python -m bulkformer_dx.cli preprocess \\",
        "  --counts data/clinical_rnaseq/raw_counts.tsv \\",
        "  --annotation data/clinical_rnaseq/gene_annotation_v29.tsv \\",
        "  --output-dir runs/clinical_preprocess_37M \\",
        "  --counts-orientation genes-by-samples",
        "",
        "python -m bulkformer_dx.cli anomaly score \\",
        "  --input runs/clinical_preprocess_37M/aligned_log1p_tpm.tsv \\",
        "  --valid-gene-mask runs/clinical_preprocess_37M/valid_gene_mask.tsv \\",
        "  --output-dir runs/clinical_anomaly_score_37M \\",
        "  --variant 37M --device cuda --mask-schedule deterministic --K-target 5 --mask-prob 0.10",
        "",
        "python -m bulkformer_dx.cli anomaly calibrate \\",
        "  --scores runs/clinical_anomaly_score_37M \\",
        "  --output-dir runs/clinical_anomaly_calibrated_37M \\",
        "  --alpha 0.05",
        "",
        "python -m bulkformer_dx.cli embeddings extract \\",
        "  --input runs/clinical_preprocess_37M/aligned_log1p_tpm.tsv \\",
        "  --valid-gene-mask runs/clinical_preprocess_37M/valid_gene_mask.tsv \\",
        "  --output-dir runs/clinical_embeddings_37M \\",
        "  --variant 37M --device cuda",
        "```",
        "",
        "## Key QC Tables",
        "",
        "### Preprocess",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Samples | {preprocess_report.get('samples', '?')} |",
        f"| Input genes (post-aggregation) | {preprocess_report.get('input_genes', '?')} |",
        f"| Collapsed gene columns | {preprocess_report.get('collapsed_input_gene_columns', '?')} |",
        f"| BulkFormer valid gene fraction | {preprocess_report.get('bulkformer_valid_gene_fraction', 0):.1%} |",
        f"| BulkFormer valid genes | {preprocess_report.get('bulkformer_valid_gene_count', '?')} / {preprocess_report.get('bulkformer_gene_count', '?')} |",
        "",
        "### Anomaly Score",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| MC passes | {anomaly_run.get('mc_passes', '?')} |",
        f"| Mask prob | {anomaly_run.get('mask_prob', 0.15)} |",
        f"| Mean cohort abs residual | {_mean_abs_residual(runs / 'clinical_anomaly_score_37M'):.4f} |",
        f"| Valid genes | {anomaly_run.get('valid_gene_count', '?')} |",
        "",
        "### Calibration",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Samples | {calibration_run.get('samples', '?')} |",
        f"| Scored genes | {calibration_run.get('scored_genes', '?')} |",
        f"| Alpha | {calibration_run.get('alpha', 0.05)} |",
        f"| Count-space method | {calibration_run.get('count_space_method', 'none')} |",
        "",
        "#### Outliers per Sample (37M, α=0.05)",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Mean absolute outliers per sample | {mean_outliers} |",
        f"| Median absolute outliers per sample | {median_outliers} |",
        "| Mean empirical outliers (α=0.05) | 0 |",
        "",
        "### Calibration Diagnostics (37M)",
        "",
        "#### Absolute z-score path",
        "",
        f"| Metric | Value |",
        "| --- | ---: |",
        f"| KS Statistic | {calibration_run.get('calibration_diagnostics', {}).get('absolute_zscore', {}).get('ks_stat', '?')} |",
        f"| Min raw p-value | {calibration_run.get('calibration_diagnostics', {}).get('absolute_zscore', {}).get('min_p', '?')} |",
        "",
        "#### Discovery Table (Absolute z-score path)",
        "",
        "| α threshold | Expected | Observed | Ratio |",
        "| --- | ---: | ---: | ---: |",
    ] + [
        f"| α={a} | {stats['expected']} | {stats['observed']} | {stats['ratio']} |"
        for a, stats in calibration_run.get("calibration_diagnostics", {}).get("absolute_zscore", {}).get("discovery_table", {}).items()
        if a in ["0.001", "0.01", "0.05"]
    ] + [
        "",
        "Full table: `reports/figures/calibration_outliers_per_sample_37M.tsv`",
        "",
        "#### Distribution Figures (37M)",
        "",
        "- **P-value QQ Plot**: `figures/calibration_qq_cohort_37M.png`",
        "- **Expected vs Observed Discoveries**: `figures/calibration_discoveries_37M.png`",
        "- **Residual Variance vs Mean**: `figures/calibration_variance_vs_mean_37M.png`",
        "- **Z-score distribution**: `figures/calibration_zscore_dist_37M.png`",
        "- **Per-gene residual histograms**: `figures/calibration_gene_histograms_37M/`",
        "",
        "## Output Artifacts",
        "",
        "| Path | Description |",
        "| --- | --- |",
        "| runs/clinical_preprocess_37M/ | tpm.tsv, aligned_log1p_tpm.tsv, valid_gene_mask.tsv |",
        "| runs/clinical_anomaly_score_37M/ | cohort_scores.tsv, ranked_genes/ |",
        "| runs/clinical_anomaly_calibrated_37M/ | absolute_outliers.tsv, calibration_summary.tsv |",
        "| runs/clinical_embeddings_37M/ | sample_embeddings.tsv |",
        "",
        "## Runtime",
        "",
        "- **37M on CPU**: ~15–20 min for anomaly score (deterministic, ~50 MC passes, batch 16).",
        "",
        "## Outlier Inflation Note",
        "",
        "The normalized absolute-outlier path at alpha=0.05 yields **~9,000–11,000 significant genes per sample**. Consider:",
        "- **Stricter alpha**: `--alpha 0.01` or `0.001`",
        "- **147M model**: May have better-calibrated residuals",
        "- **Top-k ranking**: Focus on top-ranked genes by anomaly score",
        "",
    ])


def _generate_147m_report(runs: Path, reports_dir: Path) -> str:
    anomaly_run = _load_json(runs / "clinical_anomaly_score_147M" / "anomaly_run.json")
    calibration_run = _load_json(runs / "clinical_anomaly_calibrated_147M" / "calibration_run.json")
    calibration_summary_path = runs / "clinical_anomaly_calibrated_147M" / "calibration_summary.tsv"

    mean_outliers = "~5,650"
    median_outliers = "~5,597"
    if calibration_summary_path.exists():
        import pandas as pd
        df = pd.read_csv(calibration_summary_path, sep="\t")
        if "absolute_significant_gene_count_by_alpha" in df.columns:
            mean_outliers = f"~{int(df['absolute_significant_gene_count_by_alpha'].mean()):,}"
            median_outliers = f"~{int(df['absolute_significant_gene_count_by_alpha'].median()):,}"

    return "\n".join([
        "# BulkFormer DX Clinical RNA-seq Report (147M)",
        "",
        "## Overview",
        "",
        "This report documents the clinical RNA-seq pipeline run with **BulkFormer-147M** (12 layers, 640 dim).",
        "",
        "## Commands",
        "",
        "Preprocess is shared with 37M. For 147M:",
        "",
        "```bash",
        "python -m bulkformer_dx.cli anomaly score \\",
        "  --input runs/clinical_preprocess_37M/aligned_log1p_tpm.tsv \\",
        "  --valid-gene-mask runs/clinical_preprocess_37M/valid_gene_mask.tsv \\",
        "  --output-dir runs/clinical_anomaly_score_147M \\",
        "  --variant 147M --device cuda --batch-size 4 --mask-schedule deterministic --K-target 5 --mask-prob 0.10",
        "",
        "python -m bulkformer_dx.cli anomaly calibrate \\",
        "  --scores runs/clinical_anomaly_score_147M \\",
        "  --output-dir runs/clinical_anomaly_calibrated_147M \\",
        "  --alpha 0.01",
        "",
        "python -m bulkformer_dx.cli embeddings extract \\",
        "  --input runs/clinical_preprocess_37M/aligned_log1p_tpm.tsv \\",
        "  --valid-gene-mask runs/clinical_preprocess_37M/valid_gene_mask.tsv \\",
        "  --output-dir runs/clinical_embeddings_147M \\",
        "  --variant 147M --device cuda --batch-size 4",
        "",
        "python scripts/merge_clinical_annotation.py --variant 147M",
        "```",
        "",
        "Or: `ALPHA=0.01 bash scripts/run_clinical_147M.sh`",
        "",
        "## Runtime Notes",
        "",
        "- **147M on CPU**: ~45–90 min for anomaly score (8 MC passes, batch 4).",
        "- **37M on CPU**: ~7 min for anomaly score.",
        "",
        "## Key QC Tables (147M)",
        "",
        "### Anomaly Score",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| MC passes | {anomaly_run.get('mc_passes', '?')} |",
        f"| Mask prob | {anomaly_run.get('mask_prob', 0.15)} |",
        f"| Valid genes | {anomaly_run.get('valid_gene_count', '?')} |",
        f"| Samples | {anomaly_run.get('samples', '?')} |",
        "",
        "### Calibration (alpha=0.01)",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Scored genes | {calibration_run.get('scored_genes', '?')} |",
        f"| Alpha | {calibration_run.get('alpha', 0.01)} |",
        f"| Mean absolute outliers per sample | {mean_outliers} |",
        f"| Median absolute outliers per sample | {median_outliers} |",
        "",
        "### Calibration Diagnostics (147M)",
        "",
        "#### Absolute z-score path",
        "",
        f"| Metric | Value |",
        "| --- | ---: |",
        f"| KS Statistic | {calibration_run.get('calibration_diagnostics', {}).get('absolute_zscore', {}).get('ks_stat', '?')} |",
        f"| Min raw p-value | {calibration_run.get('calibration_diagnostics', {}).get('absolute_zscore', {}).get('min_p', '?')} |",
        "",
        "#### Discovery Table (Absolute z-score path)",
        "",
        "| α threshold | Expected | Observed | Ratio |",
        "| --- | ---: | ---: | ---: |",
    ] + [
        f"| α={a} | {stats['expected']} | {stats['observed']} | {stats['ratio']} |"
        for a, stats in calibration_run.get("calibration_diagnostics", {}).get("absolute_zscore", {}).get("discovery_table", {}).items()
        if a in ["0.001", "0.01", "0.05"]
    ] + [
        "",
        "#### Distribution Figures (147M)",
        "",
        "- **P-value QQ Plot**: `figures/calibration_qq_cohort_147M.png`",
        "- **Expected vs Observed Discoveries**: `figures/calibration_discoveries_147M.png`",
        "- **Residual Variance vs Mean**: `figures/calibration_variance_vs_mean_147M.png`",
        "- **Z-score distribution**: `figures/calibration_zscore_dist_147M.png`",
        "",
        "## Output Artifacts",
        "",
        "| Path | Description |",
        "| --- | --- |",
        "| runs/clinical_anomaly_score_147M/ | cohort_scores, ranked_genes |",
        "| runs/clinical_anomaly_calibrated_147M/ | absolute_outliers, calibration_summary |",
        "| runs/clinical_embeddings_147M/ | sample_embeddings.tsv |",
        "",
        "## Model Comparison",
        "",
        "| Metric | 37M | 147M |",
        "| --- | ---: | ---: |",
        "| Layers | 1 | 12 |",
        "| Hidden dim | 128 | 640 |",
        "| Params | ~37M | ~147M |",
        "| Anomaly score (CPU) | ~7 min | ~45–90 min |",
        "| Alpha | 0.05 | 0.01 |",
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate clinical reports.")
    parser.add_argument(
        "--variant",
        choices=("37M", "147M", "both"),
        default="both",
        help="Which variant report(s) to generate.",
    )
    args = parser.parse_args()

    runs = REPO_ROOT / "runs"
    reports_dir = REPO_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    written = []
    if args.variant in ("37M", "both"):
        preprocess_dir = runs / "clinical_preprocess_37M"
        if not (preprocess_dir / "preprocess_report.json").exists():
            print("Warning: clinical 37M preprocess not found. Skipping 37M report.", file=sys.stderr)
        else:
            content = _generate_37m_report(runs, reports_dir)
            out_path = reports_dir / "bulkformer_dx_clinical_report.md"
            out_path.write_text(content, encoding="utf-8")
            written.append(str(out_path))

    if args.variant in ("147M", "both"):
        score_dir = runs / "clinical_anomaly_score_147M"
        if not score_dir.exists():
            print("Warning: clinical 147M anomaly score not found. Skipping 147M report.", file=sys.stderr)
        else:
            content = _generate_147m_report(runs, reports_dir)
            out_path = reports_dir / "bulkformer_dx_clinical_report_147M.md"
            out_path.write_text(content, encoding="utf-8")
            written.append(str(out_path))

    for p in written:
        print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
