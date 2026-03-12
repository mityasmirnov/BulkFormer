#!/usr/bin/env python3
"""Generate BulkFormer DX demo report with full benchmarking metrics.

Loads artifacts from runs/ and reports/, produces reports/bulkformer_dx_demo_report.md
with preprocess QC, anomaly QC, calibration, spike recovery, and benchmark metrics
(AUROC, AUPRC, recall@FDR, PR curves, QQ plots).

Usage:
  PYTHONPATH=. python scripts/generate_demo_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    runs = REPO_ROOT / "runs"
    reports_dir = REPO_ROOT / "reports"
    figures_dir = reports_dir / "figures"

    preprocess_dir = runs / "demo_preprocess_37M"
    preprocess_report_path = preprocess_dir / "preprocess_report.json"
    if not preprocess_report_path.exists():
        print("Error: demo preprocess not found. Run preprocess first.", file=sys.stderr)
        return 1

    with preprocess_report_path.open() as f:
        preprocess_report = json.load(f)

    anomaly_score_dir = runs / "demo_anomaly_score_37M"
    anomaly_run_path = anomaly_score_dir / "anomaly_run.json"
    cohort_scores_path = anomaly_score_dir / "cohort_scores.tsv"
    anomaly_run = {}
    if anomaly_run_path.exists():
        with anomaly_run_path.open() as f:
            anomaly_run = json.load(f)

    base_calibrated_dir = runs / "demo_anomaly_calibrated_37M"
    calibration_summary_path = base_calibrated_dir / "calibration_summary.tsv"
    calibration_run_path = base_calibrated_dir / "calibration_run.json"
    calibration_run = {}
    if calibration_run_path.exists():
        with calibration_run_path.open() as f:
            calibration_run = json.load(f)

    qc_summary_path = reports_dir / "anomaly_qc_summary.json"
    qc_summary = {}
    if qc_summary_path.exists():
        with qc_summary_path.open() as f:
            qc_summary = json.load(f)

    samples = preprocess_report.get("samples", preprocess_report.get("bulkformer_valid_gene_count", "?"))
    if isinstance(samples, dict):
        samples = "?"
    valid_frac = preprocess_report.get("bulkformer_valid_gene_fraction", 0)
    valid_genes = preprocess_report.get("bulkformer_valid_gene_count", "?")
    total_genes = preprocess_report.get("bulkformer_gene_count", "?")

    mean_abs = 0.0
    if cohort_scores_path.exists():
        import pandas as pd
        cohort_scores = pd.read_csv(cohort_scores_path, sep="\t")
        mean_abs = float(cohort_scores["mean_abs_residual"].mean()) if "mean_abs_residual" in cohort_scores.columns else 0

    emp_mean = qc_summary.get("empirical_significant_gene_count_mean", 0)
    abs_mean = qc_summary.get("absolute_significant_gene_count_mean", 0)
    abs_median = qc_summary.get("absolute_significant_gene_count_median", 0)
    spike_targets = qc_summary.get("spike_target_gene_count", 0)
    scored_before = qc_summary.get("spike_targets_scored_before", 0)
    scored_after = qc_summary.get("spike_targets_scored_after", 0)
    rank_improvement_med = qc_summary.get("spike_rank_improvement_median", 0)
    score_gain_med = qc_summary.get("spike_score_gain_median", 0)
    top100_before = qc_summary.get("spike_targets_top100_before", 0)
    top100_after = qc_summary.get("spike_targets_top100_after", 0)
    sig_before = qc_summary.get("spike_targets_significant_before", 0)
    sig_after = qc_summary.get("spike_targets_significant_after", 0)

    auroc = qc_summary.get("auroc", None)
    auprc = qc_summary.get("auprc", None)
    recall_fdr05 = qc_summary.get("recall_at_fdr_05", None)
    recall_fdr10 = qc_summary.get("recall_at_fdr_10", None)
    precision_k = qc_summary.get("precision_at_k_100", None)

    report_lines = [
        "# BulkFormer DX Demo Report",
        "",
        "## Dataset Summary",
        "",
        "- Checkpoint: `model/BulkFormer_37M.pt`",
        "- Demo counts input: `data/demo_count_data.csv`",
        "- Gene lengths: `data/gene_length_df.csv`",
        "- BulkFormer assets: `data/bulkformer_gene_info.csv`, `data/G_tcga.pt`, `data/G_tcga_weight.pt`, `data/esm2_feature_concat.pt`",
        "- Provided normalized comparison matrix: `data/demo_normalized_data.csv`",
        "",
        "## Commands Run",
        "",
        "```bash",
        "python -m bulkformer_dx.cli preprocess \\",
        "  --counts data/demo_count_data.csv \\",
        "  --annotation data/gene_length_df.csv \\",
        "  --output-dir runs/demo_preprocess_37M \\",
        "  --counts-orientation samples-by-genes",
        "PYTHONPATH=. python scripts/demo_spike_inject.py",
        "python -m bulkformer_dx.cli anomaly score \\",
        "  --input runs/demo_preprocess_37M/aligned_log1p_tpm.tsv \\",
        "  --valid-gene-mask runs/demo_preprocess_37M/valid_gene_mask.tsv \\",
        "  --output-dir runs/demo_anomaly_score_37M \\",
        "  --variant 37M --device cuda --mc-passes 16 --mask-prob 0.15",
        "python -m bulkformer_dx.cli anomaly calibrate \\",
        "  --scores runs/demo_anomaly_score_37M \\",
        "  --output-dir runs/demo_anomaly_calibrated_37M \\",
        "  --alpha 0.05",
        "python -m bulkformer_dx.cli anomaly score \\",
        "  --input runs/demo_spike_37M/aligned_log1p_tpm_spiked.tsv \\",
        "  --valid-gene-mask runs/demo_preprocess_37M/valid_gene_mask.tsv \\",
        "  --output-dir runs/demo_spike_anomaly_score_37M \\",
        "  --variant 37M --device cuda --mc-passes 16 --mask-prob 0.15",
        "python -m bulkformer_dx.cli anomaly calibrate \\",
        "  --scores runs/demo_spike_anomaly_score_37M \\",
        "  --output-dir runs/demo_spike_anomaly_calibrated_37M \\",
        "  --alpha 0.05",
        "PYTHONPATH=. python scripts/spike_recovery_metrics.py",
        "PYTHONPATH=. python scripts/generate_demo_report.py",
        "```",
        "",
        "## Key QC Tables",
        "",
        "### Preprocess",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Samples | {samples} |",
        f"| Input genes | {preprocess_report.get('input_genes', '?')} |",
        f"| BulkFormer valid gene fraction | {valid_frac:.3f} |",
        f"| BulkFormer valid genes | {valid_genes} / {total_genes} |",
        "",
        "### Anomaly Score",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| MC passes | {anomaly_run.get('mc_passes', '?')} |",
        f"| Mask prob | {anomaly_run.get('mask_prob', 0.15)} |",
        f"| Mean cohort abs residual | {mean_abs:.4f} |",
        f"| Valid genes | {anomaly_run.get('valid_gene_count', '?')} |",
        "",
        "### Calibration And Spike Recovery",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Mean empirical BY significant genes per sample | {emp_mean} |",
        f"| Mean absolute-outlier significant genes per sample | {abs_mean:.2f} |",
        f"| Spike target pairs evaluated | {spike_targets} |",
        f"| Spike targets scored before / after | {scored_before} / {scored_after} |",
        f"| Spike rank improvement median | {rank_improvement_med} |",
        f"| Spike score gain median | {score_gain_med:.3f} |",
        f"| Spike targets in top 100 before / after | {top100_before} / {top100_after} |",
        f"| Spike targets significant before / after | {sig_before} / {sig_after} |",
        "",
    ]

    if auroc is not None and auprc is not None:
        report_lines.extend([
            "### Benchmark Metrics (Spike Recovery)",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| AUROC | {auroc:.4f} |",
            f"| AUPRC | {auprc:.4f} |",
            f"| Precision at top 100 | {precision_k:.4f} |",
        ])
        if recall_fdr05 is not None:
            report_lines.append(f"| Recall at FDR 0.05 | {recall_fdr05:.4f} |")
        if recall_fdr10 is not None:
            report_lines.append(f"| Recall at FDR 0.10 | {recall_fdr10:.4f} |")
        report_lines.extend(["", ""])

    report_lines.extend([
        "## Figures",
        "",
        "### Preprocess",
        "",
        "![TPM totals](figures/preprocess_tpm_total_hist.png)",
        "",
        "![log1p TPM histogram](figures/preprocess_log1p_hist.png)",
        "",
        "![Valid gene fraction](figures/preprocess_valid_gene_fraction.png)",
        "",
        "### Anomaly Score",
        "",
        "![Mean absolute residual](figures/anomaly_mean_abs_residual_hist.png)",
        "",
        "![Gene coverage fraction](figures/anomaly_gene_coverage_hist.png)",
        "",
        "### Calibration",
        "",
        "![Absolute z-scores](figures/calibration_absolute_zscore_hist.png)",
        "",
        "![Absolute BY-adjusted p-values](figures/calibration_absolute_by_p_hist.png)",
        "",
        "### Spike Recovery",
        "",
        "![Spike rank shift](figures/spike_rank_shift.png)",
        "",
        "![Spike rank improvement](figures/spike_rank_improvement_hist.png)",
        "",
    ])

    if (figures_dir / "spike_pr_curve.png").exists():
        report_lines.extend([
            "### Benchmark",
            "",
            "![PR curve](figures/spike_pr_curve.png)",
            "",
            "![P-value QQ](figures/spike_pvalue_qq.png)",
            "",
        ])

    report_lines.extend([
        "## Interpretation",
        "",
        "- The demo preprocessing path converts counts to TPM and aligns to the BulkFormer gene panel.",
        "- Anomaly scoring uses Monte Carlo masking; residual magnitude ranks genes.",
        "- Spiked genes gain rank sharply and many become significant after recalibration.",
        "- The empirical cohort BY path is conservative; the normalized absolute-outlier path is more permissive.",
        "",
    ])

    if auroc is not None:
        report_lines.extend([
            "- Benchmark metrics (AUROC, AUPRC, recall@FDR) quantify spike recovery performance.",
            "",
        ])

    report_lines.extend([
        "## Troubleshooting Notes",
        "",
        "- Run `PYTHONPATH=. python scripts/demo_spike_inject.py` before anomaly scoring on spiked data.",
        "- Run `PYTHONPATH=. python scripts/spike_recovery_metrics.py` after calibration to produce benchmark metrics.",
        "",
    ])

    output_path = reports_dir / "bulkformer_dx_demo_report.md"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
