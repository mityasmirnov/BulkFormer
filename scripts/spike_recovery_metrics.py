#!/usr/bin/env python3
"""Compute spike recovery table and benchmark metrics from demo base vs spiked runs.

Compares base and spiked anomaly scores/calibration, produces spike_recovery.tsv
and anomaly_qc_summary.json with full benchmarking fields (AUROC, AUPRC, recall@FDR, etc.).
Generates PR curves, p-value histograms, QQ plots when figures_dir is specified.

Usage:
  python scripts/spike_recovery_metrics.py
  python scripts/spike_recovery_metrics.py --spike-dir runs/demo_spike_37M \
    --base-score runs/demo_anomaly_score_37M --spike-score runs/demo_spike_anomaly_score_37M \
    --base-calibrated runs/demo_anomaly_calibrated_37M --spike-calibrated runs/demo_spike_anomaly_calibrated_37M
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_ranked_scores(ranked_dir: Path) -> dict[str, dict[str, float]]:
    """Load per-sample gene->score mapping from ranked_genes directory."""
    result: dict[str, dict[str, float]] = {}
    if not ranked_dir.exists():
        return result
    for p in sorted(ranked_dir.glob("*.tsv")):
        sample_id = p.stem
        df = pd.read_csv(p, sep="\t")
        gene_col = "ensg_id" if "ensg_id" in df.columns else "gene"
        score_col = "anomaly_score" if "anomaly_score" in df.columns else "score_gene"
        if gene_col not in df.columns or score_col not in df.columns:
            continue
        result[sample_id] = dict(
            zip(df[gene_col].astype(str), df[score_col].astype(float), strict=True)
        )
    return result


def _load_calibrated_significance(calibrated_dir: Path) -> dict[tuple[str, str], bool]:
    """Load (sample_id, gene_id) -> is_significant from absolute_outliers or ranked_genes."""
    result: dict[tuple[str, str], bool] = {}
    abs_path = calibrated_dir / "absolute_outliers.tsv"
    if abs_path.exists():
        df = pd.read_csv(abs_path, sep="\t")
        sample_col = "sample_id"
        gene_col = "gene" if "gene" in df.columns else "ensg_id"
        sig_col = "is_significant"
        if sample_col in df.columns and gene_col in df.columns and sig_col in df.columns:
            for _, row in df.iterrows():
                result[(str(row[sample_col]), str(row[gene_col]))] = bool(row[sig_col])
        return result
    ranked_dir = calibrated_dir / "ranked_genes"
    if ranked_dir.exists():
        tsv_files = list(ranked_dir.glob("*.tsv"))
        if tsv_files:
            first_df = pd.read_csv(tsv_files[0], sep="\t", nrows=1)
            by_q = "by_adj_p_value" if "by_adj_p_value" in first_df.columns else "by_q_value"
            for p in tsv_files:
                sample_id = p.stem
                df = pd.read_csv(p, sep="\t")
                gene_col = "ensg_id" if "ensg_id" in df.columns else "gene"
                if by_q in df.columns and gene_col in df.columns:
                    for _, row in df.iterrows():
                        result[(sample_id, str(row[gene_col]))] = float(row[by_q]) <= 0.05
    return result


def _load_calibrated_pvalues(calibrated_dir: Path) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    """Load (sample_id, gene_id) -> p_raw and p_adj from calibrated outputs."""
    p_raw: dict[tuple[str, str], float] = {}
    p_adj: dict[tuple[str, str], float] = {}
    abs_path = calibrated_dir / "absolute_outliers.tsv"
    if abs_path.exists():
        df = pd.read_csv(abs_path, sep="\t")
        sample_col = "sample_id"
        gene_col = "gene" if "gene" in df.columns else "ensg_id"
        for _, row in df.iterrows():
            key = (str(row[sample_col]), str(row[gene_col]))
            p_raw[key] = float(row.get("raw_p_value", 0.5))
            p_adj[key] = float(row.get("by_adj_p_value", row.get("by_q_value", 1.0)))
    return p_raw, p_adj


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute spike recovery and benchmark metrics."
    )
    parser.add_argument(
        "--spike-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "demo_spike_37M",
        help="Spike output directory with spike_metadata.json and ground_truth_mask.npy.",
    )
    parser.add_argument(
        "--base-score",
        type=Path,
        default=REPO_ROOT / "runs" / "demo_anomaly_score_37M",
        help="Base (non-spiked) anomaly score output.",
    )
    parser.add_argument(
        "--spike-score",
        type=Path,
        default=REPO_ROOT / "runs" / "demo_spike_anomaly_score_37M",
        help="Spiked anomaly score output.",
    )
    parser.add_argument(
        "--base-calibrated",
        type=Path,
        default=REPO_ROOT / "runs" / "demo_anomaly_calibrated_37M",
        help="Base calibrated output.",
    )
    parser.add_argument(
        "--spike-calibrated",
        type=Path,
        default=REPO_ROOT / "runs" / "demo_spike_anomaly_calibrated_37M",
        help="Spiked calibrated output.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "reports",
        help="Output directory for spike_recovery.tsv and anomaly_qc_summary.json.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=REPO_ROOT / "reports" / "figures",
        help="Directory for benchmark figures (PR curve, QQ, p-value hist).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold for is_significant. Default 0.05.",
    )
    args = parser.parse_args()

    spike_dir = args.spike_dir
    metadata_path = spike_dir / "spike_metadata.json"
    gt_path = spike_dir / "ground_truth_mask.npy"
    if not metadata_path.exists():
        print(
            f"Error: {metadata_path} not found. Run demo_spike_inject.py first.",
            file=sys.stderr,
        )
        return 1
    if not args.spike_score.exists():
        print(
            f"Error: {args.spike_score} not found. Run anomaly score on spiked data first.",
            file=sys.stderr,
        )
        return 1
    if not args.spike_calibrated.exists():
        print(
            f"Error: {args.spike_calibrated} not found. Run anomaly calibrate on spiked scores first.",
            file=sys.stderr,
        )
        return 1

    with metadata_path.open() as f:
        metadata = json.load(f)
    injected_pairs = metadata["injected_pairs"]
    n_samples = metadata["n_samples"]
    n_genes = metadata["n_genes"]
    sample_ids = None
    gene_ids = None

    if gt_path.exists():
        ground_truth_mask = np.load(gt_path)
    else:
        ground_truth_mask = np.zeros((n_samples, n_genes), dtype=bool)
        for p in injected_pairs:
            si, gi = p["sample_idx"], p["gene_idx"]
            if si < n_samples and gi < n_genes:
                ground_truth_mask[si, gi] = True

    base_ranked = _load_ranked_scores(args.base_score / "ranked_genes")
    spike_ranked = _load_ranked_scores(args.spike_score / "ranked_genes")
    base_sig = _load_calibrated_significance(args.base_calibrated)
    spike_sig = _load_calibrated_significance(args.spike_calibrated)
    _, spike_p_adj = _load_calibrated_pvalues(args.spike_calibrated)

    preprocess_dir = Path(metadata.get("preprocess_dir", str(REPO_ROOT / "runs" / "demo_preprocess_37M")))
    expr_path = preprocess_dir / "aligned_log1p_tpm.tsv"
    if expr_path.exists():
        expr_df = pd.read_csv(expr_path, sep="\t", index_col=0)
        sample_ids = list(expr_df.index)
        gene_ids = list(expr_df.columns)
    if sample_ids is None or gene_ids is None:
        sample_ids = [f"sample_{i}" for i in range(n_samples)]
        gene_ids = [f"gene_{j}" for j in range(n_genes)]

    sample_idx = {s: i for i, s in enumerate(sample_ids)}
    gene_idx = {g: i for i, g in enumerate(gene_ids)}

    def _rank_and_score(sample_id: str, gene_id: str, ranked: dict[str, dict[str, float]]) -> tuple[int | None, float]:
        if sample_id not in ranked:
            return None, 0.0
        scores = ranked[sample_id]
        if gene_id not in scores:
            return None, 0.0
        sc = scores[gene_id]
        sorted_genes = sorted(scores.keys(), key=lambda g: -scores[g])
        try:
            rank = sorted_genes.index(gene_id) + 1
        except ValueError:
            rank = None
        return rank, sc

    rows = []
    for p in injected_pairs:
        sample_id = p["sample_id"]
        gene_id = p["gene_id"]
        base_rank, base_sc = _rank_and_score(sample_id, gene_id, base_ranked)
        spike_rank, spike_sc = _rank_and_score(sample_id, gene_id, spike_ranked)
        base_present = base_rank is not None
        spike_present = spike_rank is not None
        rank_improvement = (base_rank - spike_rank) if (base_rank is not None and spike_rank is not None) else 0
        score_gain = spike_sc - base_sc if (base_present and spike_present) else 0.0
        rows.append({
            "sample_id": sample_id,
            "gene": gene_id,
            "base_present": base_present,
            "spike_present": spike_present,
            "base_rank": base_rank if base_rank is not None else 0,
            "spike_rank": spike_rank if spike_rank is not None else 0,
            "base_anomaly_score": base_sc,
            "spike_anomaly_score": spike_sc,
            "base_is_significant": base_sig.get((sample_id, gene_id), False),
            "spike_is_significant": spike_sig.get((sample_id, gene_id), False),
            "rank_improvement": rank_improvement,
            "score_gain": score_gain,
        })

    spike_recovery = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spike_recovery.to_csv(args.output_dir / "spike_recovery.tsv", sep="\t", index=False)

    n_targets = len(injected_pairs)
    n_scored_before = sum(1 for r in rows if r["base_present"])
    n_scored_after = sum(1 for r in rows if r["spike_present"])
    rank_improvements = [r["rank_improvement"] for r in rows if r["base_present"] and r["spike_present"]]
    score_gains = [r["score_gain"] for r in rows if r["base_present"] and r["spike_present"]]
    top100_before = sum(1 for r in rows if r["base_present"] and r["base_rank"] is not None and r["base_rank"] <= 100)
    top100_after = sum(1 for r in rows if r["spike_present"] and r["spike_rank"] is not None and r["spike_rank"] <= 100)
    sig_before = sum(1 for r in rows if r["base_is_significant"])
    sig_after = sum(1 for r in rows if r["spike_is_significant"])

    summary = {
        "spike_target_gene_count": n_targets,
        "spike_targets_scored_before": n_scored_before,
        "spike_targets_scored_after": n_scored_after,
        "spike_rank_improvement_median": float(np.median(rank_improvements)) if rank_improvements else 0,
        "spike_rank_improvement_mean": float(np.mean(rank_improvements)) if rank_improvements else 0,
        "spike_score_gain_median": float(np.median(score_gains)) if score_gains else 0,
        "spike_score_gain_mean": float(np.mean(score_gains)) if score_gains else 0,
        "spike_targets_top100_before": top100_before,
        "spike_targets_top100_after": top100_after,
        "spike_targets_significant_before": sig_before,
        "spike_targets_significant_after": sig_after,
    }

    ground_truth_flat = ground_truth_mask.ravel()
    score_mat = np.zeros((len(sample_ids), len(gene_ids)), dtype=np.float32)
    p_adj_mat = np.ones((len(sample_ids), len(gene_ids)), dtype=np.float32)
    for sample_id, scores in spike_ranked.items():
        si = sample_idx.get(sample_id)
        if si is None:
            continue
        for gene_id, sc in scores.items():
            gi = gene_idx.get(gene_id)
            if gi is not None:
                score_mat[si, gi] = sc
    for (sample_id, gene_id), p_val in spike_p_adj.items():
        si = sample_idx.get(sample_id)
        gi = gene_idx.get(gene_id)
        if si is not None and gi is not None:
            p_adj_mat[si, gi] = p_val
    score_flat = score_mat.ravel()
    p_adj_flat = p_adj_mat.ravel()

    from bulkformer_dx.benchmark.metrics import benchmark_metrics

    bm = benchmark_metrics(
        ground_truth_flat,
        score_flat,
        p_adj=p_adj_flat,
        p_raw=None,
        k=100,
    )
    summary["auroc"] = bm["auroc"]
    summary["auprc"] = bm["auprc"]
    summary["precision_at_k_100"] = bm["precision_at_k"]
    summary["recall_at_fdr_05"] = bm.get("recall_at_fdr_05", 0)
    summary["recall_at_fdr_10"] = bm.get("recall_at_fdr_10", 0)

    if args.base_calibrated.exists():
        base_summary_path = args.base_calibrated / "calibration_summary.tsv"
        if base_summary_path.exists():
            base_cal = pd.read_csv(base_summary_path, sep="\t")
            abs_col = "absolute_significant_gene_count_by_alpha"
            emp_col = "significant_gene_count_by_0_05"
            if abs_col in base_cal.columns:
                summary["empirical_significant_gene_count_mean"] = float(
                    base_cal[emp_col].mean()
                ) if emp_col in base_cal.columns else 0
                summary["absolute_significant_gene_count_mean"] = float(base_cal[abs_col].mean())
                summary["absolute_significant_gene_count_median"] = float(base_cal[abs_col].median())

    with (args.output_dir / "anomaly_qc_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    if args.figures_dir.exists() or True:
        args.figures_dir.mkdir(parents=True, exist_ok=True)
        try:
            from bulkformer_dx.benchmark.plots import (
                plot_pr_curve,
                plot_pvalue_histogram,
                plot_pvalue_qq,
            )
            plot_pr_curve(ground_truth_flat, score_flat, args.figures_dir / "spike_pr_curve.png")
            null_mask = ~ground_truth_flat
            if null_mask.sum() > 0:
                p_null = p_adj_flat[null_mask]
                plot_pvalue_histogram(p_null, args.figures_dir / "spike_pvalue_hist_null.png")
            pos_mask = ground_truth_flat
            if pos_mask.sum() > 0:
                p_pos = p_adj_flat[pos_mask]
                plot_pvalue_histogram(p_pos, args.figures_dir / "spike_pvalue_hist_injected.png")
            plot_pvalue_qq(p_adj_flat, args.figures_dir / "spike_pvalue_qq.png")
        except ImportError as e:
            print(f"Could not generate benchmark figures: {e}", file=sys.stderr)

    print(f"Spike recovery: {args.output_dir / 'spike_recovery.tsv'}")
    print(f"Benchmark metrics: AUROC={summary['auroc']:.4f} AUPRC={summary['auprc']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
