#!/usr/bin/env python3
"""
Calibration analysis: outliers per sample, per-gene distributions with outliers highlighted.

Usage:
  python scripts/calibration_analysis.py --variant 37M [--variant 147M] [--n-genes 20]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_calibration(variant: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load calibration_summary, absolute_outliers, and calibration_run.json."""
    base = Path(f"runs/clinical_anomaly_calibrated_{variant}")
    if not base.exists():
        raise FileNotFoundError(f"Calibration dir not found: {base}")
    summary = pd.read_csv(base / "calibration_summary.tsv", sep="\t")
    outliers = pd.read_csv(base / "absolute_outliers.tsv", sep="\t")
    with open(base / "calibration_run.json") as f:
        meta = json.load(f)
    return summary, outliers, meta


def load_ranked_calibrated(variant: str) -> dict[str, pd.DataFrame]:
    """Load calibrated ranked_genes tables (with empirical_p_value, anomaly_score)."""
    ranked_dir = Path(f"runs/clinical_anomaly_calibrated_{variant}") / "ranked_genes"
    if not ranked_dir.exists():
        return {}
    return {p.stem: pd.read_csv(p, sep="\t") for p in sorted(ranked_dir.glob("*.tsv"))}


def compute_outliers_per_sample(summary: pd.DataFrame) -> pd.DataFrame:
    """Compute outlier counts per sample for all approaches from calibration_summary."""
    return summary[
        [
            "sample_id",
            "tested_genes",
            "significant_gene_count_by_0_05",
            "absolute_significant_gene_count_by_alpha",
        ]
    ].rename(
        columns={
            "significant_gene_count_by_0_05": "empirical_outliers_alpha_005",
            "absolute_significant_gene_count_by_alpha": "absolute_outliers",
        }
    )


def _top_genes_by_outliers(outliers: pd.DataFrame, n_genes: int) -> list[str]:
    """Return gene IDs with most absolute outliers."""
    sig = outliers[outliers["is_significant"]]
    gene_counts = sig.groupby("gene").size().sort_values(ascending=False)
    return gene_counts.head(n_genes).index.tolist()


def plot_gene_residual_histograms(
    outliers: pd.DataFrame,
    *,
    n_genes: int = 20,
    alpha: float = 0.01,
    output_dir: Path,
) -> tuple[list[Path], list[str]]:
    """
    For each of n_genes genes, plot cohort residual distribution with outliers highlighted.
    Picks genes with most outliers to show where calibration may be problematic.
    Returns (paths, top_genes) for reuse in empirical histograms.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    outliers = outliers.copy()
    outliers["residual"] = outliers["observed_log1p_tpm"] - outliers["expected_mu"]

    top_genes = _top_genes_by_outliers(outliers, n_genes)

    paths = []
    for gene_id in top_genes:
        sub = outliers[outliers["gene"] == gene_id]
        res = sub["residual"].values
        sig_mask = sub["is_significant"].values
        sigma = sub["expected_sigma"].iloc[0]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(
            res[~sig_mask],
            bins=min(50, max(10, len(res) // 5)),
            alpha=0.7,
            color="steelblue",
            edgecolor="black",
            label="non-outlier",
        )
        if sig_mask.any():
            ax.hist(
                res[sig_mask],
                bins=min(30, max(5, sig_mask.sum() // 2)),
                alpha=0.8,
                color="coral",
                edgecolor="black",
                label="outlier (absolute)",
            )
        ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
        ax.axvline(-2 * sigma, color="orange", linestyle=":", alpha=0.7, label=r"±2$\sigma$")
        ax.axvline(2 * sigma, color="orange", linestyle=":", alpha=0.7)
        ax.set_xlabel("Residual (observed - expected)")
        ax.set_ylabel("Count")
        ax.set_title(f"{gene_id} | σ={sigma:.3f} | {sig_mask.sum()} outliers")
        ax.legend()
        fig.tight_layout()
        out_path = output_dir / f"gene_residual_{gene_id}.png"
        fig.savefig(out_path, dpi=150)
        plt.close()
        paths.append(out_path)

    return paths, top_genes


def plot_empirical_anomaly_histograms(
    ranked: dict[str, pd.DataFrame],
    top_genes: list[str],
    *,
    output_dir: Path,
) -> list[Path]:
    """
    For each gene, plot cohort anomaly_score distribution with empirical outliers highlighted.
    Empirical outlier = empirical_p_value < 0.05 (before BY correction).
    """
    if not ranked or "empirical_p_value" not in next(iter(ranked.values())).columns:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    # Build gene -> (scores, empirical_outlier_mask)
    gene_data: dict[str, tuple[list[float], list[bool]]] = defaultdict(lambda: ([], []))
    for sample_id, tbl in ranked.items():
        for _, row in tbl.iterrows():
            g = str(row["ensg_id"])
            if g not in top_genes:
                continue
            gene_data[g][0].append(float(row["anomaly_score"]))
            gene_data[g][1].append(float(row["empirical_p_value"]) < 0.05)
    paths = []
    for gene_id in top_genes:
        if gene_id not in gene_data:
            continue
        scores = np.array(gene_data[gene_id][0])
        is_out = np.array(gene_data[gene_id][1])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(
            scores[~is_out],
            bins=min(50, max(10, len(scores) // 5)),
            alpha=0.7,
            color="steelblue",
            edgecolor="black",
            label="non-outlier",
        )
        if is_out.any():
            ax.hist(
                scores[is_out],
                bins=min(30, max(5, int(is_out.sum()) // 2 or 1)),
                alpha=0.8,
                color="coral",
                edgecolor="black",
                label="empirical p<0.05",
            )
        ax.set_xlabel("Anomaly score")
        ax.set_ylabel("Count")
        ax.set_title(f"{gene_id} (empirical) | {int(is_out.sum())} with p<0.05")
        ax.legend()
        fig.tight_layout()
        out_path = output_dir / f"gene_anomaly_empirical_{gene_id}.png"
        fig.savefig(out_path, dpi=150)
        plt.close()
        paths.append(out_path)
    return paths


def plot_zscore_distribution(outliers: pd.DataFrame, output_path: Path) -> None:
    """Plot z-score distribution and highlight significance threshold."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    z = outliers["z_score"].values
    sig = outliers["is_significant"].values

    axes[0].hist(z[~sig], bins=80, alpha=0.7, color="steelblue", label="non-outlier")
    if sig.any():
        axes[0].hist(z[sig], bins=50, alpha=0.8, color="coral", label="outlier")
    axes[0].axvline(-2, color="orange", linestyle="--", label="z=±2")
    axes[0].axvline(2, color="orange", linestyle="--")
    axes[0].set_xlabel("z-score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Z-score distribution (all genes)")
    axes[0].legend()

    axes[1].hist(np.clip(z, -5, 5), bins=100, alpha=0.7, color="steelblue")
    axes[1].axvline(-2, color="orange", linestyle="--")
    axes[1].axvline(2, color="orange", linestyle="--")
    axes[1].set_xlabel("z-score (clipped ±5)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Z-score (clipped)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibration analysis and distribution plots.")
    parser.add_argument("--variant", default="147M", help="Model variant (37M or 147M).")
    parser.add_argument("--n-genes", type=int, default=20, help="Number of genes for histograms.")
    parser.add_argument("--output-dir", default="reports/figures", help="Output directory.")
    parser.add_argument("--empirical-histograms", action="store_true", help="Also plot empirical anomaly_score histograms (slower).")
    args = parser.parse_args()

    summary, outliers, meta = load_calibration(args.variant)
    alpha = meta.get("alpha", 0.05)

    per_sample = compute_outliers_per_sample(summary)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sample.to_csv(out_dir / f"calibration_outliers_per_sample_{args.variant}.tsv", sep="\t", index=False)
    print(f"Wrote {out_dir / f'calibration_outliers_per_sample_{args.variant}.tsv'}")

    # Gene residual histograms (absolute approach)
    hist_dir = out_dir / f"calibration_gene_histograms_{args.variant}"
    paths, top_genes = plot_gene_residual_histograms(
        outliers,
        n_genes=args.n_genes,
        alpha=alpha,
        output_dir=hist_dir,
    )
    print(f"Wrote {len(paths)} gene residual histograms to {hist_dir}")

    # Empirical anomaly_score histograms (optional, slower)
    if args.empirical_histograms:
        ranked = load_ranked_calibrated(args.variant)
        emp_dir = out_dir / f"calibration_gene_histograms_empirical_{args.variant}"
        emp_paths = plot_empirical_anomaly_histograms(ranked, top_genes, output_dir=emp_dir)
        if emp_paths:
            print(f"Wrote {len(emp_paths)} empirical anomaly histograms to {emp_dir}")

    # Z-score distribution
    plot_zscore_distribution(outliers, out_dir / f"calibration_zscore_dist_{args.variant}.png")
    print(f"Wrote z-score distribution to {out_dir}")

    # Summary stats
    summary_path = out_dir / f"calibration_summary_stats_{args.variant}.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "variant": args.variant,
                "alpha": alpha,
                "samples": int(len(per_sample)),
                "mean_absolute_outliers_per_sample": float(per_sample["absolute_outliers"].mean()),
                "median_absolute_outliers_per_sample": float(per_sample["absolute_outliers"].median()),
                "mean_empirical_005_per_sample": float(per_sample["empirical_outliers_alpha_005"].mean()),
            },
            f,
            indent=2,
        )
    print(f"Wrote {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
