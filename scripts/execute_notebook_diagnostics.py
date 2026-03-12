#!/usr/bin/env python3
"""
Notebook Result Diagnostics: compare calibration methods from notebook runs.
Loads results from runs/clinical_methods_37M and produces the new suite of
diagnostic plots and summary statistics.
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bulkformer_dx.benchmark.plots import (
    plot_pvalue_qq,
    plot_expected_vs_observed_discoveries,
    plot_residual_variance_vs_mean,
    plot_stratified_pvalue_histograms,
)
from bulkformer_dx.calibration.pvalues import compute_stratified_calibration
from bulkformer_dx.benchmark.metrics import compute_calibration_diagnostics

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

METHODS = ["none", "student_t", "nb_outrider", "knn_local"]
RUNS_DIR = Path("runs/clinical_methods_37M")
OUTPUT_DIR = Path("reports/figures/notebook_integration")

def load_method_data(method: str) -> tuple[pd.DataFrame, dict] | None:
    base = RUNS_DIR / f"calibrated_{method}"
    outliers_path = base / "absolute_outliers.tsv"
    run_path = base / "calibration_run.json"
    
    if not outliers_path.exists() or not run_path.exists():
        logger.warning(f"Results for method '{method}' not found at {base}")
        return None
        
    outliers = pd.read_csv(outliers_path, sep="\t")
    with open(run_path) as f:
        meta = json.load(f)
    return outliers, meta

def run_diagnostics_for_method(method: str, outliers: pd.DataFrame, meta: dict):
    logger.info(f"Processing method: {method}")
    method_dir = OUTPUT_DIR / method
    method_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. QQ Plot with CI Bands
    # We'll use absolute_zscore/raw_p_value as the target for diagnostics
    p_raw = outliers["raw_p_value"].values
    valid_mask = np.isfinite(p_raw) & (p_raw >= 0) & (p_raw <= 1)
    p_valid = p_raw[valid_mask]
    
    if len(p_valid) == 0:
        logger.error(f"No valid p-values for {method}!")
        return

    plot_pvalue_qq(
        p_valid,
        method_dir / "qq_plot.png",
        ci_band=True,
    )
    
    # 2. Expected vs Observed Discoveries
    plot_expected_vs_observed_discoveries(
        p_valid,
        method_dir / "discoveries.png",
    )
    
    # 3. Residual Variance vs Mean
    if "observed_log1p_tpm" in outliers.columns and "expected_mu" in outliers.columns:
        outliers["residual"] = outliers["observed_log1p_tpm"] - outliers["expected_mu"]
        gene_stats = outliers.groupby("gene")["residual"].agg(["mean", "var"])
        gene_means = outliers.groupby("gene")["observed_log1p_tpm"].mean()
        
        plot_residual_variance_vs_mean(
            gene_means.values,
            gene_stats["var"].values,
            method_dir / "variance_vs_mean.png",
        )

    # 4. Stratified Analysis (by Gene Mean Expression)
    if "observed_log1p_tpm" in outliers.columns:
        # Group p-values by mean expression across cohort
        gene_means_map = outliers.groupby("gene")["observed_log1p_tpm"].mean().to_dict()
        outliers["gene_mean"] = outliers["gene"].map(gene_means_map)
        
        strata_pvals = compute_stratified_calibration(
            outliers["raw_p_value"].values,
            outliers["gene_mean"].values,
            n_bins_if_continuous=6
        )
        
        plot_stratified_pvalue_histograms(
            strata_pvals,
            method_dir / "stratified_histograms.png"
        )

    # 5. Print Summary Diagnostics
    diag = compute_calibration_diagnostics(p_valid)
    print(f"\n--- {method.upper()} Diagnostics ---")
    print(f"KS Stat: {diag.ks_stat:.4f}")
    print(f"Min P-value: {diag.min_p:.2e}")
    print("Discovery Table (Alpha | Expected | Observed | Ratio):")
    for alpha, stats in diag.discovery_table.items():
        print(f"  {alpha:<5} | {stats['expected']:>8} | {stats['observed']:>8} | {stats['ratio']:>6.2f}x")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    available_methods = []
    for m in METHODS:
        data = load_method_data(m)
        if data:
            run_diagnostics_for_method(m, *data)
            available_methods.append(m)
            
    print(f"\nDiagnostics completed for: {available_methods}")
    print(f"Figures saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
