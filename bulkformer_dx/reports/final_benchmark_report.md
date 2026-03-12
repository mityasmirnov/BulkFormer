# Final Benchmark Report

## Overview

This report summarizes the benchmark harness and method comparison framework for BulkFormer-DX anomaly detection.

## Leaderboard (Template)

| Method | AUPRC | AUROC | Precision@100 | Recall@FDR 0.05 |
|--------|-------|-------|----------------|-----------------|
| Residual | — | — | — | — |
| NLL | — | — | — | — |
| NB OUTRIDER | — | — | — | — |
| Smoke (baseline) | — | — | — | — |

*Fill with actual runs on synthetic or real data.*

## Calibration Table (Template)

| Method | KS statistic | Inflation (p_adj < 0.05 per sample) |
|--------|--------------|-------------------------------------|
| Residual | — | — |
| NLL | — | — |
| NB OUTRIDER | — | — |

## Compute/Runtime (Template)

| Method | Seconds | GPU memory |
|--------|---------|------------|
| Residual | — | — |
| NLL | — | — |

## Figures

- PR curves for key methods
- QQ plots for p-values (Gaussian vs NB)
- Dispersion trend plot (NB method)
- Performance vs MC passes (ablation)

## Reproduction

To reproduce:

```bash
# 1. Run grid
bulkformer-dx benchmark grid-run --config benchmark_config.yaml --output-dir ./benchmark_out

# 2. Use notebook
jupyter notebook notebooks/bulkformer_dx_benchmark_summary.ipynb
```

## Artifacts

- `benchmark_results.parquet` — per (sample, gene, method) scores and p-values
- `benchmark_summary.json` — metrics per method, dataset spec
- `benchmark_figures/` — QC plots, PR curves, p-value histograms
