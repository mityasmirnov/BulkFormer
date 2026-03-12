# Step One: Benchmark Scaffold

## Overview

The benchmark harness provides a reproducible evaluation scaffold for comparing anomaly scoring methods.

## Pipeline Diagram

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Synthetic  │     │   Outlier     │     │   Scoring   │
│   Dataset   │────▶│   Injection   │────▶│   Methods   │
└─────────────┘     └──────────────┘     └─────────────┘
                                               │
                                               ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Figures   │◀────│   Metrics    │◀────│  Ground     │
│  & Reports │     │   (AUROC…)   │     │  Truth      │
└─────────────┘     └──────────────┘     └─────────────┘
```

## Method Configs (Planned)

| method_id | test_type | space |
|-----------|-----------|-------|
| smoke | smoke | log1p_tpm |
| residual | zscore_2s | log1p_tpm |
| nll | pseudo_likelihood | log1p_tpm |
| nb_outrider | outrider_nb_2s | counts |

## Smoke Test Outputs

```
output_dir/
├── benchmark_summary.json    # metrics (auroc, auprc, precision_at_k, recall_at_fdr_*, ks_uniform)
├── benchmark_results.parquet # per (sample, gene) scores, p_raw, p_adj, ground_truth
└── benchmark_figures/
    └── smoke_pvalue_hist.png
```

## CLI

```bash
# Single run (smoke)
bulkformer-dx benchmark run --output-dir /tmp/bm --n-samples 30 --n-genes 200

# Grid run from config
bulkformer-dx benchmark grid-run --config config.yaml --output-dir /tmp/bm_grid
```

## Config Format (YAML/JSON)

```yaml
dataset:
  type: synthetic
  n_samples: 30
  n_genes: 200
  n_inject: 15
methods:
  - method_id: smoke
    seed: 0
  - method_id: residual
    seed: 0
```

## Metrics

- AUROC, AUPRC
- precision@k
- recall@FDR (0.05, 0.1)
- KS vs Uniform (p-value calibration)
