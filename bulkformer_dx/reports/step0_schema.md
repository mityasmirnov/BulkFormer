# Step 0: Core Schemas

## Summary

Schemas define shared data contracts for the anomaly and benchmarking pipelines. All downstream modules accept/return these schema objects instead of ad-hoc dicts.

## Schemas

### AlignedExpressionBundle

| Field | Type | Description |
|-------|------|-------------|
| expr_space | str | "log1p_tpm", "tpm", or "counts" |
| Y_obs | (n_samples, n_genes) float32 | Observed expression |
| valid_mask | (n_samples, n_genes) bool | Gene presence and usability |
| gene_ids | list[str] | Ordered gene identifiers |
| sample_ids | list[str] | Ordered sample identifiers |
| counts | optional | Raw counts for NB tests |
| gene_length_kb | optional | Gene lengths in kb |
| tpm_scaling_S | optional | S_j per sample (TPM↔counts mapping) |

### ModelPredictionBundle

| Field | Type | Description |
|-------|------|-------------|
| y_hat | (n_samples, n_genes) float32 | Predicted mean |
| sigma_hat | optional | Predicted std (sigma head or derived) |
| embedding | optional | (n_samples, d) for kNN cohort |
| mc_samples | optional | (n_mc, n_samples, n_genes) from mc_predict |

### MethodConfig

| Field | Default | Description |
|-------|---------|-------------|
| method_id | required | Unique identifier |
| space | "log1p_tpm" | "log1p_tpm" or "counts" |
| cohort_mode | "global" | "global" or "knn_local" |
| knn_k | 50 | Neighbors for kNN |
| uncertainty_source | "cohort_sigma" | cohort_sigma, sigma_head, mc_variance, nb_dispersion |
| distribution_family | "gaussian" | gaussian, student_t, negative_binomial |
| test_type | "zscore_2s" | outrider_nb_2s, zscore_2s, empirical_tail, pseudo_likelihood |
| multiple_testing | "BY" | BY, BH, none |
| mc_passes | 16 | MC masking passes |
| mask_rate | 0.15 | Mask probability |

### GeneOutlierRow

sample_id, gene_id, y_obs, y_hat, residual, score_gene, p_raw, p_adj, direction, method_id, diagnostics_json

### SampleOutlierRow

sample_id, score_sample, cohort_mode, method_id

## Example Artifact Tree

```
output_dir/
├── aligned_log1p_tpm.tsv
├── aligned_counts.tsv
├── aligned_tpm.tsv
├── gene_lengths_aligned.tsv
├── sample_scaling.tsv
├── valid_gene_mask.tsv
├── ranked_genes/
│   └── sample_001.tsv
├── benchmark_results.parquet
└── benchmark_summary.json
```

## IO Helpers

`bulkformer_dx.io.read_write` provides:

- `load_tsv`, `write_tsv`, `load_parquet`, `write_parquet`
- `load_config_dict(path)` — YAML/JSON to dict
- `load_method_config(path)` — Validated MethodConfig
- `method_config_from_dict(data)` — MethodConfig from dict
