# BulkFormer-DX Anomaly Detection Design Summary

**Date**: 2026-03-12

This document summarizes the high-level design for the BulkFormer-DX 2.0 anomaly/outlier detection framework. For full specification, see [improving_anomalyt_detection_plan.md](../bulkformer-dx/improving_anomalyt_detection_plan.md).

## Goals

- Support multiple scoring/testing methods behind a unified `MethodConfig` schema
- MC residual baseline, TabPFN-style pseudo-likelihood, OUTRIDER-style NB test in count space
- Global vs embedding-kNN local cohort calibration
- Benchmark harness with calibration diagnostics

## Architecture Overview

```
preprocess → AlignedExpressionBundle
     ↓
predict_mean / mc_predict → ModelPredictionBundle
     ↓
scoring (residual | pseudolikelihood) → ranked gene tables
     ↓
calibration (p-values, BY) → GeneOutlierTable
     ↓
benchmark (metrics, plots) → leaderboard
```

## Core Contracts

- **AlignedExpressionBundle**: expr_space, Y_obs, valid_mask, gene_ids, sample_ids, optional counts/gene_length_kb/tpm_scaling_S
- **ModelPredictionBundle**: y_hat, sigma_hat, embedding, mc_samples
- **MethodConfig**: method_id, space, cohort_mode, uncertainty_source, distribution_family, test_type, multiple_testing, runtime params

## Method Families

| Family | Space | Test | Notes |
|--------|-------|------|-------|
| Residual | log1p_tpm | empirical_tail | MC masking, mean abs residual |
| NLL | log1p_tpm | pseudo_likelihood | MC masked log-prob (Gaussian/Student-t) |
| NB OUTRIDER | counts | outrider_nb_2s | Discrete-safe two-sided NB p-value |
| Z-score | log1p_tpm | zscore_2s | Gaussian or Student-t tails |

## References

- [improving_anomalyt_detection_plan.md](../bulkformer-dx/improving_anomalyt_detection_plan.md) — Full implementation plan
- [deep-research-report.md](../bulkformer-dx/deep-research-report.md) — Calibration analysis and fixes
