# `bulkformer_dx` Script-by-Script Guide

This guide is a practical map of the diagnostics package so contributors can quickly understand where logic lives, which entrypoint to run, and what each script produces.

## Quick Entry Points

- CLI root: `python -m bulkformer_dx.cli --help`
- Python package entrypoint: `python -m bulkformer_dx`
- Legacy predict-only CLI: `python -m bulkformer_dx.predict_cli --help`

## Top-Level Modules

| Script | Purpose | Typical Inputs | Typical Outputs |
|---|---|---|---|
| `cli.py` | Unified command dispatcher for preprocess / anomaly / benchmark / embeddings / tissue / proteomics workflows. | CLI arguments | Exit code and workflow artifacts on disk |
| `preprocess.py` | Counts loading, orientation handling, Ensembl normalization, TPM + `log1p(TPM)`, BulkFormer gene alignment. | counts table + gene annotations | aligned matrices, masks, metadata report |
| `bulkformer_model.py` | BulkFormer asset discovery, checkpoint cleanup/loading, graph construction, embedding extraction helpers. | model/data asset paths + checkpoint variant | loaded model object and/or predictions |
| `embeddings.py` | CLI helpers for exporting sample embeddings from aligned input matrices. | aligned expression table | embedding tables + metadata |
| `tissue.py` | Tissue classifier training and prediction using BulkFormer embeddings and optional PCA/RF. | aligned expression + labels (train) | trained model artifacts + predictions |
| `proteomics.py` | Frozen-backbone RNA→protein prediction, ranking, and optional BY-adjusted calls. | aligned RNA + protein targets | trained heads, predictions, ranked protein calls |
| `predict_cli.py` | Backward-compatible CLI command family for direct prediction utilities. | CLI args | serialized predictions and diagnostics |

## Anomaly Subpackage (`anomaly/`)

| Script | What it does |
|---|---|
| `cli.py` | Registers anomaly-specific CLI commands and file contracts. |
| `scoring.py` | Monte Carlo masking anomaly scoring with per-sample ranked gene outputs + QC summaries. |
| `head.py` | Lightweight heads used for uncertainty / anomaly scoring variants. |
| `calibration.py` | Cohort calibration wrappers and adjusted p-value workflow glue. |
| `nb_test.py` | Negative-binomial based testing helpers used in calibration/evaluation. |

## Benchmark Subpackage (`benchmark/`)

| Script | What it does |
|---|---|
| `cli.py` | Exposes benchmark smoke/grid command-line workflows. |
| `datasets.py` | Synthetic cohort generators and helper data structures. |
| `inject.py` | Controlled outlier injection to build known-ground-truth anomaly labels. |
| `metrics.py` | AUROC/AUPRC/recall@FDR and related benchmark summary statistics. |
| `plots.py` | PR curves, p-value histograms, QQ plots, and benchmark visual outputs. |
| `runner.py` | End-to-end benchmark harness for smoke tests and residual/grid runs. |

## Supporting Packages

| Folder | Purpose |
|---|---|
| `io/` | File schemas, config loading, read/write helpers, and payload validation. |
| `calibration/` | Cohort selection, p-value construction, and multiple testing corrections. |
| `cohort/` | Global and kNN cohort-building strategies for calibration. |
| `scoring/` | Shared residual and pseudo-likelihood scoring engines. |
| `stats/` | Gaussian/NB/dispersion routines used by anomaly methods. |
| `model/` | Model wrappers and uncertainty-related interfaces for scoring modules. |

## Contributor Notes

- If you are adding a new workflow, update this file and `docs/bulkformer-dx/cli-reference.md` together.
- Prefer adding short, local comments where transformations are non-obvious (shape assumptions, masking rules, calibration contracts).
- Keep I/O contracts explicit in docstrings (`sample x gene` vs `gene x sample`) to prevent silent shape bugs.
