# Diagnostics Implementation Summary

This document provides one place to review the implementation and repository modifications that have
been completed for the `bulkformer_dx` diagnostics toolkit.

## Scope

The completed work adds a repo-local Python package, command-line entrypoints, workflow-specific
modules, focused tests, and supporting documentation for:

- RNA preprocessing
- BulkFormer asset/model loading
- RNA anomaly scoring
- frozen-backbone anomaly head training
- cohort calibration and N=1 absolute outlier calling
- tissue validation
- frozen-backbone proteomics prediction
- environment/setup and rollout documentation

## Package And CLI Additions

The repo now includes a dedicated `bulkformer_dx` package with:

- `bulkformer_dx/cli.py` as the top-level CLI entrypoint
- command groups for `preprocess`, `anomaly`, `tissue`, and `proteomics`
- workflow-specific modules under `bulkformer_dx/anomaly/` plus top-level tissue and proteomics
  modules
- module-level `run()` handlers so every workflow can be executed from `python -m bulkformer_dx.cli`

This turns the earlier notebook-oriented and script-oriented pieces into a reusable toolkit with
stable input/output contracts.

## Implemented Workflows

### Preprocessing

`bulkformer_dx/preprocess.py` now implements the RNA preprocessing contract used by all downstream
workflows:

- loads counts as either `genes-by-samples` or `samples-by-genes`
- normalizes Ensembl IDs by trimming version suffixes
- resolves gene lengths from either an explicit length column or `start` / `end`
- converts counts to TPM and then to `log1p(TPM)`
- aligns the matrix to the BulkFormer gene panel in model order
- fills missing genes with the BulkFormer-compatible mask value `-10`
- exports `valid_gene_mask.tsv` and `preprocess_report.json`

### BulkFormer Model Loading

`bulkformer_dx/bulkformer_model.py` now provides reusable model loading and feature extraction
helpers:

- variant-aware checkpoint resolution using `model/config.py`
- auto-discovery that prefers a local `BulkFormer_37M.pt` when available
- clear missing-asset errors that point users to `model/README.md` and `data/README.md`
- checkpoint state-dict cleanup for wrapper prefixes such as `module.` and `model.`
- graph construction for the original BulkFormer backbone via `torch_geometric`
- helpers for expression prediction, per-gene embeddings, and per-sample embeddings

The toolkit reuses the original repo backbone code instead of forking it.

### RNA Anomaly Scoring

`bulkformer_dx/anomaly/scoring.py` implements the ranking workflow:

- Monte Carlo masking over valid genes only
- BulkFormer-compatible masking semantics with `-10`
- residual aggregation across masked passes
- per-sample ranked gene outputs
- cohort-level summary and gene-QC outputs

The default workflow remains ranking-oriented and cohort-friendly rather than trying to make a hard
per-gene binary call at scoring time.

### Anomaly Head Training

`bulkformer_dx/anomaly/head.py` adds small frozen-backbone heads on top of BulkFormer embeddings:

- `sigma_nll` as the recommended default objective
- `injected_outlier` as an explicitly synthetic optional mode
- frozen BulkFormer backbone during head training
- compact checkpoint and metric outputs for downstream reuse

### Cohort Calibration

`bulkformer_dx/anomaly/calibration.py` adds the calibration stage that was called out in the plan:

- empirical cohort-tail p-values per gene from ranked anomaly scores
- per-sample Benjamini-Yekutieli correction by default
- normalized absolute outlier calls using observed expression, predicted mean, and cohort-derived
  residual scale
- flattened `absolute_outliers.tsv` output for cohort review
- optional `nb_approx` TPM-derived negative-binomial approximation, clearly labeled as an
  approximation rather than raw-count OUTRIDER-style inference

The normalized absolute outlier implementation is now explicit and typed rather than being implied
through ranking metrics alone. The key additions are:

- `compute_normalized_outliers(...)` in `bulkformer_dx/anomaly/calibration.py`
- constants for the normalized path such as:
  - `DEFAULT_ALPHA = 0.05`
  - `SIGMA_EPSILON = 1e-6`
  - `expected_mu`
  - `expected_sigma`
  - `z_score`
  - `raw_p_value`
  - `by_adj_p_value`
  - `is_significant`
- an extended `CalibrationResult` dataclass that now carries:
  - calibrated per-sample ranked tables
  - the flattened `absolute_outliers` DataFrame
  - the calibration summary
  - run metadata

The mathematical logic implemented for the normalized table is:

- `z = (Y - mu) / (sigma + 1e-6)`
- `raw_p_value = 2 * norm.sf(abs(z))`
- `by_adj_p_value = BY(raw_p_value)` applied within each sample
- `is_significant = by_adj_p_value < alpha`

For the current cohort-based integration, the values are sourced as follows:

- `Y`: `observed_expression` from the ranked anomaly tables
- `mu`: `mean_predicted_expression` from the ranked anomaly tables
- `sigma`: a robust empirical residual scale estimated per gene across the cohort from
  `mean_signed_residual`, using MAD with a standard-deviation fallback

This means the implementation now supports both:

- empirical residual ranking and empirical cohort-tail calibration
- explicit normalized absolute outlier calling suitable for N=1-style interpretation once a cohort
  reference exists

The calibrated per-sample ranked tables were also extended with additive columns rather than changing
the old contract:

- `expected_mu`
- `expected_sigma`
- `z_score`
- `raw_p_value`
- `by_adj_p_value`
- `is_significant`

The top-level calibration output directory now contains:

- `ranked_genes/<sample>.tsv`
- `absolute_outliers.tsv`
- `calibration_summary.tsv`
- `calibration_run.json`

`calibration_summary.tsv` now includes normalized absolute-outlier summary fields in addition to the
older empirical ones, including:

- `absolute_significant_gene_count_by_alpha`
- `min_absolute_by_adj_p_value`

`calibration_run.json` now records:

- the selected `alpha`
- the absolute-outlier method description
- the count-space method and approximation note when `nb_approx` is enabled

The CLI surface was kept backward-compatible and only extended additively:

- `bulkformer_dx/anomaly/cli.py` now accepts `anomaly calibrate --alpha`
- the existing `score`, `head`, `train-head`, and `calibrate` subcommands remain unchanged in name
- the existing ranking outputs under `ranked_genes/` remain valid for prior consumers

### Tissue Validation

`bulkformer_dx/tissue.py` now supports both training and prediction:

- BulkFormer sample embedding extraction
- optional PCA before classification
- `RandomForestClassifier` training
- serialized sklearn artifact bundles
- prediction-time contract checks so the saved gene set, aggregation mode, and BulkFormer asset
  choices stay consistent

### Proteomics

`bulkformer_dx/proteomics.py` implements the frozen-backbone RNA-to-protein workflow:

- alignment of RNA and proteomics tables by sample ID
- optional log2 transformation and optional center/scale target normalization
- shallow `linear` and `mlp` heads
- masked MSE so missing protein targets are excluded from optimization
- prediction export, residual export, ranked per-sample protein tables, and optional BY-adjusted
  protein calls
- serialized feature contracts so inference reuses the same selected genes, aggregation mode, and
  BulkFormer asset choices as training

## Documentation And Repo Modifications

The implementation work also expanded the repo documentation surface:

- top-level `README.md` now advertises the diagnostics toolkit, environment flow, and Ralph rollout
  workflow
- `docs/README.md` indexes the diagnostics docs
- `docs/bulkformer-dx/README.md` provides the toolkit overview
- workflow docs now exist for preprocessing, anomaly, tissue, and proteomics
- `docs/bulkformer-dx/architecture.md` documents module layout, data contracts, and end-to-end flow
- `docs/bulkformer-dx/cli-reference.md` documents CLI arguments and outputs
- `docs/PIPELINE_rna_anomaly.md` and `docs/PIPELINE_proteomics.md` provide end-to-end examples
- `docs/installation.md` and the platform-specific install docs document the pinned setup path

For the absolute outlier addition specifically, the docs now also capture:

- the normalized z-score formula
- the two-sided normal p-value calculation
- the rationale for using BY correction under gene-gene correlation
- the new `absolute_outliers.tsv` artifact and its schema
- the fact that the `gene` field in that flattened table uses the Ensembl-style gene identifiers from
  the BulkFormer-aligned matrix

Repo workflow/setup modifications were also added around the diagnostics rollout:

- platform-specific environment YAMLs in `envs/`
- bootstrap and verification flow for the pinned PyTorch 2.5.1 and PyG setup
- Ralph workflow assets and documentation for fresh-context rollout execution

## Validation Coverage

The implementation is backed by focused tests in `tests/`:

- `test_bulkformer_dx_cli.py`
- `test_preprocess.py`
- `test_bulkformer_model.py`
- `test_anomaly_scoring.py`
- `test_anomaly_head.py`
- `test_anomaly_calibration.py`
- `test_anomaly_synthetic.py`
- `test_tissue.py`
- `test_proteomics.py`

These tests cover CLI exposure, preprocessing math and alignment behavior, model-loading helpers,
mask semantics, anomaly calibration behavior, synthetic outlier recovery, and the tissue/proteomics
training and prediction flows.

The normalized absolute outlier implementation specifically added or extended coverage for:

- the additive calibration columns in `test_anomaly_calibration.py`
- the new `absolute_outliers.tsv` export in `test_anomaly_calibration.py`
- CLI default handling for `--alpha`
- direct single-sample use of `compute_normalized_outliers(...)`
- the synthetic high-signal test in `test_anomaly_synthetic.py`

That synthetic test validates the mathematical logic directly:

- `1 x 10,000` genes
- `50` injected outliers at `mu ± 6*sigma`
- recall requirement `>= 0.95`
- false positives `< 10`
- injected z-scores approximately `±6`

This validation was designed to confirm the math and multiple-testing logic itself, without relying
on any external OUTRIDER benchmark or external dataset.

## Current Result

The repo now contains a coherent diagnostics toolkit built around the original BulkFormer backbone,
with documented CLI workflows, reusable Python APIs, targeted tests, and supporting setup/developer
workflow documentation.

It also now contains an explicit absolute outlier interpretation layer on top of the anomaly stack,
bridging the existing sigma-style anomaly semantics to user-visible z-scores, p-values, BY-adjusted
calls, and a cohort-level `absolute_outliers.tsv` artifact.
