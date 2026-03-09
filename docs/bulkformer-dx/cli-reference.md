# CLI Reference

All workflows are available through `python -m bulkformer_dx.cli`.

## Top-Level Groups

- `preprocess`
- `anomaly`
- `tissue`
- `proteomics`

## `preprocess`

Purpose: convert raw counts into BulkFormer-aligned `log1p(TPM)`.

Main arguments:

- `--counts`: raw counts table
- `--annotation`: gene annotation table
- `--output-dir`: output directory
- `--counts-orientation`: `genes-by-samples` or `samples-by-genes`
- `--bulkformer-gene-info`: alternate gene panel
- `--fill-value`: value used for missing genes after BulkFormer alignment
- `--missing-gene-length-bp`: fallback length when the annotation lacks a gene

Outputs:

- `tpm.tsv`
- `log1p_tpm.tsv`
- `aligned_log1p_tpm.tsv`
- `valid_gene_mask.tsv`
- `preprocess_report.json`

## `anomaly score`

Purpose: run Monte Carlo masking and rank contextual RNA anomalies per sample.

Main arguments:

- `--input`
- `--valid-gene-mask`
- `--output-dir`
- `--variant` or `--checkpoint-path`
- `--device`
- `--batch-size`
- `--mask-prob`
- `--mc-passes`

Outputs:

- `cohort_scores.tsv`
- `gene_qc.tsv`
- `ranked_genes/<sample>.tsv`
- `anomaly_run.json`

## `anomaly head` / `anomaly train-head`

Purpose: train a small frozen-backbone anomaly head.

Main arguments:

- `--input`
- `--valid-gene-mask`
- `--output-dir`
- `--mode`: `sigma_nll` or `injected_outlier`
- `--hidden-dim`
- `--epochs`
- `--learning-rate`
- `--weight-decay`
- `--min-sigma`
- `--injection-rate`
- `--outlier-scale`

Outputs:

- `<mode>_head.pt`
- `training_metrics.json`

## `anomaly calibrate`

Purpose: produce cohort-based empirical p-values and BY-adjusted calls.

Main arguments:

- `--scores`: anomaly output directory or `ranked_genes/`
- `--output-dir`
- `--count-space-method`: `none` or `nb_approx`

Outputs:

- `ranked_genes/<sample>.tsv`
- `calibration_summary.tsv`
- `calibration_run.json`

## `tissue train`

Purpose: train a tissue classifier from BulkFormer sample embeddings.

Main arguments:

- `--input`
- `--labels`
- `--output-dir`
- `--valid-gene-mask`
- `--aggregation`
- `--pca-components`
- `--n-estimators`
- `--max-depth`

Outputs:

- `tissue_model.joblib`
- `training_summary.json`

## `tissue predict`

Purpose: apply a saved tissue classifier artifact to new samples.

Main arguments:

- `--input`
- `--artifact-path`
- `--output-dir`
- `--valid-gene-mask`

Outputs:

- `tissue_predictions.tsv`
- `prediction_summary.json`

## `proteomics train`

Purpose: train a frozen-backbone RNA-to-protein head.

Main arguments:

- `--input`
- `--proteomics`
- `--output-dir`
- `--valid-gene-mask`
- `--aggregation`
- `--head-type`: `linear` or `mlp`
- `--hidden-dim`
- `--epochs`
- `--val-fraction`
- `--patience`
- `--log2-transform` or `--already-log2`
- `--center-scale`
- `--alpha`

Outputs:

- `protein_head.pt`
- `predicted_proteomics.tsv`
- `observed_proteomics.tsv`
- `residuals.tsv`
- `q_values.tsv`
- `ranked_proteins/<sample>.tsv`
- `proteomics_summary.tsv`
- `prediction_summary.json`

## `proteomics predict`

Purpose: apply a saved protein head artifact to new RNA samples.

Main arguments:

- `--input`
- `--artifact-path`
- optional `--proteomics` for residual-based ranking on held-out matched samples
- `--output-dir`

Outputs:

- `predicted_proteomics.tsv`
- optional residual/ranking outputs when observed proteomics are supplied
