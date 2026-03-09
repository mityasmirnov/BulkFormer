# Anomaly Workflow

The `anomaly score` workflow performs Monte Carlo masking over BulkFormer-valid genes only, runs BulkFormer expression prediction on each masked pass, and aggregates residuals into ranked per-sample anomaly tables plus cohort-level QC summaries.

## Inputs

- `--input`: BulkFormer-aligned sample-by-gene expression table, typically `aligned_log1p_tpm.tsv` from preprocessing.
- `--valid-gene-mask`: `valid_gene_mask.tsv` from preprocessing.
- BulkFormer assets: the local checkpoint plus graph, graph weights, gene embeddings, and gene info table. The existing model loader defaults are used unless you override them on the CLI.

## Outputs

- `cohort_scores.tsv`: per-sample summary metrics such as mean absolute residual, RMSE, masked observation count, and gene coverage.
- `gene_qc.tsv`: cohort-wide gene-level mask coverage and residual summaries.
- `ranked_genes/<sample>.tsv`: per-sample ranked anomaly scores for genes that were actually masked at least once.
- `anomaly_run.json`: run metadata including `mc_passes`, `mask_prob`, and mask coverage details.

## Example

```bash
python -m bulkformer_dx.cli anomaly score \
  --input output/aligned_log1p_tpm.tsv \
  --valid-gene-mask output/valid_gene_mask.tsv \
  --output-dir output/anomaly \
  --variant 37M \
  --device cpu
```

## Anomaly Head

The `anomaly head` workflow trains a small MLP on top of frozen BulkFormer gene embeddings.
The recommended default is `--mode sigma_nll`, which learns a Gaussian mean and sigma for
per-gene residuals using an NLL objective. An optional `--mode injected_outlier` path trains a
binary classifier against synthetic gene-level perturbations for controlled experiments.

Example:

```bash
python -m bulkformer_dx.cli anomaly head \
  --input output/aligned_log1p_tpm.tsv \
  --valid-gene-mask output/valid_gene_mask.tsv \
  --output-dir output/anomaly_head \
  --mode sigma_nll \
  --variant 37M \
  --device cpu
```

Outputs:

- `<mode>_head.pt`: trained head checkpoint plus basic metadata.
- `training_metrics.json`: compact training metrics for the selected objective.

The underlying BulkFormer backbone stays frozen during head training.

## Calibration

The `anomaly calibrate` workflow calibrates ranked gene residuals against the cohort-wide
distribution for each gene, then applies Benjamini-Yekutieli correction within each sample by
default. This keeps the primary path empirical and non-parametric while still giving each sample a
multiple-testing-aware ranking.

Inputs:

- `--scores`: path to the anomaly scoring output directory or its `ranked_genes/` subdirectory.
- `--output-dir`: directory for calibrated ranked tables and cohort summaries.
- `--count-space-method`: optional count-space support path. `none` keeps the workflow purely
  empirical. `nb_approx` adds a TPM-derived negative-binomial approximation and is explicitly
  labeled as approximate rather than raw-count inference.

Outputs:

- `ranked_genes/<sample>.tsv`: calibrated ranked tables with `empirical_p_value` and `by_q_value`.
- `calibration_summary.tsv`: per-sample summary including minimum empirical/BY values and the count
  of BY-significant genes.
- `calibration_run.json`: run metadata plus an approximation note when `nb_approx` is enabled.

Example:

```bash
python -m bulkformer_dx.cli anomaly calibrate \
  --scores output/anomaly \
  --output-dir output/anomaly_calibrated \
  --count-space-method nb_approx
```

The `nb_approx` path is intended only as a count-space ranking aid after `log1p(TPM)` conversion.
It does not reproduce OUTRIDER and should be interpreted as an approximation.
