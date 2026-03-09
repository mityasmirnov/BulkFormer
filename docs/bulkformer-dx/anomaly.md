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

The cohort calibration subcommand is still a placeholder and will be implemented in a later rollout step.
