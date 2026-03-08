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

The uncertainty head and cohort calibration subcommands are still placeholders and will be implemented in later rollout steps.
