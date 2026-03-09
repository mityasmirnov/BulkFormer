# Proteomics Workflow

The `proteomics` workflow freezes BulkFormer, extracts one transcriptome embedding per sample, and
fits a shallow protein head against a sample-by-protein intensity table.

Supported head types:

- `linear`: one linear projection from transcriptome embedding to proteins.
- `mlp`: a two-layer MLP with one GELU hidden layer.

The loss is masked MSE, so missing protein measurements stay excluded from optimization and from
residual summaries.

## Inputs

- `--input`: BulkFormer-aligned RNA matrix such as `aligned_log1p_tpm.tsv`.
- `--proteomics`: sample-by-protein table whose first column is `sample_id`.
- Optional `--valid-gene-mask`: restrict BulkFormer aggregation to genes observed in the RNA cohort.
- BulkFormer checkpoint/assets: resolved the same way as the RNA anomaly workflow.

Proteomics target space options:

- `--already-log2`: declare the proteomics table is already in log2 intensity space.
- `--log2-transform`: log2-transform strictly positive intensities before fitting.
- `--center-scale`: z-score each protein using training-set statistics before fitting; predictions are
  written back in the final log2 space.

## Train

```bash
python -m bulkformer_dx.cli proteomics train \
  --input preprocess_out/aligned_log1p_tpm.tsv \
  --proteomics fibroblast_proteomics.tsv \
  --output-dir proteomics_train \
  --variant 37M \
  --valid-gene-mask preprocess_out/valid_gene_mask.tsv \
  --already-log2 \
  --head-type mlp \
  --hidden-dim 256 \
  --device cpu
```

Training writes:

- `protein_head.pt`: serialized head weights plus feature contract.
- `predicted_proteomics.tsv`: fitted protein predictions in final log2 space.
- `observed_proteomics.tsv`: aligned observed protein table in final log2 space.
- `residuals.tsv`: observed minus predicted protein residuals.
- `ranked_proteins/<sample>.tsv`: per-sample protein rankings by absolute residual, with optional
  `p_value`, `padj`, `call`, and `rank_within_sample`.
- `proteomics_summary.tsv` and `prediction_summary.json`: compact run metrics and artifact metadata.

## Predict

```bash
python -m bulkformer_dx.cli proteomics predict \
  --input preprocess_out/aligned_log1p_tpm.tsv \
  --proteomics heldout_proteomics.tsv \
  --artifact-path proteomics_train/protein_head.pt \
  --output-dir proteomics_predict \
  --device cpu
```

Prediction reuses the saved gene set, BulkFormer contract, aggregation method, and proteomics target
transform from training. If you pass an explicit `--variant` or checkpoint path at prediction time,
it must match the training artifact.

If `--proteomics` is omitted in predict mode, the workflow still writes predicted protein values, but
residual ranking and BY-adjusted calls require observed protein intensities.

## Calibration

When observed protein intensities are available, the workflow computes:

- protein-wise robust residual centers/scales using median and MAD with a standard-deviation fallback
- two-sided residual p-values from the standardized residuals
- per-sample Benjamini-Yekutieli correction (`padj`) and binary `call` flags at `--alpha`

This matches the style of the RNA calibration workflow while staying in protein residual space.
