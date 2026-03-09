# Proteomics Pipeline

This pipeline trains a frozen BulkFormer RNA-to-proteomics head and ranks per-sample protein
residuals.

## 1. Preprocess RNA

```bash
python -m bulkformer_dx.cli preprocess \
  --counts /path/to/raw_counts.tsv \
  --annotation /path/to/gene_annotation_v29.tsv \
  --output-dir preprocess_out
```

## 2. Train The Protein Head

```bash
python -m bulkformer_dx.cli proteomics train \
  --input preprocess_out/aligned_log1p_tpm.tsv \
  --proteomics fibroblast_proteomics.tsv \
  --output-dir proteomics_train \
  --valid-gene-mask preprocess_out/valid_gene_mask.tsv \
  --variant 37M \
  --already-log2
```

Outputs include:

- `protein_head.pt`
- `predicted_proteomics.tsv`
- `observed_proteomics.tsv`
- `residuals.tsv`
- `ranked_proteins/<sample>.tsv`

## 3. Predict On New Samples

```bash
python -m bulkformer_dx.cli proteomics predict \
  --input preprocess_out/aligned_log1p_tpm.tsv \
  --proteomics heldout_proteomics.tsv \
  --artifact-path proteomics_train/protein_head.pt \
  --output-dir proteomics_predict
```

If observed proteomics are available, the workflow also writes residual rankings plus BY-adjusted
protein calls. If they are absent, it still writes predicted protein values.

See `docs/bulkformer-dx/proteomics.md` for the detailed CLI surface and target-transform options.
