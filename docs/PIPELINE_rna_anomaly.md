# RNA Anomaly Pipeline

End-to-end RNA anomaly workflow:

## 1. Preprocess Counts

```bash
python -m bulkformer_dx.cli preprocess \
  --counts /path/to/raw_counts.tsv \
  --annotation /path/to/gene_annotation_v29.tsv \
  --output-dir preprocess_out
```

This produces:

- `tpm.tsv`
- `log1p_tpm.tsv`
- `aligned_log1p_tpm.tsv`
- `valid_gene_mask.tsv`
- `preprocess_report.json`

The TPM formula implemented in code and docs is:

`rate = counts / (gene_length_bp / 1000)`, `TPM = rate / sum(rate) * 1e6`, `log1p(TPM) = ln(TPM + 1)`.

## 2. Score Gene-Level Anomalies

```bash
python -m bulkformer_dx.cli anomaly score \
  --input preprocess_out/aligned_log1p_tpm.tsv \
  --valid-gene-mask preprocess_out/valid_gene_mask.tsv \
  --output-dir anomaly_out \
  --variant 37M \
  --device cpu
```

## 3. Train The Optional Small Head

Recommended baseline:

```bash
python -m bulkformer_dx.cli anomaly train-head \
  --input preprocess_out/aligned_log1p_tpm.tsv \
  --valid-gene-mask preprocess_out/valid_gene_mask.tsv \
  --output-dir anomaly_head_out \
  --mode sigma_nll \
  --variant 37M
```

## 4. Calibrate Cohort Calls

```bash
python -m bulkformer_dx.cli anomaly calibrate \
  --scores anomaly_out \
  --output-dir anomaly_calibrated
```

The default path is empirical + BY correction. `--count-space-method nb_approx` adds the documented
TPM-derived NB approximation.

## 5. Optional Tissue Validation

```bash
python -m bulkformer_dx.cli tissue train \
  --input preprocess_out/aligned_log1p_tpm.tsv \
  --labels tissue_labels.tsv \
  --output-dir tissue_train \
  --variant 37M
```

See `docs/bulkformer-dx/preprocess.md`, `docs/bulkformer-dx/anomaly.md`, and
`docs/bulkformer-dx/tissue.md` for workflow-specific details.
