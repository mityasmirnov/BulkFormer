# BulkFormer DX Clinical RNA-seq Report (147M)

## Overview

This report documents the clinical RNA-seq pipeline run with **BulkFormer-147M** (12 layers, 640 dim).

## Commands

Preprocess is shared with 37M. For 147M:

```bash
python -m bulkformer_dx.cli anomaly score \
  --input runs/clinical_preprocess_37M/aligned_log1p_tpm.tsv \
  --valid-gene-mask runs/clinical_preprocess_37M/valid_gene_mask.tsv \
  --output-dir runs/clinical_anomaly_score_147M \
  --variant 147M --device cuda --batch-size 4 --mask-schedule deterministic --K-target 5 --mask-prob 0.10

python -m bulkformer_dx.cli anomaly calibrate \
  --scores runs/clinical_anomaly_score_147M \
  --output-dir runs/clinical_anomaly_calibrated_147M \
  --alpha 0.01

python -m bulkformer_dx.cli embeddings extract \
  --input runs/clinical_preprocess_37M/aligned_log1p_tpm.tsv \
  --valid-gene-mask runs/clinical_preprocess_37M/valid_gene_mask.tsv \
  --output-dir runs/clinical_embeddings_147M \
  --variant 147M --device cuda --batch-size 4

python scripts/merge_clinical_annotation.py --variant 147M
```

Or: `ALPHA=0.01 bash scripts/run_clinical_147M.sh`

## Runtime Notes

- **147M on CPU**: ~45–90 min for anomaly score (8 MC passes, batch 4).
- **37M on CPU**: ~7 min for anomaly score.

## Key QC Tables (147M)

### Anomaly Score

| Metric | Value |
| --- | ---: |
| MC passes | 8 |
| Mask prob | 0.15 |
| Valid genes | 19751 |
| Samples | 146 |

### Calibration (alpha=0.01)

| Metric | Value |
| --- | ---: |
| Scored genes | 2097123 |
| Alpha | 0.01 |
| Mean absolute outliers per sample | ~39 |
| Median absolute outliers per sample | ~20 |

## Output Artifacts

| Path | Description |
| --- | --- |
| runs/clinical_anomaly_score_147M/ | cohort_scores, ranked_genes |
| runs/clinical_anomaly_calibrated_147M/ | absolute_outliers, calibration_summary |
| runs/clinical_embeddings_147M/ | sample_embeddings.tsv |

## Model Comparison

| Metric | 37M | 147M |
| --- | ---: | ---: |
| Layers | 1 | 12 |
| Hidden dim | 128 | 640 |
| Params | ~37M | ~147M |
| Anomaly score (CPU) | ~7 min | ~45–90 min |
| Alpha | 0.05 | 0.01 |
