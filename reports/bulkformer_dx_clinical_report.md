# BulkFormer DX Clinical RNA-seq Report

## Dataset Summary

- **Checkpoint**: `model/BulkFormer_37M.pt`
- **Raw counts**: `data/clinical_rnaseq/raw_counts.tsv` (genes x samples, Ensembl IDs with versions)
- **Gene annotation**: `data/clinical_rnaseq/gene_annotation_v29.tsv` (gene_id, start, end for TPM)
- **Sample annotation**: `data/clinical_rnaseq/sample_annotation.tsv` (SAMPLE_ID, KNOWN_MUTATION, CATEGORY, TISSUE, etc.)
- **BulkFormer assets**: `data/bulkformer_gene_info.csv`, `data/G_tcga.pt`, `data/G_tcga_weight.pt`, `data/esm2_feature_concat.pt`

## Commands Run

```bash
python -m bulkformer_dx.cli preprocess \
  --counts data/clinical_rnaseq/raw_counts.tsv \
  --annotation data/clinical_rnaseq/gene_annotation_v29.tsv \
  --output-dir runs/clinical_preprocess_37M \
  --counts-orientation genes-by-samples

python -m bulkformer_dx.cli anomaly score \
  --input runs/clinical_preprocess_37M/aligned_log1p_tpm.tsv \
  --valid-gene-mask runs/clinical_preprocess_37M/valid_gene_mask.tsv \
  --output-dir runs/clinical_anomaly_score_37M \
  --variant 37M --device cuda --mask-schedule deterministic --K-target 5 --mask-prob 0.10

python -m bulkformer_dx.cli anomaly calibrate \
  --scores runs/clinical_anomaly_score_37M \
  --output-dir runs/clinical_anomaly_calibrated_37M \
  --alpha 0.05

python -m bulkformer_dx.cli embeddings extract \
  --input runs/clinical_preprocess_37M/aligned_log1p_tpm.tsv \
  --valid-gene-mask runs/clinical_preprocess_37M/valid_gene_mask.tsv \
  --output-dir runs/clinical_embeddings_37M \
  --variant 37M --device cuda
```

## Key QC Tables

### Preprocess

| Metric | Value |
| --- | ---: |
| Samples | 146 |
| Input genes (post-aggregation) | 60788 |
| Collapsed gene columns | 41 |
| BulkFormer valid gene fraction | 98.7% |
| BulkFormer valid genes | 19751 / 20010 |

### Anomaly Score

| Metric | Value |
| --- | ---: |
| MC passes | 16 |
| Mask prob | 0.15 |
| Mean cohort abs residual | 0.8603 |
| Valid genes | 19751 |

### Calibration

| Metric | Value |
| --- | ---: |
| Samples | 146 |
| Scored genes | 2668794 |
| Alpha | 0.05 |
| Count-space method | none |

#### Outliers per Sample (37M, α=0.05)

| Metric | Value |
| --- | ---: |
| Mean absolute outliers per sample | ~79 |
| Median absolute outliers per sample | ~40 |
| Mean empirical outliers (α=0.05) | 0 |

Full table: `reports/figures/calibration_outliers_per_sample_37M.tsv`

#### Distribution Figures

- **Z-score distribution**: `figures/calibration_zscore_dist_37M.png`
- **Per-gene residual histograms**: `figures/calibration_gene_histograms_37M/`

## Output Artifacts

| Path | Description |
| --- | --- |
| runs/clinical_preprocess_37M/ | tpm.tsv, aligned_log1p_tpm.tsv, valid_gene_mask.tsv |
| runs/clinical_anomaly_score_37M/ | cohort_scores.tsv, ranked_genes/ |
| runs/clinical_anomaly_calibrated_37M/ | absolute_outliers.tsv, calibration_summary.tsv |
| runs/clinical_embeddings_37M/ | sample_embeddings.tsv |

## Runtime

- **37M on CPU**: ~7 min for anomaly score (16 MC passes, batch 16).

## Outlier Inflation Note

The normalized absolute-outlier path at alpha=0.05 yields **~9,000–11,000 significant genes per sample**. Consider:
- **Stricter alpha**: `--alpha 0.01` or `0.001`
- **147M model**: May have better-calibrated residuals
- **Top-k ranking**: Focus on top-ranked genes by anomaly score
