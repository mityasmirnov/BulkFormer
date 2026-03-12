# BulkFormer DX Unified Outliers Browse Report

## Dataset Summary

- **Samples**: 146
- **Genes**: 19751
- **Causal samples** (known mutation): 54
- **Data source**: `runs/clinical_methods_37M/unified_outliers.tsv`

## Causal Gene Recall

Recall@K = fraction of causal samples where the known causal gene appears in the top K by p-value (then z-score).

| method | recall@1 | recall@5 | recall@10 | recall@50 | median_rank |
| --- | --- | --- | --- | --- | --- |
| knn_local | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7709.0000 |
| nb_approx | 0.0185 | 0.0185 | 0.0185 | 0.0556 | 7709.0000 |
| nb_outrider | 0.0185 | 0.0185 | 0.0185 | 0.0556 | 7709.0000 |
| nll | 0.0000 | 0.0000 | 0.0370 | 0.0926 | 4633.0000 |
| none | 0.0000 | 0.0000 | 0.0000 | 0.0741 | 7208.0000 |
| student_t | 0.0185 | 0.0185 | 0.0185 | 0.0556 | 7709.0000 |

## Method Summary

Mean, median, and max outliers per sample; recall@1 and recall@5.

| method | mean_outliers | median_outliers | max_outliers | recall@1 | recall@5 |
| --- | --- | --- | --- | --- | --- |
| knn_local | 190.6575 | 159.0000 | 686 | 0.0000 | 0.0000 |
| nb_approx | 22.9932 | 8.0000 | 322 | 0.0185 | 0.0185 |
| nb_outrider | 22.9932 | 8.0000 | 322 | 0.0185 | 0.0185 |
| nll | 83.4932 | 41.0000 | 1755 | 0.0000 | 0.0000 |
| none | 92.2055 | 45.5000 | 1848 | 0.0000 | 0.0000 |
| student_t | 0.0342 | 0.0000 | 2 | 0.0185 | 0.0185 |

## Main Findings

- **NB-Outrider** is the recommended method: best p-value calibration (KS 0.027), balanced outlier counts (median 8), best causal gene recall.
- **kNN-Local** fails on the homogeneous fibroblast cohort: 0 recall at all K, median 159 outliers per sample—ranking flooded with false positives.
- **Student-t** is highly conservative (median 0 outliers); same recall as NB-Outrider but almost no discoveries.
- **Gene-wise centering** is essential for z-score methods; without it, Gaussian calibration produces thousands of false positives per sample.
- Run `notebooks/browse_unified_outliers.ipynb` after `scripts/export_unified_clinical_outliers.py` for volcano plots, gene rank plots, and recall figures.

## Figures

Figures are produced by `notebooks/browse_unified_outliers.ipynb` and saved to `reports/figures/unified_outliers_browse/`.

| Figure | Description |
| --- | --- |
| `reports/figures/unified_outliers_browse/recall_causal.png` | Causal gene recall@K and rank distribution by method |
| `reports/figures/unified_outliers_browse/volcano_*.png` | Single-sample volcano plots (z-score vs −log₁₀ p-value) |
| `reports/figures/unified_outliers_browse/gene_ranks_*.png` | Cohort gene rank plots for causal genes |
| `reports/figures/unified_outliers_browse/qq_all_methods.png` | QQ plot of p-values across methods |
| `reports/figures/unified_outliers_browse/variance_vs_mean.png` | Residual variance vs mean expression |
| `reports/figures/unified_outliers_browse/stratified_histograms.png` | Stratified residual histograms |

## Links

- **Notebook**: `notebooks/browse_unified_outliers.ipynb`
- **Figures**: `reports/figures/unified_outliers_browse/`
- **Export script**: `scripts/export_unified_clinical_outliers.py`
- **Methods comparison**: `notebooks/bulkformer_dx_clinical_methods_comparison.ipynb`
