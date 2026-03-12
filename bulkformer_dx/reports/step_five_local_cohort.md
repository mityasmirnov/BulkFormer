# Step 5: Local Cohort Calibration

## Overview

Implemented centralized p-value and multiple-testing logic and global vs kNN local cohort calibration.

## Implemented Components

### 1. `bulkformer_dx/calibration/pvalues.py`

- **`empirical_tail_pvalue(distribution, observed_value, upper_tail=True)`**: Leave-one-out empirical p-value against cohort background with +1 pseudo-count.
- **`zscore_two_sided_pvalue(z_scores, use_student_t, student_t_df)`**: Two-sided p-values from z-scores (Gaussian or Student-t).

### 2. `bulkformer_dx/calibration/multitest.py`

- **`benjamini_hochberg(p_values)`**: BH FDR correction (assumes independence).
- **`benjamini_yekutieli(p_values)`**: BY FDR correction (arbitrary dependence).
- **`apply_within_sample(pvalue_matrix, method)`**: Apply BH/BY per row (within-sample).

### 3. `bulkformer_dx/calibration/cohort.py`

- **`get_cohort_indices(sample_ids, cohort_mode, embedding, knn_k)`**: Returns `sample_id -> list[int]` of cohort member indices.
  - `global`: All samples except self.
  - `knn_local`: k nearest neighbors in embedding space (excludes self).

### 4. Cohort-Aware Calibration

`calibrate_ranked_gene_scores` now supports:

- **`cohort_mode`**: `"global"` (default) or `"knn_local"`.
- **`knn_k`**: Number of neighbors for kNN (default 50).
- **`embeddings`**: (n_samples, d) array for kNN cohort selection.

When `knn_local` is used:

- Sigma is estimated per sample from its k nearest neighbors only.
- Gene-wise residual centers are computed per sample from its cohort.
- Improves calibration when cohorts are heterogeneous.

### 5. CLI Options

```bash
bulkformer-dx anomaly calibrate --scores path/to/scores --output-dir path/to/out \
  --cohort-mode global|knn_local \
  --knn-k 50 \
  --embedding-path path/to/embeddings.npy  # optional for knn_local
```

For `knn_local`:

- Embeddings are loaded from `scores_dir/embeddings.npy` if NLL scoring saved them.
- Or provide `--embedding-path` to a `.npy` or `.npz` file.

### 6. Embedding Persistence

NLL scoring (`anomaly score --score-type nll`) now saves `embeddings.npy` to the output directory. Sample order matches `cohort_scores.tsv`. Calibration can then use `knn_local` without `--embedding-path`.

## Usage

### Global cohort (default)

```bash
bulkformer-dx anomaly calibrate --scores scores/ --output-dir calibrated/
```

### kNN local cohort (requires embeddings)

```bash
# After NLL scoring (embeddings saved automatically)
bulkformer-dx anomaly score --score-type nll --input preprocess/ --valid-gene-mask ... --output-dir scores/
bulkformer-dx anomaly calibrate --scores scores/ --output-dir calibrated/ --cohort-mode knn_local --knn-k 50

# Or with explicit embedding file
bulkformer-dx anomaly calibrate --scores scores/ --output-dir calibrated/ \
  --cohort-mode knn_local --knn-k 50 --embedding-path embeddings.npy
```

## Expected Artifacts

- `calibration_run.json` includes `cohort_mode` and `knn_k` when applicable.
- `absolute_outliers.tsv` has per-sample z-scores and p-values calibrated per cohort.

## Failure Modes

When `knn_local` is used:

- **Too small k**: With k &lt; 10, sigma estimates can be noisy; consider k ≥ 20.
- **Mixed cohorts**: If embeddings do not separate subpopulations, local cohort may mix samples from different conditions, degrading calibration.
- **Missing embeddings**: `knn_local` requires embeddings; either run NLL scoring first or provide `--embedding-path`.

## Next Steps (Benchmark)

To fully evaluate global vs local:

1. Run benchmark harness with `cohort_mode` in the method grid.
2. Compare significant-gene distributions, overlap metrics, and injected benchmark performance.
3. Report UMAP/PCA of embeddings colored by metadata when available.
