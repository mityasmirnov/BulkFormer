# Step Three: OUTRIDER-Style NB Test in Count Space

This report documents the implementation of the OUTRIDER-style negative binomial test using BulkFormer as the mean model.

## Implementation Summary

### Key Mapping

Expected count from predicted TPM:

```
μ_count = pred_TPM × (S_j / 10⁶) × L_g
```

where `S_j = Σ_h K_{jh} / L^{kb}_h` is the sample scaling factor and `L_g` is gene length in kb.

### OUTRIDER Two-Sided P-Value

Discrete-safe formula:

```
p_le = P(X ≤ k)
p_eq = P(X = k)
p_ge = 1 - p_le + p_eq
p_2s = 2 × min(0.5, p_le, p_ge)
```

Clamped to [0, 1]. Uses NB parameterization with mean μ and size = 1/α (Var = μ + αμ²).

### Dispersion Estimation

- **MLE** (`fit_nb_dispersion_mle`): Per-gene dispersion via scipy `minimize_scalar` on NB log-likelihood.
- **Moments** (`fit_nb_dispersion_moments`): α = (Var - μ) / μ² from sample moments.
- **Shrinkage** (optional): DESeq2-like trend `α(μ) = asymptDisp + extraPois/μ`, with convex combination in log-space.

### Caching

Dispersion parameters are cached in `nb_params.tsv` and `nb_params_metadata.json` under the cache directory (default: `output_dir/nb_params_cache/`).

### Calibration Wiring

- `count_space_method="nb_outrider"` in `calibrate_ranked_gene_scores`.
- Requires `--count-space-path` pointing to preprocess output (aligned_counts.tsv, gene_lengths_aligned.tsv, sample_scaling.tsv).
- BY correction applied within each sample.
- Output columns: `nb_outrider_p_raw`, `nb_outrider_p_adj`, `nb_outrider_direction`, `nb_outrider_expected_count`.

## Usage

```bash
# Preprocess (produces count-space artifacts)
python -m bulkformer_dx preprocess --counts counts.tsv --annotation anno.tsv --output-dir preprocess_out

# Anomaly score
python -m bulkformer_dx anomaly score --input preprocess_out/aligned_log1p_tpm.tsv --valid-gene-mask preprocess_out/valid_gene_mask.tsv --output-dir scores_out

# Calibrate with NB OUTRIDER
python -m bulkformer_dx anomaly calibrate --scores scores_out --output-dir calibrated_out --count-space-method nb_outrider --count-space-path preprocess_out
```

## Unit Tests

- `expected_counts_from_predicted_tpm`: Formula correctness.
- `outrider_two_sided_nb_pvalue`: Discrete-safe behavior, clamping.
- `fit_nb_dispersion_mle` / `fit_nb_dispersion_moments`: Valid alpha/size.
- `compute_nb_outrider_for_calibration`: Integration with count-space artifacts.

## Diagnostics (To Run)

For full diagnostics on a cohort:

1. **Dispersion plot**: Gene-wise α vs mean count with trend overlay.
2. **P-value histogram**: On null synthetic data (should be ~uniform).
3. **QQ plot**: P-values vs Uniform(0,1).
4. **Power**: AUPRC/AUROC on injected outliers, recall@FDR 0.05/0.1.

Run the benchmark harness with synthetic injected data to generate these.
