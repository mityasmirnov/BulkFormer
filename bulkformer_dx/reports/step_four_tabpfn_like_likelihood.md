# Step Four: TabPFN-Style Pseudo-Likelihood Scoring

## Overview

TabPFN frames unsupervised outlier detection as density estimation via the chain rule. BulkFormer is masked-conditional, so we implement a **pseudo-likelihood** surrogate: estimate log p(y_g | context) for masked genes across MC passes and aggregate.

## Implementation

### PL-mc (MC Masked Likelihood)

- Reuse MC mask plan from residual scoring
- For each pass: mask genes, predict masked values, compute per-masked-gene log-prob under chosen distribution
- Aggregate: per-gene mean NLL over passes; per-sample mean NLL

### Distribution Options

| Distribution | Space | Sigma source |
|--------------|-------|--------------|
| Gaussian | log1p(TPM) | cohort_sigma, sigma_head, mc_variance |
| Student-t | log1p(TPM) | Same |
| NB | counts | nb_dispersion (when counts available) |

### Uncertainty Sources

- **cohort_sigma**: MAD-based per-gene sigma from cohort residuals
- **sigma_head**: Learned sigma head (if checkpoint has it)
- **mc_variance**: Variance of predictions across MC masks

### CLI

```bash
bulkformer-dx anomaly score --input preprocess_out/aligned_log1p_tpm.tsv \
  --valid-gene-mask preprocess_out/valid_gene_mask.tsv \
  --output-dir scores_out \
  --score-type nll
```

## Modules

- `bulkformer_dx/scoring/pseudolikelihood.py`: `compute_mc_masked_loglikelihood_scores`
- `bulkformer_dx/stats/gaussian.py`: `gaussian_logpdf`, `student_t_logpdf`
- `bulkformer_dx/model/uncertainty.py`: `resolve_sigma`

## Comparison

| Metric | Residual | NLL |
|--------|----------|-----|
| Score | mean \|residual\| | mean -log p(y) |
| Calibration | Empirical tail | Z-score or empirical |
| Sigma | Optional | Required for p(y) |

## Planned Diagnostics

- NLL vs residual scatter on same injected dataset
- Performance vs MC passes
- Sigma source comparison (cohort vs mc_variance)
