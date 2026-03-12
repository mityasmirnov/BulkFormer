# BulkFormer DX Harmonized Report Template

Use this template for demo, clinical, and methods-comparison runs so outputs are directly comparable.

## 1) Run Metadata

- **Report date**:
- **Report type**: demo | clinical | methods_comparison
- **Variant**: 37M | 147M
- **Git commit**:
- **Environment**: (conda env, python, torch, device)
- **Input cohort**: (counts file, sample count, gene count)
- **Notebook/script entrypoint**:

## 2) Objective

- Primary question:
- Clinical/benchmark context:
- Success criteria:

## 3) Data and Preprocessing QC

### 3.1 Inputs
- Counts path:
- Annotation path:
- Sample annotation path:

### 3.2 QC summary table

| Metric | Value |
| --- | ---: |
| Samples | |
| Raw genes | |
| Collapsed genes | |
| BulkFormer valid genes | |
| BulkFormer valid gene fraction | |
| TPM total median | |

### 3.3 QC figures
- TPM totals histogram:
- Valid-gene fraction histogram:
- Log1p distribution:

## 4) Scoring Configuration

| Parameter | Value |
| --- | --- |
| Model variant | |
| MC passes | |
| Mask probability | |
| Batch size | |
| Device | |
| Scoring mode | residual | nll |

## 5) Calibration Configuration

| Parameter | Value |
| --- | --- |
| Method(s) | none | student_t | nb_approx | nb_outrider | knn_local | nll |
| Alpha | |
| Multiple-testing correction | |
| Cohort strategy | global | local |

## 6) Core Results

### 6.1 Cohort-level summary

| Metric | Value |
| --- | ---: |
| Mean abs residual | |
| Genes scored | |
| Mean significant genes/sample | |
| Median significant genes/sample | |
| Max significant genes/sample | |

### 6.2 Method comparison (if applicable)

| Method | Mean outliers | Median outliers | Max outliers | Notes |
| --- | ---: | ---: | ---: | --- |

### 6.3 Benchmark metrics (spike-in or synthetic)

| Metric | Value |
| --- | ---: |
| AUROC | |
| AUPRC | |
| Recall @ FDR 0.05 | |
| Recall @ FDR 0.10 | |
| Precision@K | |

## 7) Diagnostics and Sanity Checks

- P-value distribution diagnostic:
- QQ plot interpretation:
- Outliers-per-sample distribution:
- Top genes plausibility spot-check:
- Cross-method overlap summary:

## 8) Known Issues / Deviations

- Failures (command + error):
- Non-finite values encountered:
- Missing artifacts:
- Mitigations applied:

## 9) Interpretation and Decision

- Preferred method for this cohort:
- Why it is preferred:
- Confidence level: low | medium | high
- Caveats:

## 10) Next Analyses

- Calibration stress tests:
- Stability/reproducibility checks:
- Biological validation checks:
- Threshold sensitivity analyses:

## 11) Reproducibility Appendix

### 11.1 Commands run

```bash
# paste exact commands here
```

### 11.2 Artifact index

| Artifact | Path |
| --- | --- |
| Run manifest | |
| Main report | |
| Figures directory | |
| Tables directory | |

