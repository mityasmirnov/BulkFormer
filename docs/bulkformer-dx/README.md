# BulkFormer Diagnostics Toolkit

This section tracks the `bulkformer_dx` command-line toolkit for BulkFormer diagnostics workflows.

## Planned Workflows

- `preprocess`: implemented counts loading, TPM normalization, `log1p(TPM)`, and BulkFormer gene alignment.
- `anomaly`: Monte Carlo masking residual ranking, frozen-backbone head training, and cohort calibration are implemented.
- `tissue`: train/predict tissue validation using BulkFormer embeddings plus optional PCA and a random forest classifier.
- `proteomics`: train/predict frozen-backbone protein heads with masked losses, residual ranking, and optional BY-adjusted protein calls.

## Status

All four workflows are now implemented with unit tests and working CLIs:

- `preprocess`: counts -> TPM -> `log1p(TPM)` plus BulkFormer alignment.
- `anomaly`: Monte Carlo masking, sigma-NLL or injected-outlier heads, and cohort calibration.
- `tissue`: embedding extraction, PCA, random forest training, and prediction.
- `proteomics`: frozen BulkFormer embeddings, linear/MLP heads, masked regression, residual ranking, and optional BY-adjusted protein calls.

See the workflow docs in this directory for end-to-end command examples.
