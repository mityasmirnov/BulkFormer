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

## Read This Next

- `implementation-summary.md`: unified summary of the implementations and repo modifications
- `architecture.md`: how the package is organized internally
- `cli-reference.md`: top-level CLI inventory and output contracts
- `preprocess.md`: counts normalization and alignment details
- `anomaly.md`: scoring, small heads, and calibration
- `tissue.md`: tissue training and prediction
- `proteomics.md`: RNA-to-proteomics prediction and protein outlier ranking

See the workflow docs in this directory for end-to-end command examples.
