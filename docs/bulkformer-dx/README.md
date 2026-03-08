# BulkFormer Diagnostics Toolkit

This section tracks the new `bulkformer_dx` command-line toolkit for BulkFormer diagnostics workflows.

## Planned Workflows

- `preprocess`: implemented counts loading, TPM normalization, `log1p(TPM)`, and BulkFormer gene alignment.
- `anomaly`: anomaly ranking, optional head training, and cohort calibration.
- `tissue`: tissue validation training and prediction flows.
- `proteomics`: proteomics training and inference from BulkFormer embeddings.

## Status

The preprocessing workflow is now implemented with unit tests and a working CLI. The remaining command groups are still scaffolded for follow-up rollout steps.
