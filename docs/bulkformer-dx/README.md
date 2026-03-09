# BulkFormer Diagnostics Toolkit

This section tracks the new `bulkformer_dx` command-line toolkit for BulkFormer diagnostics workflows.

## Planned Workflows

- `preprocess`: implemented counts loading, TPM normalization, `log1p(TPM)`, and BulkFormer gene alignment.
- `anomaly`: Monte Carlo masking residual ranking, frozen-backbone head training, and cohort calibration are implemented.
- `tissue`: tissue validation training and prediction flows.
- `proteomics`: proteomics training and inference from BulkFormer embeddings.

## Status

The preprocessing workflow plus the full anomaly workflow stack are now implemented with unit tests
and working CLIs. The remaining command groups stay scaffolded for follow-up rollout steps.
