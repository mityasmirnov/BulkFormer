# BulkFormer Docs

## Setup

- [Installation](installation.md): platform-specific conda environments plus exact PyTorch 2.5.1 and PyG install commands for macOS MPS and Linux CUDA.

## Diagnostics Toolkit

- [BulkFormer Diagnostics Toolkit](bulkformer-dx/README.md): overview of the `bulkformer_dx` package and current rollout status.
- [Preprocess Workflow](bulkformer-dx/preprocess.md): counts loading, TPM normalization, `log1p(TPM)`, and BulkFormer alignment usage.
- [Anomaly Workflow](bulkformer-dx/anomaly.md): placeholder for anomaly scoring, head training, and calibration docs.
- [Tissue Workflow](bulkformer-dx/tissue.md): placeholder for tissue validation training and prediction docs.
- [Proteomics Workflow](bulkformer-dx/proteomics.md): placeholder for proteomics training and inference docs.

## Development

- [Ralph Workflow](development/ralph-workflow.md): repo-local fresh-context workflow for the diagnostics rollout, including branch and commit discipline.
