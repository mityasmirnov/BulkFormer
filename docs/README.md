# BulkFormer Docs

## Setup

- [Installation](installation.md): platform-specific conda environments plus exact PyTorch 2.5.1 and PyG install commands for macOS MPS and Linux CUDA.
- [INSTALL macOS M1](INSTALL_mac_m1.md): Apple Silicon / MPS-specific install and verification steps.
- [INSTALL Linux CUDA](INSTALL_linux_cuda.md): Linux GPU setup with official PyTorch 2.5.1 CUDA 11.8 and PyG wheel commands.

## Diagnostics Toolkit

- [BulkFormer Diagnostics Toolkit](bulkformer-dx/README.md): overview of the `bulkformer_dx` package and current rollout status.
- [Implementation Summary](bulkformer-dx/implementation-summary.md): unified summary of the toolkit implementations and repo modifications completed so far.
- [Diagnostics Architecture](bulkformer-dx/architecture.md): module-level architecture, data flow, and artifact contracts.
- [Diagnostics CLI Reference](bulkformer-dx/cli-reference.md): command-by-command argument and output reference.
- [Preprocess Workflow](bulkformer-dx/preprocess.md): counts loading, TPM normalization, `log1p(TPM)`, and BulkFormer alignment usage.
- [Anomaly Workflow](bulkformer-dx/anomaly.md): anomaly scoring, frozen-backbone head training, and cohort calibration docs.
- [Tissue Workflow](bulkformer-dx/tissue.md): tissue validation training and prediction docs.
- [Proteomics Workflow](bulkformer-dx/proteomics.md): proteomics training, prediction, and residual ranking docs.
- [RNA Anomaly Pipeline](PIPELINE_rna_anomaly.md): repo-level end-to-end RNA preprocessing and anomaly workflow.
- [Proteomics Pipeline](PIPELINE_proteomics.md): repo-level end-to-end frozen-backbone proteomics workflow.

## Development

- [Ralph Workflow](development/ralph-workflow.md): repo-local fresh-context workflow for the diagnostics rollout, including branch and commit discipline.
