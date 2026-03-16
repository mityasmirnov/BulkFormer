# Linux Server Setup For BulkFormer

This is the recommended setup path for running BulkFormer and `bulkformer_dx` on a Linux GPU server.

Use this document when you want:

- the `37M` model working first
- a reproducible conda or mamba environment
- a verified PyTorch Geometric stack
- an explicit check that `torch_sparse` and `SparseTensor` are working before you start the project

This guide assumes:

- Linux x86_64
- NVIDIA GPU
- NVIDIA driver compatible with CUDA 11.8 runtime wheels
- repo root is the current working directory

## 1. Choose The Environment

Use:

- environment file: `envs/bulkformer_linux_cuda.yaml`
- environment name: `bulkformer-cuda`

Do not use `bulkformer.yaml` for new server setup. It remains in the repo only for older compatibility workflows.

## 2. Install Conda Or Mamba

`mamba` is preferred because it resolves faster and more reliably.

If you already have `conda`, that is fine too.

Quick check:

```bash
mamba --version || conda --version
```

## 3. Create Or Update The Base Environment

Create:

```bash
mamba env create -f envs/bulkformer_linux_cuda.yaml
```

Update in place:

```bash
mamba env update -n bulkformer-cuda -f envs/bulkformer_linux_cuda.yaml --prune
```

If you do not have `mamba`, replace `mamba` with `conda`.

## 4. Bootstrap PyTorch And PyG

Install the pinned torch family and PyTorch Geometric wheel set:

```bash
./scripts/dev/bootstrap_env.sh bulkformer-cuda linux-cuda
```

This script pins:

- `torch==2.5.1`
- `torchvision==0.20.1`
- `torchaudio==2.5.1`
- PyG wheels from `https://data.pyg.org/whl/torch-2.5.1+cu118.html`

Do not mix other CUDA wheel families such as `cu121` or `cu124` into the same environment.

## 5. Verify CUDA, `torch_sparse`, And `SparseTensor`

Run the repo verification script:

```bash
./scripts/dev/verify_env.sh bulkformer-cuda linux-cuda
```

For Linux CUDA, this verification is expected to prove all of the following:

- Python interpreter resolves from `bulkformer-cuda`
- `torch`, `torch_geometric`, `scanpy`, and related imports work
- `torch.cuda.is_available()` is true
- `torch_sparse` imports successfully
- a real `torch_geometric.typing.SparseTensor(...)` object can be constructed
- the targeted `bulkformer_dx` tests pass

If this step fails, do not proceed to the project runs yet.

## 6. Activate Or Use `conda run`

Interactive use:

```bash
conda activate bulkformer-cuda
python -m bulkformer_dx.cli --help
```

Shell-independent use:

```bash
conda run -n bulkformer-cuda python -m bulkformer_dx.cli --help
```

On servers, `conda run -n bulkformer-cuda ...` is often the safer default because it avoids accidentally picking the system Python.

## 7. Place The Required Assets

For the `37M` setup, place these files exactly here relative to the repo root:

```text
model/
  BulkFormer_37M.pt
data/
  bulkformer_gene_info.csv
  G_tcga.pt
  G_tcga_weight.pt
  esm2_feature_concat.pt
  demo_count_data.csv
  demo_normalized_data.csv
  demo_count_adata.h5ad
  gene_length_df.csv
```

Source guidance:

- `model/README.md`
- `data/README.md`
- latest Zenodo record documented in `data/README.md`

For current `bulkformer_dx` defaults, prefer the `G_tcga.pt` and `G_tcga_weight.pt` assets because those match the loader defaults directly.

## 8. Run The 37M Smoke Test

Before the full demo workflow, confirm model loading and a real forward pass:

```bash
conda run -n bulkformer-cuda python scripts/smoke_test_37m.py --device cuda > reports/smoke_test_stdout.txt 2>&1
```

Expected outcome:

- exit code `0`
- `reports/smoke_test_stdout.txt` contains JSON
- `input_shape` is `[2, 20010]`
- `predicted_expression_shape` is `[2, 20010]`

If your server does not expose CUDA yet, you can temporarily use `--device cpu`, but the goal on Linux is to get the CUDA path healthy.

## 9. Run The Demo RNA Workflow

Preprocess:

```bash
conda run -n bulkformer-cuda python -m bulkformer_dx.cli preprocess \
  --counts data/demo_count_data.csv \
  --annotation data/gene_length_df.csv \
  --output-dir runs/demo_preprocess_37M \
  --counts-orientation samples-by-genes
```

Anomaly scoring:

```bash
conda run -n bulkformer-cuda python -m bulkformer_dx.cli anomaly score \
  --input runs/demo_preprocess_37M/aligned_log1p_tpm.tsv \
  --valid-gene-mask runs/demo_preprocess_37M/valid_gene_mask.tsv \
  --output-dir runs/demo_anomaly_score_37M \
  --variant 37M \
  --device cuda \
  --mask-schedule deterministic --K-target 5 --mask-prob 0.10
```

Calibration:

```bash
conda run -n bulkformer-cuda python -m bulkformer_dx.cli anomaly calibrate \
  --scores runs/demo_anomaly_score_37M \
  --output-dir runs/demo_anomaly_calibrated_37M \
  --alpha 0.05
```

## 10. Initial Sanity Checks

After preprocess, inspect:

- `runs/demo_preprocess_37M/preprocess_report.json`
- `runs/demo_preprocess_37M/tpm.tsv`
- `runs/demo_preprocess_37M/aligned_log1p_tpm.tsv`
- `runs/demo_preprocess_37M/valid_gene_mask.tsv`

Look for:

- sample count is correct
- input gene count is correct
- `bulkformer_valid_gene_fraction` is high
- no missing annotation lengths if you expect the official demo
- TPM totals are near `1e6`

After anomaly scoring, inspect:

- `runs/demo_anomaly_score_37M/cohort_scores.tsv`
- `runs/demo_anomaly_score_37M/gene_qc.tsv`

Look for:

- `gene_coverage_fraction` remains high
- `masked_count` is not zero for most genes

After calibration, inspect:

- `runs/demo_anomaly_calibrated_37M/calibration_summary.tsv`
- `runs/demo_anomaly_calibrated_37M/absolute_outliers.tsv`

Look for:

- empirical BY calls are usually sparse
- if absolute outlier calls are unexpectedly huge, check fill-value handling and distribution shift

## 11. Troubleshooting

### `torch.cuda.is_available()` is false

- check `nvidia-smi`
- confirm the NVIDIA driver supports CUDA 11.8 runtime wheels
- confirm you bootstrapped with `linux-cuda`, not `macos-mps`

### `torch_sparse` import fails

- re-run `./scripts/dev/bootstrap_env.sh bulkformer-cuda linux-cuda`
- verify you did not install a mismatched torch or PyG wheel family afterward
- remove and recreate the env if the package set was mixed manually

### `SparseTensor` construction fails

- treat this as a broken environment
- do not start BulkFormer runs until `./scripts/dev/verify_env.sh bulkformer-cuda linux-cuda` passes
- the Linux goal is to have the real sparse stack healthy, not to rely on a degraded setup

### PyG extension warnings (pyg-lib, torch-scatter, torch-sparse, etc.)

If you see `UserWarning: An issue occurred while importing 'torch-scatter'` or similar, the PyG extension wheels are built for a different PyTorch/CUDA version than your environment. Fix:

1. Check your PyTorch and CUDA versions:
   ```bash
   python -c "import torch; print(torch.__version__, torch.version.cuda)"
   ```
2. Reinstall matching wheels from the [PyG wheel index](https://data.pyg.org/whl/):
   ```bash
   pip install --force-reinstall --no-cache-dir torch_scatter torch_sparse torch_cluster torch_spline_conv \
     -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
   ```
   Replace `${TORCH}` and `${CUDA}` with your versions (e.g. `2.10.0` and `cu128`).
3. If `pyg_lib` has no matching wheel, uninstall it: `pip uninstall pyg_lib`. The other extensions work without it; BulkFormer uses `torch_sparse` for `SparseTensor` on GPU.

### Wrong Python interpreter

Prefer:

```bash
conda run -n bulkformer-cuda python -m bulkformer_dx.cli --help
```

instead of assuming `python` resolves correctly in a remote shell session.

## 12. First Commands To Run On A Fresh Server

```bash
git clone <your-fork-url>
cd BulkFormer
mamba env create -f envs/bulkformer_linux_cuda.yaml
./scripts/dev/bootstrap_env.sh bulkformer-cuda linux-cuda
./scripts/dev/verify_env.sh bulkformer-cuda linux-cuda
conda run -n bulkformer-cuda python scripts/smoke_test_37m.py --device cuda
conda run -n bulkformer-cuda python -m bulkformer_dx.cli preprocess --help
conda run -n bulkformer-cuda python -m bulkformer_dx.cli anomaly --help
```

After those pass, the project is initialized correctly for Linux server work.
