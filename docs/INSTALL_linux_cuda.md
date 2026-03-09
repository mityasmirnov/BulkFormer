# Install On Linux CUDA 11.8

This is the production-oriented GPU install path for BulkFormer.

## 1. Create The Base Environment

```bash
mamba env create -f envs/bulkformer_linux_cuda.yaml
```

If it already exists:

```bash
mamba env update -n bulkformer-cuda -f envs/bulkformer_linux_cuda.yaml --prune
```

## 2. Install PyTorch 2.5.1

Official conda-style install:

```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

Official pip-style install:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

The repo bootstrap script applies the same pinned family inside the conda env:

```bash
./scripts/dev/bootstrap_env.sh bulkformer-cuda linux-cuda
```

## 3. Install PyTorch Geometric Wheels

Match the PyG wheels to the exact torch/CUDA build:

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.1+cu118.html
pip install torch-geometric==2.6.1
```

## 4. Verify CUDA And PyG

```bash
conda run -n bulkformer-cuda python -c "import torch; print({'cuda_available': torch.cuda.is_available(), 'cuda_version': torch.version.cuda, 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})"
./scripts/dev/verify_env.sh bulkformer-cuda linux-cuda
```

## 5. Notes

- Keep the torch family and PyG wheel index aligned; do not mix `cu118` and `cu124` packages in one env.
- Ensure the host NVIDIA driver supports CUDA 11.8 runtime wheels.
- If `nvidia-smi` is missing from `PATH`, `verify_env.sh` still checks torch CUDA availability directly.

See `docs/installation.md` for the repo-wide environment workflow and shell-independent `conda run` usage.
