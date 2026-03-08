# Installation

Use the platform-specific environment files in `envs/` for new installs. The legacy `bulkformer.yaml` is kept unchanged for compatibility with older setups.

## macOS (Apple Silicon, MPS)

Create the base environment:

```bash
conda env create -f envs/bulkformer_macos_mps.yaml
conda activate bulkformer-mps
```

Install PyTorch 2.5.1:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
```

Install PyTorch Geometric and its wheel-backed extensions:

```bash
python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.1+cpu.html
python -m pip install torch-geometric==2.6.1
```

Verify that MPS is visible to PyTorch:

```bash
python -c "import torch; print({'mps_built': torch.backends.mps.is_built(), 'mps_available': torch.backends.mps.is_available()})"
python -c "import torch; x = torch.ones(2, device='mps'); print(x.device, x.tolist())"
```

Notes:

- MPS requires Apple Silicon and a recent macOS release. Intel Macs will not expose the `mps` device.
- The PyG wheels above are the CPU wheel set for `torch==2.5.1`; they install cleanly on macOS, but some graph ops may still need CPU fallback even when PyTorch itself sees `mps`.
- If you hit unsupported-op errors during graph execution, rerun on CPU or switch to the Linux CUDA setup below for the most reliable full-model execution.

## Linux (CUDA 11.8)

Create the base environment:

```bash
conda env create -f envs/bulkformer_linux_cuda.yaml
conda activate bulkformer-cuda
```

Install PyTorch 2.5.1 with CUDA 11.8 wheels:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

Install PyTorch Geometric and its CUDA 11.8 extensions:

```bash
python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.1+cu118.html
python -m pip install torch-geometric==2.6.1
```

Verify that CUDA is available:

```bash
nvidia-smi
python -c "import torch; print({'cuda_available': torch.cuda.is_available(), 'cuda_version': torch.version.cuda, 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})"
```

Notes:

- These commands target the same `torch==2.5.1` release used by this repo, pinned to CUDA 11.8 for the PyG extension wheels.
- Make sure your NVIDIA driver supports CUDA 11.8-compatible runtime wheels before installing.
- For remote servers, prefer running inside a fresh shell after `conda activate` so `python -m pip` resolves into the new environment.

## Quick sanity check

After either install path completes, confirm the core stack imports:

```bash
python -c "import torch, torch_geometric, pandas, scanpy; print(torch.__version__, torch_geometric.__version__)"
```

Then follow `model/README.md` and `data/README.md` to download the pretrained checkpoints and data assets before opening `bulkformer_extract_feature.ipynb`.
