# Installation

Use the platform-specific environment files in `envs/` for all new installs. The top-level
`bulkformer.yaml` remains in the repo only as a legacy compatibility export and is not the
recommended setup path.

The canonical flow is:

1. Create or update the base conda environment from the platform YAML.
2. Bootstrap the pinned PyTorch and PyG stack with `scripts/dev/bootstrap_env.sh`.
3. Verify the resulting environment with `scripts/dev/verify_env.sh`.

`mamba` is recommended because it is faster and produces more predictable solves than classic
`conda`, but the commands below work with either.

## macOS (Apple Silicon, MPS)

Create the base environment:

```bash
mamba env create -f envs/bulkformer_macos_mps.yaml
```

If the environment already exists, update it in place:

```bash
mamba env update -n bulkformer-mps -f envs/bulkformer_macos_mps.yaml --prune
```

Install the pinned PyTorch 2.5.1 and PyG stack:

```bash
./scripts/dev/bootstrap_env.sh bulkformer-mps macos-mps
```

Verify the environment end to end:

```bash
./scripts/dev/verify_env.sh bulkformer-mps macos-mps
```

Notes:

- MPS requires Apple Silicon and a recent macOS release. Intel Macs will not expose the `mps`
  device.
- The PyG wheel set for macOS uses the CPU wheel channel for `torch==2.5.1`; PyTorch itself can
  still see `mps`, but some graph-heavy operations may need CPU fallback.
- On conda-based macOS installs, PyG may still emit warnings about optional extension libraries
  loading from a different Python framework path. `verify_env.sh` now checks that
  `torch_geometric.typing.SparseTensor` remains available despite those warnings.
- If you hit unsupported-op errors during model execution, rerun the model on CPU or use the
  Linux CUDA path below for the most reliable full-model execution.

## Linux (CUDA 11.8)

Create the base environment:

```bash
mamba env create -f envs/bulkformer_linux_cuda.yaml
```

If the environment already exists, update it in place:

```bash
mamba env update -n bulkformer-cuda -f envs/bulkformer_linux_cuda.yaml --prune
```

Install the pinned PyTorch 2.5.1 and PyG CUDA 11.8 stack:

```bash
./scripts/dev/bootstrap_env.sh bulkformer-cuda linux-cuda
```

Verify the environment end to end:

```bash
./scripts/dev/verify_env.sh bulkformer-cuda linux-cuda
```

Notes:

- These commands pin the torch family to CUDA 11.8 so the PyG extension wheels remain compatible.
- Make sure the host NVIDIA driver supports CUDA 11.8 runtime wheels before bootstrapping.
- `verify_env.sh` prints `nvidia-smi` when available and still performs the torch CUDA check even
  if `nvidia-smi` is absent from `PATH`.
- The Linux verification path now also checks that `torch_sparse` imports cleanly and that a real
  `torch_geometric.typing.SparseTensor` object can be constructed. Treat a failure there as a
  broken PyG install, not as a warning you can ignore.
- For a server-first checklist including asset placement, smoke-test commands, and first-run demo
  commands, use `docs/INSTALL_linux_server.md`.

## CI-like Usage Without Activation

If you want a shell-independent workflow, prefer `conda run` over relying on activation:

```bash
conda run -n bulkformer-mps python -m pytest tests/test_bulkformer_dx_cli.py
conda run -n bulkformer-mps python -m bulkformer_dx.cli --help
```

That approach avoids the common mistake of accidentally running the system `python` instead of the
project interpreter.

## Daily Usage

Activation is still fine for interactive work:

```bash
conda activate bulkformer-mps
python -m bulkformer_dx.cli --help
pytest tests/test_anomaly_head.py
```

If your shell still resolves the wrong `python`, use `python3` explicitly or fall back to
`conda run -n <env> ...`.

## Assets

After the Python environment is ready, follow `model/README.md` and `data/README.md` to download
the pretrained checkpoints and required data assets before opening
`bulkformer_extract_feature.ipynb` or loading a pretrained BulkFormer model.
