# Install On macOS M1/M2

This is the Apple Silicon / MPS-specific install path for BulkFormer.

## 1. Create The Base Environment

```bash
mamba env create -f envs/bulkformer_macos_mps.yaml
```

If it already exists:

```bash
mamba env update -n bulkformer-mps -f envs/bulkformer_macos_mps.yaml --prune
```

## 2. Bootstrap PyTorch And PyG

```bash
./scripts/dev/bootstrap_env.sh bulkformer-mps macos-mps
```

That installs:

- `torch==2.5.1`
- `torchvision==0.20.1`
- `torchaudio==2.5.1`
- PyG wheels from `https://data.pyg.org/whl/torch-2.5.1+cpu.html`

PyTorch can still use `mps`, but some PyG operators may stay CPU-backed on macOS. That is acceptable
for prototyping and smoke tests.

## 3. Verify MPS

```bash
conda run -n bulkformer-mps python -c "import torch; print({'mps_built': torch.backends.mps.is_built(), 'mps_available': torch.backends.mps.is_available()})"
./scripts/dev/verify_env.sh bulkformer-mps macos-mps
```

The direct runtime pattern is:

```python
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
tensor = torch.randn(2, 3, device=device)
```

## 4. Troubleshooting

- If `torch.backends.mps.is_built()` is `False`, the installed torch wheel was not built with MPS.
- If `is_built()` is `True` but `is_available()` is `False`, the host macOS/Python stack is not exposing MPS correctly.
- If PyG emits extension warnings, rerun `./scripts/dev/verify_env.sh bulkformer-mps macos-mps` and confirm `SparseTensor` still imports.
- If a specific op is unsupported on MPS, rerun the workflow on `--device cpu`.

See `docs/installation.md` for the shared cross-platform environment flow.
