#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <env-name> [macos-mps|linux-cuda|auto]" >&2
  exit 1
fi

env_name="$1"
platform="${2:-auto}"

if [[ "$platform" == "auto" ]]; then
  case "$(uname -s)" in
    Darwin) platform="macos-mps" ;;
    Linux) platform="linux-cuda" ;;
    *)
      echo "Unsupported host platform: $(uname -s)" >&2
      exit 1
      ;;
  esac
fi

conda_bin="${CONDA_EXE:-}"
if [[ -z "$conda_bin" ]]; then
  conda_bin="$(command -v mamba || command -v conda || true)"
fi
if [[ -z "$conda_bin" ]]; then
  echo "Could not locate the conda executable on PATH." >&2
  exit 1
fi
conda_base="$(cd "$(dirname "$conda_bin")/.." && pwd)"
env_python="$conda_base/envs/$env_name/bin/python"

if [[ ! -x "$env_python" ]]; then
  echo "Could not find Python for conda environment: $env_name" >&2
  echo "Expected interpreter at: $env_python" >&2
  exit 1
fi

echo "== Python =="
"$env_python" -V
"$env_python" -c "import sys; print(sys.executable)"

echo "== Core imports =="
"$env_python" -c "import pandas, scanpy, torch, torch_geometric; from torch_geometric.typing import SparseTensor; print({'torch': torch.__version__, 'torch_geometric': torch_geometric.__version__, 'pandas': pandas.__version__, 'scanpy': scanpy.__version__, 'sparse_tensor_available': SparseTensor is not None})"

case "$platform" in
  macos|macos-mps)
    echo "== MPS check =="
    "$env_python" -c "import torch; print({'mps_built': torch.backends.mps.is_built(), 'mps_available': torch.backends.mps.is_available()})"
    ;;
  linux|linux-cuda)
    echo "== CUDA check =="
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi
    else
      echo "nvidia-smi not found on PATH; skipping driver printout"
    fi
    "$env_python" -c "import torch; print({'cuda_available': torch.cuda.is_available(), 'cuda_version': torch.version.cuda, 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})"
    ;;
  *)
    echo "Unsupported verification platform: $platform" >&2
    exit 1
    ;;
esac

echo "== Targeted tests =="
"$env_python" -m pytest \
  tests/test_bulkformer_dx_cli.py \
  tests/test_preprocess.py \
  tests/test_anomaly_scoring.py \
  tests/test_anomaly_head.py \
  tests/test_bulkformer_model.py \
  -k "not smoke"
