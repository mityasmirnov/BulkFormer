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

case "$platform" in
  macos|macos-mps)
    torch_packages=(
      torch==2.5.1
      torchvision==0.20.1
      torchaudio==2.5.1
    )
    pyg_wheel_url="https://data.pyg.org/whl/torch-2.5.1+cpu.html"
    ;;
  linux|linux-cuda)
    torch_packages=(
      torch==2.5.1
      torchvision==0.20.1
      torchaudio==2.5.1
      --index-url
      https://download.pytorch.org/whl/cu118
    )
    pyg_wheel_url="https://data.pyg.org/whl/torch-2.5.1+cu118.html"
    ;;
  *)
    echo "Unsupported target platform: $platform" >&2
    exit 1
    ;;
esac

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

echo "Bootstrapping BulkFormer Python stack into conda environment: $env_name"
"${env_python}" -m pip install "${torch_packages[@]}"
"${env_python}" -m pip install \
  pyg_lib \
  torch_scatter \
  torch_sparse \
  torch_cluster \
  torch_spline_conv \
  -f "$pyg_wheel_url"
"${env_python}" -m pip install torch-geometric==2.6.1

echo "Finished bootstrapping PyTorch and PyG for $platform"
