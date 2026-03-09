"""Local 37M BulkFormer smoke test for demo RNA validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bulkformer_dx.bulkformer_model import (
    extract_sample_embeddings,
    load_bulkformer_model,
    predict_expression,
)

DEFAULT_OBSERVED_GENES = {
    0: {0: 1.25, 10: 2.5, 250: 0.75, 1024: 3.0},
    1: {1: 0.5, 10: 1.0, 512: 2.0, 2048: 4.5},
}


def build_demo_expression(sample_count: int, gene_length: int, fill_value: float) -> np.ndarray:
    expression = np.full((sample_count, gene_length), fill_value, dtype=np.float32)
    for sample_idx in range(sample_count):
        for gene_idx, value in DEFAULT_OBSERVED_GENES.get(sample_idx, {}).items():
            if gene_idx < gene_length:
                expression[sample_idx, gene_idx] = np.float32(value)
    return expression


def display_asset_path(path: Path) -> str:
    """Prefer repo-relative asset locations for portable smoke-test output."""
    for directory_name in ("model", "data"):
        candidate = REPO_ROOT / directory_name / path.name
        if candidate.exists():
            return str(Path(directory_name) / path.name)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return path.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="37M", help="BulkFormer variant label.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for model loading and the forward pass.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=2,
        help="Number of synthetic samples to include in the smoke-test input.",
    )
    parser.add_argument(
        "--fill-value",
        type=float,
        default=-10.0,
        help="Fill value used for genes that are treated as missing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    loaded = load_bulkformer_model(variant=args.variant, device=args.device)
    expression = build_demo_expression(
        sample_count=args.sample_count,
        gene_length=loaded.config["gene_length"],
        fill_value=args.fill_value,
    )

    predicted_expression = predict_expression(
        loaded.model,
        expression,
        batch_size=1,
        mask_prob=0.15,
        device=loaded.device,
    )
    sample_embeddings = extract_sample_embeddings(
        loaded.model,
        expression,
        batch_size=1,
        aggregation="mean",
        device=loaded.device,
    )

    payload = {
        "variant": loaded.assets.variant,
        "device": str(loaded.device),
        "checkpoint_path": display_asset_path(loaded.assets.checkpoint_path),
        "graph_path": display_asset_path(loaded.assets.graph_path),
        "graph_weights_path": display_asset_path(loaded.assets.graph_weights_path),
        "gene_embedding_path": display_asset_path(loaded.assets.gene_embedding_path),
        "gene_info_path": display_asset_path(loaded.assets.gene_info_path),
        "config": loaded.config,
        "input_shape": list(expression.shape),
        "predicted_expression_shape": list(predicted_expression.shape),
        "sample_embedding_shape": list(sample_embeddings.shape),
        "input_non_fill_count": int(np.sum(expression != args.fill_value)),
        "input_preview": expression[:, :12].tolist(),
        "prediction_preview": predicted_expression[:, :12].tolist(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
