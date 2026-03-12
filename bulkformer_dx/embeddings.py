"""Sample embedding extraction from BulkFormer-aligned expression."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bulkformer_dx.anomaly.scoring import (
    load_aligned_expression,
    load_valid_gene_mask,
    resolve_valid_gene_flags,
)
from bulkformer_dx.bulkformer_model import extract_sample_embeddings, load_bulkformer_model

DEFAULT_AGGREGATION = "mean"
SUPPORTED_MODES = ("extract",)


def extract_embeddings(
    expression: pd.DataFrame,
    valid_gene_flags: np.ndarray,
    *,
    variant: str = "37M",
    device: str = "cpu",
    aggregation: str = DEFAULT_AGGREGATION,
    batch_size: int = 8,
    model_kwargs: dict[str, Any] | None = None,
) -> np.ndarray:
    """Extract per-sample embeddings from BulkFormer."""
    kwargs = model_kwargs or {}
    kwargs.setdefault("variant", variant)
    kwargs.setdefault("device", device)
    loaded = load_bulkformer_model(**kwargs)
    gene_indices = np.where(valid_gene_flags)[0].tolist()
    return extract_sample_embeddings(
        loaded.model,
        expression,
        batch_size=batch_size,
        aggregation=aggregation,
        device=loaded.device,
        gene_indices=gene_indices,
    )


def write_embeddings_dataframe(
    embeddings: np.ndarray,
    sample_ids: pd.Index,
    output_path: Path,
) -> Path:
    """Write embeddings as a Samples x (sample_id + dims) TSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [f"dim_{i}" for i in range(embeddings.shape[1])]
    df = pd.DataFrame(embeddings, index=sample_ids, columns=columns)
    df.index.name = "sample_id"
    df.reset_index(inplace=True)
    df.to_csv(output_path, sep="\t", index=False)
    return output_path


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the embeddings command group."""
    parser = subparsers.add_parser(
        "embeddings",
        help="Extract BulkFormer sample embeddings.",
        description="Extract per-sample embeddings from BulkFormer-aligned expression matrices.",
    )
    parser.add_argument(
        "mode",
        choices=SUPPORTED_MODES,
        help="Extract sample embeddings to a TSV file.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to aligned_log1p_tpm.tsv from preprocessing.",
    )
    parser.add_argument(
        "--valid-gene-mask",
        required=True,
        help="Path to valid_gene_mask.tsv from preprocessing.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where sample_embeddings.tsv will be written.",
    )
    parser.add_argument(
        "--variant",
        default="37M",
        help="BulkFormer model variant. Defaults to 37M.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for inference. Defaults to cpu.",
    )
    parser.add_argument(
        "--aggregation",
        default=DEFAULT_AGGREGATION,
        help="Sample embedding aggregation. Defaults to mean.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for embedding extraction. Defaults to 8.",
    )
    parser.set_defaults(func=_run_extract)


def _run_extract(args: argparse.Namespace) -> int:
    """Run embeddings extract subcommand."""
    if args.mode != "extract":
        return 1
    input_path = Path(args.input)
    valid_gene_mask_path = Path(args.valid_gene_mask)
    output_dir = Path(args.output_dir)


    expression = load_aligned_expression(input_path)
    valid_gene_mask = load_valid_gene_mask(valid_gene_mask_path)
    valid_gene_flags = resolve_valid_gene_flags(valid_gene_mask, expression.columns)

    embeddings = extract_embeddings(
        expression,
        valid_gene_flags,
        variant=args.variant,
        device=args.device,
        aggregation=args.aggregation,
        batch_size=args.batch_size,
    )

    output_path = output_dir / "sample_embeddings.tsv"
    write_embeddings_dataframe(embeddings, expression.index, output_path)

    summary = {
        "samples": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "output_path": str(output_path),
    }
    summary_path = output_dir / "embeddings_run.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {output_path} ({embeddings.shape[0]} samples x {embeddings.shape[1]} dims)")
    return 0
