"""Sample embedding extraction from BulkFormer-aligned expression."""

from __future__ import annotations

import argparse
import json
import os
import signal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from bulkformer_dx.anomaly.scoring import (
    load_aligned_expression,
    load_valid_gene_mask,
    resolve_valid_gene_flags,
)
from bulkformer_dx.bulkformer_model import extract_sample_embeddings, load_bulkformer_model

DEFAULT_AGGREGATION = "mean"
SUPPORTED_MODES = ("extract",)


def _atomic_write_tsv(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, path)


def write_embeddings_dataframe(
    embeddings: np.ndarray,
    sample_ids: pd.Index,
    output_path: Path,
) -> Path:
    """Write embeddings as a Samples x (sample_id + dims) TSV (atomic)."""
    output_path = Path(output_path)
    columns = [f"dim_{i}" for i in range(embeddings.shape[1])]
    df = pd.DataFrame(embeddings, index=sample_ids, columns=columns)
    df.index.name = "sample_id"
    df = df.reset_index()
    _atomic_write_tsv(df, output_path)
    return output_path


def extract_embeddings(
    expression: pd.DataFrame,
    valid_gene_flags: np.ndarray,
    *,
    variant: str = "37M",
    device: str = "cpu",
    aggregation: str = DEFAULT_AGGREGATION,
    batch_size: int = 8,
    model_kwargs: dict[str, Any] | None = None,
    checkpoint_dir: Path | None = None,
) -> np.ndarray:
    """Extract per-sample embeddings; resume from checkpoint_dir if present.

    Interrupt (SIGINT/SIGTERM) flushes the partial checkpoint and re-raises
    SystemExit(130) so the process exits cleanly after the current batch.
    """
    kwargs = dict(model_kwargs or {})
    kwargs.setdefault("variant", variant)
    kwargs.setdefault("device", device)
    n_samples = int(expression.shape[0])
    gene_indices = np.where(valid_gene_flags)[0].tolist()

    start_idx = 0
    partial: np.ndarray | None = None
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        progress_path = ckpt_dir / "progress.json"
        partial_path = ckpt_dir / "embeddings_partial.npy"
        if progress_path.is_file() and partial_path.is_file():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            start_idx = int(progress.get("next_index", 0))
            partial = np.load(partial_path)
            if partial.shape[0] != start_idx:
                raise ValueError(
                    f"Checkpoint mismatch: progress next_index={start_idx} but "
                    f"partial rows={partial.shape[0]}"
                )
            if start_idx >= n_samples:
                return partial

    interrupt = {"flag": False}

    def _handle(signum: int, _frame: Any) -> None:
        interrupt["flag"] = True
        print(f"embeddings: got signal {signum}; will checkpoint after current batch", flush=True)

    prev_int = signal.signal(signal.SIGINT, _handle)
    prev_term = signal.signal(signal.SIGTERM, _handle)
    loaded = load_bulkformer_model(**kwargs)
    try:
        chunks: list[np.ndarray] = [] if partial is None else [partial]
        for batch_start in range(start_idx, n_samples, batch_size):
            if interrupt["flag"]:
                break
            batch_end = min(batch_start + batch_size, n_samples)
            batch_expr = expression.iloc[batch_start:batch_end]
            batch_emb = extract_sample_embeddings(
                loaded.model,
                batch_expr,
                batch_size=batch_expr.shape[0],
                aggregation=aggregation,
                device=loaded.device,
                gene_indices=gene_indices,
            )
            chunks.append(np.asarray(batch_emb, dtype=np.float32))
            if ckpt_dir is not None:
                merged = np.concatenate(chunks, axis=0)
                # np.save appends .npy unless the path already ends with .npy
                tmp_npy = ckpt_dir / "embeddings_partial.tmp.npy"
                np.save(tmp_npy, merged)
                os.replace(tmp_npy, ckpt_dir / "embeddings_partial.npy")
                (ckpt_dir / "progress.json").write_text(
                    json.dumps(
                        {
                            "next_index": batch_end,
                            "n_samples": n_samples,
                            "embedding_dim": int(merged.shape[1]),
                            "aggregation": aggregation,
                            "variant": variant,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
        if not chunks:
            raise RuntimeError("No embedding batches were produced.")
        embeddings = np.concatenate(chunks, axis=0)
        if interrupt["flag"]:
            raise SystemExit(130)
        return embeddings
    finally:
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)
        del loaded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
        help="Path to valid_gene_mask.tsv from preprocessing. Optional if --input is a directory.",
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
        help="Batch size for embedding extraction. Defaults to 8 (~1.6 GiB peak on 37M).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from output-dir/checkpoint if present.",
    )
    parser.set_defaults(func=_run_extract)


def _run_extract(args: argparse.Namespace) -> int:
    """Run embeddings extract subcommand."""
    if args.mode != "extract":
        return 1
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if input_path.is_dir():
        expr_path = input_path / "aligned_log1p_tpm.tsv"
        valid_mask_path = input_path / "valid_gene_mask.tsv"
    else:
        expr_path = input_path
        if args.valid_gene_mask is None:
            raise ValueError("--valid-gene-mask is required when --input is a file.")
        valid_mask_path = Path(args.valid_gene_mask)

    expression = load_aligned_expression(expr_path)
    valid_gene_mask = load_valid_gene_mask(valid_mask_path)
    valid_gene_flags = resolve_valid_gene_flags(valid_gene_mask, expression.columns)

    # Always checkpoint so Ctrl-C / SIGTERM is recoverable; --resume is documented alias.
    ckpt_dir = output_dir / "checkpoint"
    _ = args.resume
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    embeddings = extract_embeddings(
        expression,
        valid_gene_flags,
        variant=args.variant,
        device=args.device,
        aggregation=args.aggregation,
        batch_size=args.batch_size,
        checkpoint_dir=ckpt_dir,
    )

    if embeddings.shape[0] != expression.shape[0]:
        # Incomplete after interrupt — checkpoint already on disk.
        print(
            f"Incomplete: {embeddings.shape[0]}/{expression.shape[0]} samples; "
            f"resume with --resume from {ckpt_dir}",
            flush=True,
        )
        return 130

    output_path = output_dir / "sample_embeddings.tsv"
    best_path = output_dir / "sample_embeddings.best.tsv"
    write_embeddings_dataframe(embeddings, expression.index, output_path)
    write_embeddings_dataframe(embeddings, expression.index, best_path)

    peak_mib = 0.0
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        peak_mib = torch.cuda.max_memory_allocated() / (1024**2)

    summary = {
        "samples": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "output_path": str(output_path),
        "best_path": str(best_path),
        "variant": args.variant,
        "aggregation": args.aggregation,
        "batch_size": int(args.batch_size),
        "device": args.device,
        "peak_vram_mib": round(peak_mib, 1),
        "status": "complete",
    }
    summary_path = output_dir / "embeddings_run.json"
    tmp_summary = summary_path.with_suffix(".json.tmp")
    tmp_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_summary, summary_path)

    # Clear checkpoint only after best export succeeds.
    for name in ("embeddings_partial.npy", "progress.json"):
        p = ckpt_dir / name
        if p.is_file():
            p.unlink()

    print(
        f"Wrote {output_path} and {best_path} "
        f"({embeddings.shape[0]} samples x {embeddings.shape[1]} dims; "
        f"peak_vram_mib={peak_mib:.1f})",
        flush=True,
    )
    return 0
