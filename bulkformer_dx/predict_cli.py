"""CLI for BulkFormer inference producing ModelPredictionBundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bulkformer_dx.model.bulkformer import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MC_PASSES,
    DEFAULT_MASK_PROB,
    bundle_from_paths,
    mc_predict,
    predict_mean,
)


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the predict command."""
    parser = subparsers.add_parser(
        "predict",
        help="Run BulkFormer inference and write ModelPredictionBundle outputs.",
        description=(
            "Standardized inference API producing y_hat, embeddings, and optional "
            "MC samples for anomaly scoring and benchmarking."
        ),
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Preprocess output directory (aligned_log1p_tpm.tsv, valid_gene_mask.tsv, etc.).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for y_hat, embeddings, and optional mc_samples.",
    )
    parser.add_argument(
        "--variant",
        default="37M",
        help="BulkFormer model variant. Defaults to 37M.",
    )
    parser.add_argument(
        "--checkpoint-path",
        help="Optional path to a specific BulkFormer checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for inference. Defaults to cpu.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for forward passes.",
    )
    parser.add_argument(
        "--mc-passes",
        type=int,
        default=0,
        help="If > 0, run MC masking and write mc_samples. Default 0 (mean only).",
    )
    parser.add_argument(
        "--mask-prob",
        type=float,
        default=DEFAULT_MASK_PROB,
        help="Fraction of valid genes to mask per MC pass. Used when mc-passes > 0.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for deterministic MC masking.",
    )
    parser.set_defaults(func=run_predict)


def run_predict(args: argparse.Namespace) -> int:
    """Execute predict command."""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = bundle_from_paths(input_dir, expr_space="log1p_tpm")

    model_kwargs = {
        "variant": args.variant,
        "checkpoint_path": args.checkpoint_path,
        "device": args.device,
    }
    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

    if args.mc_passes > 0:
        pred_bundle, mc_samples = mc_predict(
            bundle,
            mc_passes=args.mc_passes,
            mask_prob=args.mask_prob,
            seed=args.seed,
            batch_size=args.batch_size,
            model_kwargs=model_kwargs,
        )
        import pandas as pd

        for i in range(mc_samples.shape[0]):
            pd.DataFrame(
                mc_samples[i],
                index=bundle.sample_ids,
                columns=bundle.gene_ids,
            ).to_csv(output_dir / f"mc_sample_{i:04d}.tsv", sep="\t")
    else:
        pred_bundle = predict_mean(
            bundle,
            batch_size=args.batch_size,
            model_kwargs=model_kwargs,
        )

    import pandas as pd

    pd.DataFrame(
        pred_bundle.y_hat,
        index=bundle.sample_ids,
        columns=bundle.gene_ids,
    ).to_csv(output_dir / "y_hat.tsv", sep="\t")

    if pred_bundle.embedding is not None:
        emb_cols = [f"dim_{i}" for i in range(pred_bundle.embedding.shape[1])]
        pd.DataFrame(
            pred_bundle.embedding,
            index=bundle.sample_ids,
            columns=emb_cols,
        ).to_csv(output_dir / "sample_embeddings.tsv", sep="\t")

    if pred_bundle.sigma_hat is not None:
        pd.DataFrame(
            pred_bundle.sigma_hat,
            index=bundle.sample_ids,
            columns=bundle.gene_ids,
        ).to_csv(output_dir / "sigma_hat.tsv", sep="\t")

    summary = {
        "samples": len(bundle.sample_ids),
        "genes": len(bundle.gene_ids),
        "embedding_dim": int(pred_bundle.embedding.shape[1]) if pred_bundle.embedding is not None else None,
        "mc_passes": args.mc_passes if args.mc_passes > 0 else None,
        "sigma_hat_written": pred_bundle.sigma_hat is not None,
    }
    with (output_dir / "predict_run.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {output_dir / 'y_hat.tsv'} ({pred_bundle.y_hat.shape[0]} x {pred_bundle.y_hat.shape[1]})")
    if pred_bundle.embedding is not None:
        print(f"Wrote {output_dir / 'sample_embeddings.tsv'} ({pred_bundle.embedding.shape[0]} x {pred_bundle.embedding.shape[1]})")
    if pred_bundle.sigma_hat is not None:
        print(f"Wrote {output_dir / 'sigma_hat.tsv'} (mc_variance)")
    return 0
