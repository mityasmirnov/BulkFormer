"""Standardized BulkFormer inference API producing ModelPredictionBundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bulkformer_dx.bulkformer_model import load_bulkformer_model
from bulkformer_dx.io.schemas import AlignedExpressionBundle, MethodConfig, ModelPredictionBundle

DEFAULT_BATCH_SIZE = 16
DEFAULT_MC_PASSES = 16
DEFAULT_MASK_PROB = 0.15
DEFAULT_AGGREGATION = "mean"


@dataclass(slots=True)
class RuntimeConfig:
    """Runtime parameters for inference (matches MethodConfig.runtime)."""

    mc_passes: int = DEFAULT_MC_PASSES
    mask_rate: float = DEFAULT_MASK_PROB
    seed: int = 0
    batch_size: int = DEFAULT_BATCH_SIZE


def predict_mean(
    bundle: AlignedExpressionBundle,
    *,
    loaded_model: Any | None = None,
    variant: str | None = None,
    checkpoint_path: str | Path | None = None,
    model_dir: Path | None = None,
    device: str = "cpu",
    batch_size: int = DEFAULT_BATCH_SIZE,
    aggregation: str = DEFAULT_AGGREGATION,
    model_kwargs: dict[str, Any] | None = None,
) -> ModelPredictionBundle:
    """Run BulkFormer inference and return predictions plus sample embeddings.

    Produces y_hat (predicted mean in expr_space) and embedding (sample-level
    vectors for kNN cohort selection). No masking is applied (mask_prob=0).

    Args:
        bundle: Aligned expression data.
        loaded_model: Optional LoadedBulkFormer to reuse (skips loading).
        variant: Model variant (e.g. "37M"). Auto-discovered if omitted.
        checkpoint_path: Explicit checkpoint path.
        model_dir: Directory to search for checkpoints.
        device: Device for inference.
        batch_size: Batch size for forward passes.
        aggregation: Sample embedding aggregation ("mean", "max", "median", "all").
        model_kwargs: Additional kwargs for load_bulkformer_model.

    Returns:
        ModelPredictionBundle with y_hat and embedding populated.
    """
    from bulkformer_dx.bulkformer_model import extract_sample_embeddings, predict_expression

    if loaded_model is not None:
        loaded = loaded_model
    else:
        kwargs = model_kwargs or {}
        if variant is not None:
            kwargs["variant"] = variant
        if checkpoint_path is not None:
            kwargs["checkpoint_path"] = checkpoint_path
        if model_dir is not None:
            kwargs["model_dir"] = model_dir
        kwargs.setdefault("device", device)
        loaded = load_bulkformer_model(**kwargs)
    Y = np.asarray(bundle.Y_obs, dtype=np.float32)
    valid_indices = np.where(bundle.valid_mask.any(axis=0))[0].tolist()

    y_hat = predict_expression(
        loaded.model,
        Y,
        batch_size=batch_size,
        mask_prob=0.0,
        device=loaded.device,
    )

    embedding = extract_sample_embeddings(
        loaded.model,
        Y,
        batch_size=batch_size,
        mask_prob=0.0,
        aggregation=aggregation,
        device=loaded.device,
        gene_indices=valid_indices if valid_indices else None,
    )

    return ModelPredictionBundle(
        y_hat=y_hat,
        sigma_hat=None,
        embedding=embedding,
        mc_samples=None,
    )


def predict_sigma_head(
    bundle: AlignedExpressionBundle,
    *,
    loaded_model: Any | None = None,
    **load_kwargs: Any,
) -> np.ndarray | None:
    """Return sigma_hat from a learned sigma head if present.

    Current BulkFormer checkpoints do not include a sigma head. Returns None
    when no sigma head weights exist.
    """
    if loaded_model is not None:
        model = loaded_model.model if hasattr(loaded_model, "model") else loaded_model
    else:
        loaded = load_bulkformer_model(**load_kwargs)
        model = loaded.model

    state_keys = {k for k in model.state_dict().keys() if "sigma" in k.lower()}
    if not state_keys:
        return None

    # Sigma head exists but we don't have a standard interface yet.
    # Return None until sigma head is implemented in the model.
    return None


def predict(
    bundle: AlignedExpressionBundle,
    method_config: MethodConfig,
    *,
    loaded_model: Any | None = None,
    variant: str | None = None,
    checkpoint_path: str | Path | None = None,
    model_dir: Path | None = None,
    device: str = "cpu",
    batch_size: int = DEFAULT_BATCH_SIZE,
    model_kwargs: dict[str, Any] | None = None,
) -> ModelPredictionBundle:
    """Unified inference entrypoint driven by MethodConfig.

    Dispatches to predict_mean (no MC) or mc_predict (MC masking) based on
    method_config.mc_passes. Uses method_config.seed for deterministic MC masking.

    Args:
        bundle: Aligned expression data.
        method_config: Method configuration (mc_passes, mask_rate, seed, etc.).
        loaded_model: Optional LoadedBulkFormer to reuse.
        variant: Model variant.
        checkpoint_path: Explicit checkpoint path.
        model_dir: Directory to search for checkpoints.
        device: Device for inference.
        batch_size: Batch size for forward passes.
        model_kwargs: Additional kwargs for load_bulkformer_model.

    Returns:
        ModelPredictionBundle with y_hat, embedding, and optional sigma_hat/mc_samples.
    """
    if method_config.mc_passes > 0:
        pred_bundle, _ = mc_predict(
            bundle,
            loaded_model=loaded_model,
            mc_passes=method_config.mc_passes,
            mask_prob=method_config.mask_rate,
            seed=method_config.seed,
            batch_size=batch_size,
            variant=variant,
            checkpoint_path=checkpoint_path,
            device=device,
            model_kwargs=model_kwargs,
        )
        return pred_bundle
    return predict_mean(
        bundle,
        loaded_model=loaded_model,
        variant=variant,
        checkpoint_path=checkpoint_path,
        model_dir=model_dir,
        device=device,
        batch_size=batch_size,
        model_kwargs=model_kwargs,
    )


def mc_predict(
    bundle: AlignedExpressionBundle,
    *,
    loaded_model: Any | None = None,
    mc_passes: int = DEFAULT_MC_PASSES,
    mask_prob: float = DEFAULT_MASK_PROB,
    seed: int = 0,
    batch_size: int = DEFAULT_BATCH_SIZE,
    variant: str | None = None,
    checkpoint_path: str | Path | None = None,
    device: str = "cpu",
    model_kwargs: dict[str, Any] | None = None,
) -> tuple[ModelPredictionBundle, np.ndarray]:
    """Run MC masking passes and return predictions plus mc_samples.

    Uses deterministic seeding for reproducibility. Each pass masks a random
    subset of valid genes, runs the model, and collects predictions.

    Args:
        bundle: Aligned expression data.
        mc_passes: Number of MC masking passes.
        mask_prob: Fraction of valid genes to mask per pass.
        seed: RNG seed for deterministic masking.
        batch_size: Batch size for forward passes.
        variant: Model variant.
        checkpoint_path: Checkpoint path.
        device: Device for inference.
        model_kwargs: Additional kwargs for load_bulkformer_model.

    Returns:
        Tuple of (ModelPredictionBundle with y_hat, embedding, mc_samples) and
        mc_samples array (n_mc, n_samples, n_genes).
    """
    from bulkformer_dx.anomaly.scoring import generate_mc_mask_plan
    from bulkformer_dx.bulkformer_model import extract_sample_embeddings, predict_expression

    if loaded_model is not None:
        loaded = loaded_model
    else:
        kwargs = model_kwargs or {}
        if variant is not None:
            kwargs["variant"] = variant
        if checkpoint_path is not None:
            kwargs["checkpoint_path"] = checkpoint_path
        kwargs.setdefault("device", device)
        loaded = load_bulkformer_model(**kwargs)
    Y = np.asarray(bundle.Y_obs, dtype=np.float32)
    n_samples, n_genes = Y.shape

    valid_gene_flags = np.any(bundle.valid_mask, axis=0)
    valid_gene_count = int(valid_gene_flags.sum())
    if valid_gene_count == 0:
        raise ValueError("At least one valid gene is required for MC prediction.")

    rng = np.random.default_rng(seed)
    mask_plan = generate_mc_mask_plan(
        valid_gene_flags,
        sample_count=n_samples,
        mc_passes=mc_passes,
        mask_prob=mask_prob,
        rng=rng,
    )

    MASK_TOKEN = -10.0
    mc_samples = np.full((mc_passes, n_samples, n_genes), np.nan, dtype=np.float32)

    for pass_idx in range(mc_passes):
        masked = Y.copy()
        masked[mask_plan[:, pass_idx, :]] = MASK_TOKEN
        genes_masked_this_pass = mask_plan[:, pass_idx, :].sum(axis=1)
        mask_fraction = float(genes_masked_this_pass[0] / n_genes) if n_genes else 0.0

        pred = predict_expression(
            loaded.model,
            masked,
            batch_size=batch_size,
            mask_prob=mask_fraction,
            device=loaded.device,
        )
        mc_samples[pass_idx] = pred

    mu = np.nanmean(mc_samples, axis=0).astype(np.float32)
    sigma_hat = None
    if mc_passes > 1:
        sigma_hat = np.nanstd(mc_samples, axis=0, ddof=1).astype(np.float32)
        sigma_hat = np.maximum(sigma_hat, 1e-6)
    embedding = extract_sample_embeddings(
        loaded.model,
        Y,
        batch_size=batch_size,
        mask_prob=0.0,
        aggregation=DEFAULT_AGGREGATION,
        device=loaded.device,
    )

    return (
        ModelPredictionBundle(
            y_hat=mu,
            sigma_hat=sigma_hat,
            embedding=embedding,
            mc_samples=mc_samples,
        ),
        mc_samples,
    )


def bundle_from_preprocess_result(
    result: Any,
    expr_space: str = "log1p_tpm",
) -> AlignedExpressionBundle:
    """Build AlignedExpressionBundle from PreprocessResult."""
    from bulkformer_dx.anomaly.scoring import resolve_valid_gene_flags

    if expr_space == "log1p_tpm":
        Y = result.aligned_log1p_tpm.to_numpy(dtype=np.float32)
    elif expr_space == "tpm":
        Y = result.aligned_tpm.to_numpy(dtype=np.float32)
    else:
        Y = result.aligned_counts.to_numpy(dtype=np.float32)

    gene_ids = [str(g) for g in result.aligned_log1p_tpm.columns]
    sample_ids = [str(s) for s in result.aligned_log1p_tpm.index]
    valid_flags = resolve_valid_gene_flags(
        result.valid_gene_mask,
        result.aligned_log1p_tpm.columns,
    )
    valid_mask = np.broadcast_to(valid_flags, (len(sample_ids), len(gene_ids)))

    counts = None
    if hasattr(result, "aligned_counts") and result.aligned_counts is not None:
        counts = result.aligned_counts.to_numpy(dtype=np.float32)

    gene_length_kb = None
    if hasattr(result, "gene_lengths_aligned") and result.gene_lengths_aligned is not None:
        if "length_kb" in result.gene_lengths_aligned.columns:
            gene_length_kb = result.gene_lengths_aligned["length_kb"].to_numpy(dtype=np.float32)

    tpm_scaling_S = None
    if hasattr(result, "sample_scaling") and result.sample_scaling is not None:
        if "S_j" in result.sample_scaling.columns:
            tpm_scaling_S = result.sample_scaling["S_j"].to_numpy(dtype=np.float32)

    return AlignedExpressionBundle(
        expr_space=expr_space,
        Y_obs=Y,
        valid_mask=valid_mask,
        gene_ids=gene_ids,
        sample_ids=sample_ids,
        counts=counts,
        gene_length_kb=gene_length_kb,
        tpm_scaling_S=tpm_scaling_S,
        metadata=None,
    )


def bundle_from_paths(
    input_dir: Path,
    *,
    expr_space: str = "log1p_tpm",
) -> AlignedExpressionBundle:
    """Build AlignedExpressionBundle from preprocess output directory."""
    import pandas as pd

    from bulkformer_dx.anomaly.scoring import load_aligned_expression, load_valid_gene_mask, resolve_valid_gene_flags

    input_dir = Path(input_dir)
    if expr_space == "log1p_tpm":
        expr_path = input_dir / "aligned_log1p_tpm.tsv"
    elif expr_space == "tpm":
        expr_path = input_dir / "aligned_tpm.tsv"
    else:
        expr_path = input_dir / "aligned_counts.tsv"

    if not expr_path.exists():
        raise FileNotFoundError(f"Expression file not found: {expr_path}")

    expression = load_aligned_expression(expr_path)
    valid_gene_mask = load_valid_gene_mask(input_dir / "valid_gene_mask.tsv")
    valid_flags = resolve_valid_gene_flags(valid_gene_mask, expression.columns)

    Y = expression.to_numpy(dtype=np.float32)
    gene_ids = [str(g) for g in expression.columns]
    sample_ids = [str(s) for s in expression.index]
    valid_mask = np.broadcast_to(valid_flags, (len(sample_ids), len(gene_ids)))

    counts = None
    counts_path = input_dir / "aligned_counts.tsv"
    if counts_path.exists():
        counts_df = load_aligned_expression(counts_path)
        counts = counts_df.to_numpy(dtype=np.float32)

    gene_length_kb = None
    lengths_path = input_dir / "gene_lengths_aligned.tsv"
    if lengths_path.exists():
        lengths_df = pd.read_csv(lengths_path, sep="\t")
        if "length_kb" in lengths_df.columns:
            lengths_df = lengths_df.set_index("ensg_id").reindex(gene_ids)
            gene_length_kb = lengths_df["length_kb"].to_numpy(dtype=np.float32)

    tpm_scaling_S = None
    scaling_path = input_dir / "sample_scaling.tsv"
    if scaling_path.exists():
        scaling_df = pd.read_csv(scaling_path, sep="\t")
        if "S_j" in scaling_df.columns:
            scaling_df = scaling_df.set_index("sample_id").reindex(sample_ids)
            tpm_scaling_S = scaling_df["S_j"].to_numpy(dtype=np.float32)

    return AlignedExpressionBundle(
        expr_space=expr_space,
        Y_obs=Y,
        valid_mask=valid_mask,
        gene_ids=gene_ids,
        sample_ids=sample_ids,
        counts=counts,
        gene_length_kb=gene_length_kb,
        tpm_scaling_S=tpm_scaling_S,
        metadata=None,
    )
