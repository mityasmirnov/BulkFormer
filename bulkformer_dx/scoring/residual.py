"""MC residual-based anomaly scoring."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from bulkformer_dx.anomaly.scoring import (
    generate_deterministic_mask_plan,
    generate_mc_mask_plan,
    resolve_valid_gene_flags,
    score_expression_anomalies,
)
from bulkformer_dx.io.schemas import (
    AlignedExpressionBundle,
    GeneOutlierRow,
    MethodConfig,
    ModelPredictionBundle,
    SampleOutlierRow,
)

MASK_TOKEN = -10.0


def _safe_divide(num: np.ndarray, denom: np.ndarray) -> np.ndarray:
    out = np.full(num.shape, np.nan, dtype=float)
    np.divide(num, denom, out=out, where=denom > 0)
    return out


def compute_residual_scores(
    bundle: AlignedExpressionBundle,
    preds: ModelPredictionBundle,
    *,
    config: MethodConfig | None = None,
    mask_plan: np.ndarray | None = None,
    mc_passes: int = 16,
    mask_prob: float = 0.15,
    seed: int = 0,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Compute MC residual-based anomaly scores.

    Uses mean absolute residual over MC masks as score_gene. Compatible with
    existing anomaly scoring outputs.

    Args:
        bundle: Aligned expression data.
        preds: Model predictions. If mc_samples present, uses those; else uses y_hat.
        config: Optional method config (overrides mc_passes, mask_prob, seed).
        mask_plan: Optional precomputed mask plan (samples, mc_passes, genes).
        mc_passes: MC passes when mask_plan not provided.
        mask_prob: Mask probability.
        seed: RNG seed.

    Returns:
        (ranked_gene_scores dict, cohort_scores DataFrame).
    """
    if config is not None:
        mc_passes = config.mc_passes
        mask_prob = config.mask_rate
        seed = config.seed
        mask_schedule = config.mask_schedule
        K_target = config.K_target
    else:
        mask_schedule = "stochastic"
        K_target = 5

    Y = np.asarray(bundle.Y_obs, dtype=float)
    n_samples, n_genes = Y.shape
    valid_flags = np.any(bundle.valid_mask, axis=0)
    gene_ids = bundle.gene_ids
    sample_ids = bundle.sample_ids

    if preds.mc_samples is not None and mask_schedule == "deterministic":
        mc_passes = preds.mc_samples.shape[0]

    rng = np.random.default_rng(seed)
    if mask_plan is None:
        if mask_schedule == "deterministic":
            mask_plan = generate_deterministic_mask_plan(
                valid_flags,
                sample_count=n_samples,
                K_target=K_target,
                mask_prob=mask_prob,
                seed=seed,
                rng=rng,
            )
            mc_passes = mask_plan.shape[1]
        else:
            mask_plan = generate_mc_mask_plan(
                valid_flags,
                sample_count=n_samples,
                mc_passes=mc_passes,
                mask_prob=mask_prob,
                rng=rng,
            )

    # If we have mc_samples, use them; else replicate y_hat
    if preds.mc_samples is not None and preds.mc_samples.shape[0] == mc_passes:
        predicted = preds.mc_samples  # (n_mc, n_samples, n_genes)
    else:
        predicted = np.broadcast_to(
            preds.y_hat[np.newaxis, :, :],
            (mc_passes, n_samples, n_genes),
        ).copy()

    observed = Y[np.newaxis, :, :]
    residuals = observed - predicted
    mask_float = np.transpose(mask_plan.astype(float), (1, 0, 2))
    abs_residual_sum = (np.abs(residuals) * mask_float).sum(axis=0)
    signed_residual_sum = (residuals * mask_float).sum(axis=0)
    mask_counts = mask_plan.sum(axis=1)
    predicted_sum = (predicted * mask_float).sum(axis=0)

    mean_abs_residual = _safe_divide(abs_residual_sum, mask_counts)
    mean_signed_residual = _safe_divide(signed_residual_sum, mask_counts)
    mean_predicted = _safe_divide(predicted_sum, mask_counts)

    ranked_gene_scores: dict[str, pd.DataFrame] = {}
    for i, sample_id in enumerate(sample_ids):
        mask_i = mask_counts[i] > 0
        genes_scored = np.where(mask_i)[0]
        if len(genes_scored) == 0:
            ranked_gene_scores[sample_id] = pd.DataFrame()
            continue
        rows = []
        for g in genes_scored:
            rows.append(
                GeneOutlierRow(
                    sample_id=sample_id,
                    gene_id=gene_ids[g],
                    y_obs=float(Y[i, g]),
                    y_hat=float(mean_predicted[i, g]),
                    residual=float(mean_signed_residual[i, g]),
                    score_gene=float(mean_abs_residual[i, g]),
                    p_raw=None,
                    p_adj=None,
                    direction=None,
                    method_id=config.method_id if config else "residual",
                    diagnostics_json={"masked_count": int(mask_counts[i, g])},
                )
            )
        df = pd.DataFrame([
            {
                "sample_id": r.sample_id,
                "gene_id": r.gene_id,
                "y_obs": r.y_obs,
                "y_hat": r.y_hat,
                "residual": r.residual,
                "score_gene": r.score_gene,
                "p_raw": r.p_raw,
                "p_adj": r.p_adj,
                "direction": r.direction,
                "method_id": r.method_id,
                "diagnostics_json": r.diagnostics_json,
            }
            for r in rows
        ])
        df = df.sort_values("score_gene", ascending=False).reset_index(drop=True)
        ranked_gene_scores[sample_id] = df

    sample_scores = np.nanmean(mean_abs_residual, axis=1)
    cohort_scores = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "score_sample": sample_scores,
            "cohort_mode": config.cohort_mode if config else "global",
            "method_id": config.method_id if config else "residual",
        }
    )
    return ranked_gene_scores, cohort_scores


def residual_scores_from_anomaly_result(
    result: Any,
    config: MethodConfig | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Convert AnomalyScoringResult to residual-style ranked tables.

    Bridges existing anomaly/scoring.py output to the unified format.
    """
    ranked = {}
    for sample_id, df in result.ranked_gene_scores.items():
        if df.empty:
            ranked[sample_id] = df
            continue
        renamed = df.rename(columns={
            "ensg_id": "gene_id",
            "anomaly_score": "score_gene",
            "mean_signed_residual": "residual",
            "observed_expression": "y_obs",
            "mean_predicted_expression": "y_hat",
        })
        renamed["sample_id"] = sample_id
        renamed["p_raw"] = None
        renamed["p_adj"] = None
        renamed["direction"] = None
        renamed["method_id"] = config.method_id if config else "residual"
        ranked[sample_id] = renamed

    cohort = result.cohort_scores.copy()
    cohort["score_sample"] = cohort["mean_abs_residual"]
    cohort["cohort_mode"] = config.cohort_mode if config else "global"
    cohort["method_id"] = config.method_id if config else "residual"
    return ranked, cohort
