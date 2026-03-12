"""TabPFN-style MC masked pseudo-likelihood scoring."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from bulkformer_dx.anomaly.scoring import generate_mc_mask_plan
from bulkformer_dx.io.schemas import (
    AlignedExpressionBundle,
    GeneOutlierRow,
    MethodConfig,
    ModelPredictionBundle,
)
from bulkformer_dx.model.uncertainty import resolve_sigma
from bulkformer_dx.stats.gaussian import gaussian_logpdf, student_t_logpdf
from bulkformer_dx.stats.nb import nb_logpmf

MASK_TOKEN = -10.0
DEFAULT_STUDENT_T_DF = 5.0


def _collect_masked_residuals_by_gene(
    mask_plan: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
    gene_ids: list[str],
) -> dict[str, np.ndarray]:
    """Collect residuals for masked positions per gene."""
    n_samples, n_genes = observed.shape
    residuals_by_gene: dict[str, list[float]] = {g: [] for g in gene_ids}
    for s in range(n_samples):
        for p in range(mask_plan.shape[1]):
            for g in range(n_genes):
                if mask_plan[s, p, g]:
                    res = observed[s, g] - predicted[p, s, g]
                    if np.isfinite(res):
                        residuals_by_gene[gene_ids[g]].append(res)
    return {
        g: np.array(vals, dtype=float)
        for g, vals in residuals_by_gene.items()
        if len(vals) > 0
    }


def compute_mc_masked_loglikelihood_scores(
    bundle: AlignedExpressionBundle,
    preds: ModelPredictionBundle,
    *,
    config: MethodConfig | None = None,
    mask_plan: np.ndarray | None = None,
    mc_passes: int = 16,
    mask_prob: float = 0.15,
    seed: int = 0,
    distribution: str = "gaussian",
    uncertainty_source: str = "cohort_sigma",
    student_t_df: float = DEFAULT_STUDENT_T_DF,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Compute MC masked pseudo-likelihood (NLL) scores.

    For each MC pass, masked genes get predictions. Compute -log p(y_obs | y_hat, sigma)
    and aggregate per-gene (mean over passes) and per-sample (mean over masked genes).

    Args:
        bundle: Aligned expression data.
        preds: Model predictions with mc_samples (n_mc, n_samples, n_genes).
        config: Optional method config.
        mask_plan: Optional precomputed mask plan.
        mc_passes: MC passes.
        mask_prob: Mask probability.
        seed: RNG seed.
        distribution: "gaussian" | "student_t" | "negative_binomial".
        uncertainty_source: "cohort_sigma" | "sigma_head" | "mc_variance".
        student_t_df: Degrees of freedom for Student-t.

    Returns:
        (ranked_gene_scores dict, cohort_scores DataFrame).
    """
    if preds.mc_samples is None:
        raise ValueError("Pseudolikelihood scoring requires mc_samples in preds.")

    if config is not None:
        mc_passes = config.mc_passes
        mask_prob = config.mask_rate
        seed = config.seed
        distribution = config.distribution_family
        uncertainty_source = config.uncertainty_source
        student_t_df = config.student_t_df

    Y = np.asarray(bundle.Y_obs, dtype=float)
    n_samples, n_genes = Y.shape
    valid_flags = np.any(bundle.valid_mask, axis=0)
    gene_ids = bundle.gene_ids
    sample_ids = bundle.sample_ids
    mc_samples = np.asarray(preds.mc_samples, dtype=float)

    if mc_samples.shape[0] != mc_passes:
        raise ValueError(
            f"mc_samples first dim {mc_samples.shape[0]} != mc_passes {mc_passes}."
        )

    rng = np.random.default_rng(seed)
    if mask_plan is None:
        mask_plan = generate_mc_mask_plan(
            valid_flags,
            sample_count=n_samples,
            mc_passes=mc_passes,
            mask_prob=mask_prob,
            rng=rng,
        )

    # Residuals for cohort_sigma
    residuals_by_gene = _collect_masked_residuals_by_gene(
        mask_plan, Y, mc_samples, gene_ids
    )
    sigma_by_gene = resolve_sigma(
        uncertainty_source,
        cohort_residuals=residuals_by_gene,
        sigma_hat=preds.sigma_hat,
        mc_samples=mc_samples,
        gene_ids=gene_ids,
    )

    # Build sigma array (n_samples, n_genes)
    sigma_arr = np.full((n_samples, n_genes), np.nan, dtype=float)
    for j, g in enumerate(gene_ids):
        sigma_arr[:, j] = sigma_by_gene.get(g, 1e-6)

    # Compute NLL per (pass, sample, gene) for masked positions
    nll_sum_by_sg = np.zeros((n_samples, n_genes), dtype=float)
    nll_count_by_sg = np.zeros((n_samples, n_genes), dtype=float)

    for p in range(mc_passes):
        pred_p = mc_samples[p]
        for s in range(n_samples):
            for g in range(n_genes):
                if not mask_plan[s, p, g]:
                    continue
                y_obs = Y[s, g]
                y_hat = pred_p[s, g]
                sigma = sigma_arr[s, g]
                if distribution == "gaussian":
                    logp = gaussian_logpdf(
                        np.array([y_obs]),
                        np.array([y_hat]),
                        np.array([sigma]),
                    )[0]
                elif distribution == "student_t":
                    logp = student_t_logpdf(
                        np.array([y_obs]),
                        np.array([y_hat]),
                        np.array([sigma]),
                        df=student_t_df,
                    )[0]
                elif distribution == "negative_binomial" and bundle.counts is not None:
                    k = int(bundle.counts[s, g])
                    mu = max(np.expm1(y_hat) * 1e6 / 1e6, 1e-10)
                    size = 1.0 / 0.1
                    logp = nb_logpmf(
                        np.array([k]),
                        np.array([mu]),
                        np.array([size]),
                    )[0]
                else:
                    logp = gaussian_logpdf(
                        np.array([y_obs]),
                        np.array([y_hat]),
                        np.array([sigma]),
                    )[0]
                nll = -logp
                if np.isfinite(nll):
                    nll_sum_by_sg[s, g] += nll
                    nll_count_by_sg[s, g] += 1

    # Aggregate
    nll_count_by_sg = np.maximum(nll_count_by_sg, 1e-10)
    mean_nll_by_sg = nll_sum_by_sg / nll_count_by_sg
    mean_nll_by_sg[nll_count_by_sg < 0.5] = np.nan

    ranked_gene_scores: dict[str, pd.DataFrame] = {}
    for i, sample_id in enumerate(sample_ids):
        scored = nll_count_by_sg[i] > 0
        genes_scored = np.where(scored)[0]
        if len(genes_scored) == 0:
            ranked_gene_scores[sample_id] = pd.DataFrame()
            continue
        rows = []
        for g in genes_scored:
            nll_val = mean_nll_by_sg[i, g]
            y_obs = Y[i, g]
            y_hat = np.nanmean(mc_samples[:, i, g])
            rows.append(
                GeneOutlierRow(
                    sample_id=sample_id,
                    gene_id=gene_ids[g],
                    y_obs=float(y_obs),
                    y_hat=float(y_hat),
                    residual=float(y_obs - y_hat),
                    score_gene=float(nll_val),
                    p_raw=None,
                    p_adj=None,
                    direction=None,
                    method_id=config.method_id if config else "nll",
                    diagnostics_json={
                        "nll_score": float(nll_val),
                        "sigma_used": float(sigma_arr[i, g]),
                        "masked_count": int(nll_count_by_sg[i, g]),
                    },
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

    sample_nll = np.nanmean(mean_nll_by_sg, axis=1)
    cohort_scores = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "score_sample": sample_nll,
            "cohort_mode": config.cohort_mode if config else "global",
            "method_id": config.method_id if config else "nll",
        }
    )
    return ranked_gene_scores, cohort_scores
