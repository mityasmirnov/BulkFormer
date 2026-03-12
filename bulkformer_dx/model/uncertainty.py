"""Sigma/uncertainty selection for anomaly scoring."""

from __future__ import annotations

from typing import Any

import numpy as np

SIGMA_EPSILON = 1e-6
MAD_TO_SIGMA = 1.4826


def sigma_from_cohort_residuals(
    residuals_by_gene: dict[str, np.ndarray],
    *,
    epsilon: float = SIGMA_EPSILON,
) -> dict[str, float]:
    """Estimate robust sigma per gene from cohort residuals (MAD-based).

    Args:
        residuals_by_gene: Gene ID -> array of residuals across samples.
        epsilon: Minimum sigma for numerical stability.

    Returns:
        Gene ID -> sigma.
    """
    sigma_by_gene: dict[str, float] = {}
    for gene_id, residuals in residuals_by_gene.items():
        residuals = np.asarray(residuals, dtype=float)
        residuals = residuals[np.isfinite(residuals)]
        if residuals.size == 0:
            sigma_by_gene[gene_id] = epsilon
            continue
        center = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - center)))
        sigma = MAD_TO_SIGMA * mad
        if not np.isfinite(sigma) or sigma <= epsilon:
            ddof = 1 if residuals.size > 1 else 0
            sigma = float(np.std(residuals, ddof=ddof))
        if not np.isfinite(sigma) or sigma <= epsilon:
            sigma = epsilon
        sigma_by_gene[gene_id] = sigma
    return sigma_by_gene


def sigma_from_sigma_head(
    sigma_hat: np.ndarray,
    gene_ids: list[str],
    *,
    epsilon: float = SIGMA_EPSILON,
) -> dict[str, float]:
    """Use model sigma head predictions as per-gene sigma.

    sigma_hat is (n_samples, n_genes). Returns gene-wise mean or median.

    Args:
        sigma_hat: Predicted sigma from sigma head.
        gene_ids: Ordered gene IDs.
        epsilon: Minimum sigma.

    Returns:
        Gene ID -> sigma.
    """
    sigma_hat = np.asarray(sigma_hat, dtype=float)
    if sigma_hat.ndim != 2:
        raise ValueError("sigma_hat must be 2D (samples, genes).")
    if sigma_hat.shape[1] != len(gene_ids):
        raise ValueError("sigma_hat columns must match gene_ids length.")
    sigma_by_gene = {}
    for j, gene_id in enumerate(gene_ids):
        col = sigma_hat[:, j]
        col = col[np.isfinite(col) & (col > 0)]
        sigma = float(np.median(col)) if col.size else epsilon
        sigma_by_gene[gene_id] = max(sigma, epsilon)
    return sigma_by_gene


def sigma_from_mc_variance(
    mc_samples: np.ndarray,
    gene_ids: list[str],
    *,
    epsilon: float = SIGMA_EPSILON,
) -> dict[str, float]:
    """Use variance of MC predictions across masks as sigma proxy.

    mc_samples is (n_mc, n_samples, n_genes). Returns gene-wise std across MC.

    Args:
        mc_samples: MC prediction samples.
        gene_ids: Ordered gene IDs.
        epsilon: Minimum sigma.

    Returns:
        Gene ID -> sigma (sqrt of mean variance across samples).
    """
    mc_samples = np.asarray(mc_samples, dtype=float)
    if mc_samples.ndim != 3:
        raise ValueError("mc_samples must be 3D (n_mc, n_samples, n_genes).")
    if mc_samples.shape[2] != len(gene_ids):
        raise ValueError("mc_samples genes must match gene_ids length.")
    var_per_gene = np.nanvar(mc_samples, axis=0, ddof=1)
    mean_var = np.nanmean(var_per_gene, axis=0)
    sigma_arr = np.sqrt(np.maximum(mean_var, 0.0))
    sigma_by_gene = {
        gene_id: max(float(sigma_arr[j]), epsilon)
        for j, gene_id in enumerate(gene_ids)
    }
    return sigma_by_gene


def resolve_sigma(
    source: str,
    *,
    cohort_residuals: dict[str, np.ndarray] | None = None,
    sigma_hat: np.ndarray | None = None,
    mc_samples: np.ndarray | None = None,
    gene_ids: list[str] | None = None,
    epsilon: float = SIGMA_EPSILON,
) -> dict[str, float]:
    """Resolve sigma per gene from the requested uncertainty source.

    Args:
        source: "cohort_sigma" | "sigma_head" | "mc_variance".
        cohort_residuals: For cohort_sigma.
        sigma_hat: For sigma_head.
        mc_samples: For mc_variance.
        gene_ids: Required for sigma_head and mc_variance.
        epsilon: Minimum sigma.

    Returns:
        Gene ID -> sigma.
    """
    if source == "cohort_sigma":
        if cohort_residuals is None:
            raise ValueError("cohort_sigma requires cohort_residuals.")
        return sigma_from_cohort_residuals(cohort_residuals, epsilon=epsilon)
    if source == "sigma_head":
        if sigma_hat is None or gene_ids is None:
            raise ValueError("sigma_head requires sigma_hat and gene_ids.")
        return sigma_from_sigma_head(sigma_hat, gene_ids, epsilon=epsilon)
    if source == "mc_variance":
        if mc_samples is None or gene_ids is None:
            raise ValueError("mc_variance requires mc_samples and gene_ids.")
        return sigma_from_mc_variance(mc_samples, gene_ids, epsilon=epsilon)
    raise ValueError(
        f"Unsupported uncertainty_source {source!r}. "
        "Use cohort_sigma, sigma_head, or mc_variance."
    )
