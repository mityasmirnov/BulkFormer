"""Negative Binomial dispersion estimation for OUTRIDER-style count-space tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from bulkformer_dx.stats.nb import nb_logpmf

DISPERSION_EPS = 1e-10
DISPERSION_MAX = 1e4
MEAN_EPS = 1e-8


@dataclass(slots=True)
class DispersionFitResult:
    """Per-gene dispersion fit result."""

    alpha: float
    size: float  # 1/alpha
    converged: bool
    n_obs: int


def fit_nb_dispersion_mle(
    mu: np.ndarray,
    k: np.ndarray,
    *,
    alpha_init: float = 0.1,
    alpha_bounds: tuple[float, float] = (1e-6, DISPERSION_MAX),
) -> DispersionFitResult:
    """Fit NB dispersion (alpha) per gene via MLE.

    Var = mu + alpha * mu^2. Size = 1/alpha.

    Args:
        mu: Expected counts (must be > 0 where k is observed).
        k: Observed counts (non-negative integers).
        alpha_init: Initial guess for optimization.
        alpha_bounds: (min, max) for alpha.

    Returns:
        DispersionFitResult with alpha, size, converged flag.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    k = np.asarray(k, dtype=float).ravel()
    if mu.shape != k.shape:
        raise ValueError("mu and k must have the same shape.")

    valid = (mu > MEAN_EPS) & (k >= 0) & np.isfinite(mu) & np.isfinite(k)
    if valid.sum() < 2:
        return DispersionFitResult(
            alpha=alpha_init,
            size=1.0 / max(alpha_init, DISPERSION_EPS),
            converged=False,
            n_obs=int(valid.sum()),
        )

    mu_v = mu[valid]
    k_v = np.floor(k[valid]).astype(int)

    def neg_loglik(alpha: float) -> float:
        if alpha <= 0:
            return 1e30
        size = 1.0 / max(alpha, DISPERSION_EPS)
        ll = nb_logpmf(k_v, mu_v, np.full_like(mu_v, size))
        return -float(np.sum(ll))

    result = minimize_scalar(
        neg_loglik,
        bounds=alpha_bounds,
        method="bounded",
        options={"xatol": 1e-8},
    )
    alpha_hat = max(float(result.x), DISPERSION_EPS)
    return DispersionFitResult(
        alpha=alpha_hat,
        size=1.0 / alpha_hat,
        converged=result.success,
        n_obs=int(valid.sum()),
    )


def fit_nb_dispersion_moments(
    mu: np.ndarray,
    k: np.ndarray,
) -> DispersionFitResult:
    """Estimate NB dispersion via method of moments.

    Var = mu + alpha * mu^2  =>  alpha = (Var - mu) / mu^2
    """
    mu = np.asarray(mu, dtype=float).ravel()
    k = np.asarray(k, dtype=float).ravel()
    valid = (mu > MEAN_EPS) & (k >= 0) & np.isfinite(mu) & np.isfinite(k)
    if valid.sum() < 2:
        return DispersionFitResult(
            alpha=0.1,
            size=10.0,
            converged=False,
            n_obs=int(valid.sum()),
        )
    mu_v = mu[valid]
    k_v = k[valid]
    var_k = np.var(k_v, ddof=1)
    mean_mu = float(np.mean(mu_v))
    if mean_mu <= 0 or mean_mu**2 < 1e-20:
        return DispersionFitResult(
            alpha=0.1,
            size=10.0,
            converged=False,
            n_obs=int(valid.sum()),
        )
    alpha_raw = (var_k - mean_mu) / (mean_mu**2)
    alpha_hat = np.clip(alpha_raw, DISPERSION_EPS, DISPERSION_MAX)
    return DispersionFitResult(
        alpha=float(alpha_hat),
        size=1.0 / float(alpha_hat),
        converged=True,
        n_obs=int(valid.sum()),
    )


def fit_nb_dispersion_moments_per_gene(
    mu_matrix: np.ndarray,
    k_matrix: np.ndarray,
    gene_indices: np.ndarray | None = None,
) -> dict[int, DispersionFitResult]:
    """Fit dispersion per gene via moments (samples x genes matrices)."""
    n_samples, n_genes = mu_matrix.shape
    if k_matrix.shape != (n_samples, n_genes):
        raise ValueError("mu_matrix and k_matrix must have the same shape.")
    if gene_indices is None:
        gene_indices = np.arange(n_genes)

    results: dict[int, DispersionFitResult] = {}
    for g in gene_indices:
        results[int(g)] = fit_nb_dispersion_moments(
            mu_matrix[:, g],
            k_matrix[:, g],
        )
    return results


def fit_deseq2_trend(
    mean_counts: np.ndarray,
    dispersions: np.ndarray,
    *,
    asympt_disp: float = 0.1,
    extra_pois: float = 0.0,
) -> tuple[float, float]:
    """Fit parametric trend: alpha(mu) = asymptDisp + extraPois / mu.

    Returns (asympt_disp, extra_pois) fitted.
    """
    mean_counts = np.asarray(mean_counts, dtype=float).ravel()
    dispersions = np.asarray(dispersions, dtype=float).ravel()
    valid = (mean_counts > MEAN_EPS) & (dispersions > 0) & np.isfinite(mean_counts)
    if valid.sum() < 2:
        return asympt_disp, extra_pois

    mu = mean_counts[valid]
    alpha = dispersions[valid]
    # alpha = asymptDisp + extraPois / mu
    # Regress alpha on 1/mu: alpha = a + b * (1/mu)
    inv_mu = 1.0 / mu
    inv_mu_mean = np.mean(inv_mu)
    alpha_mean = np.mean(alpha)
    cov = np.mean((alpha - alpha_mean) * (inv_mu - inv_mu_mean))
    var_inv = np.var(inv_mu)
    if var_inv > 1e-20:
        extra_pois = cov / var_inv
        asympt_disp = alpha_mean - extra_pois * inv_mu_mean
    asympt_disp = max(asympt_disp, DISPERSION_EPS)
    extra_pois = max(extra_pois, 0.0)
    return float(asympt_disp), float(extra_pois)


def shrink_dispersion_to_trend(
    alpha: float,
    mean_count: float,
    asympt_disp: float,
    extra_pois: float,
    *,
    prior_weight: float = 0.5,
) -> float:
    """Shrink dispersion toward DESeq2-like trend.

    trend_alpha = asymptDisp + extraPois / mean_count
    shrunk = (1 - prior_weight) * alpha + prior_weight * trend_alpha
    In log-space for stability.
    """
    if mean_count <= 0 or alpha <= 0:
        return max(alpha, DISPERSION_EPS)
    trend_alpha = asympt_disp + extra_pois / max(mean_count, MEAN_EPS)
    trend_alpha = max(trend_alpha, DISPERSION_EPS)
    log_alpha = np.log(alpha)
    log_trend = np.log(trend_alpha)
    shrunk_log = (1 - prior_weight) * log_alpha + prior_weight * log_trend
    return float(np.exp(shrunk_log))
