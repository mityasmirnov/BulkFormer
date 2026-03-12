"""Gaussian and Student-t log-PDF utilities for log1p(TPM) space."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, t as student_t_dist

DEFAULT_STUDENT_T_DF = 5.0


def gaussian_logpdf(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Compute log p(y | mu, sigma) for Gaussian N(mu, sigma^2).

    Args:
        y: Observed values.
        mu: Mean.
        sigma: Standard deviation.
        epsilon: Minimum sigma for numerical stability.

    Returns:
        Log probability density, same shape as y.
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma_safe = np.maximum(sigma, epsilon)
    return norm.logpdf(y, loc=mu, scale=sigma_safe)


def student_t_logpdf(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    df: float = DEFAULT_STUDENT_T_DF,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Compute log p(y | mu, sigma, df) for scaled Student-t.

    Uses location mu and scale sigma. Heavier tails than Gaussian;
    often better calibrated for omics residuals.

    Args:
        y: Observed values.
        mu: Location.
        sigma: Scale.
        df: Degrees of freedom.
        epsilon: Minimum sigma for numerical stability.

    Returns:
        Log probability density, same shape as y.
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma_safe = np.maximum(sigma, epsilon)
    z = (y - mu) / sigma_safe
    return student_t_dist.logpdf(z, df=df) - np.log(sigma_safe)
