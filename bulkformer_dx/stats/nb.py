"""Negative Binomial pmf/cdf and OUTRIDER-style two-sided p-value."""

from __future__ import annotations

import numpy as np
from scipy.stats import nbinom

# NB parameterization: X ~ NB(n, p) with n=size, p=probability
# mean = n * (1-p) / p  =>  p = n / (n + mean)
# size = 1/alpha in mean-dispersion param: Var = mu + alpha*mu^2


def nb_logpmf(k: np.ndarray, mu: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Log P(X = k) for NB with mean mu and size parameter.

    size = 1/alpha where Var = mu + alpha*mu^2.
    Uses scipy nbinom(n, p) with n=size, p = size/(size+mu).

    Args:
        k: Observed counts (int or float, will be floored).
        mu: Mean (must be > 0).
        size: Size parameter (1/dispersion), must be > 0.

    Returns:
        Log pmf, same shape as k.
    """
    k = np.asarray(k, dtype=float)
    mu = np.asarray(mu, dtype=float)
    size = np.asarray(size, dtype=float)
    mu_safe = np.maximum(mu, 1e-10)
    size_safe = np.maximum(size, 1e-10)
    p = size_safe / (size_safe + mu_safe)
    return nbinom.logpmf(np.floor(k).astype(int), n=size_safe, p=p)


def outrider_two_sided_nb_pvalue(
    k: int | float,
    mu: float,
    size: float,
    *,
    mu_epsilon: float = 1e-10,
    size_epsilon: float = 1e-10,
) -> float:
    """OUTRIDER discrete-safe two-sided NB p-value.

    Formula:
        p_le = P(X <= k)
        p_eq = P(X = k)
        p_ge = 1 - p_le + p_eq
        p_2s = 2 * min(0.5, p_le, p_ge)

    Args:
        k: Observed count.
        mu: Expected mean (must be > 0).
        size: NB size parameter (1/dispersion).
        mu_epsilon: Minimum mu for numerical stability.
        size_epsilon: Minimum size for numerical stability.

    Returns:
        Two-sided p-value in [0, 1].
    """
    k_int = int(np.floor(k))
    mu_safe = max(float(mu), mu_epsilon)
    size_safe = max(float(size), size_epsilon)
    p = size_safe / (size_safe + mu_safe)

    p_le = float(nbinom.cdf(k_int, n=size_safe, p=p))
    p_eq = float(nbinom.pmf(k_int, n=size_safe, p=p))
    p_ge = 1.0 - p_le + p_eq

    p_2s = 2.0 * min(0.5, p_le, p_ge)
    return float(np.clip(p_2s, 0.0, 1.0))
