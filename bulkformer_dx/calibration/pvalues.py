"""Centralized p-value computation for anomaly scoring and calibration."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, t as student_t


def empirical_tail_pvalue(
    distribution: np.ndarray,
    observed_value: float,
    *,
    upper_tail: bool = True,
) -> float:
    """Estimate empirical p-value against a cohort background distribution.

    Uses a +1 pseudo-count so finite p-values remain available for small cohorts.
    The current sample should not be part of its own reference (leave-one-out style).

    Args:
        distribution: Array of cohort values (e.g. anomaly scores for other samples).
        observed_value: Value for the sample being tested.
        upper_tail: If True, p = P(X >= observed); else p = P(X <= observed).

    Returns:
        Empirical p-value in [0, 1].
    """
    finite = np.asarray(distribution, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Empirical calibration requires at least one finite cohort score.")
    if upper_tail:
        exceedances = float(np.count_nonzero(finite >= observed_value))
    else:
        exceedances = float(np.count_nonzero(finite <= observed_value))
    return (exceedances + 1.0) / (finite.size + 1.0)


def zscore_two_sided_pvalue(
    z_scores: np.ndarray,
    *,
    use_student_t: bool = False,
    student_t_df: float = 5.0,
) -> np.ndarray:
    """Compute two-sided p-values from z-scores.

    p = 2 * P(|Z| >= |z|) under the chosen distribution.

    Args:
        z_scores: Z-scores (any shape).
        use_student_t: If True, use Student-t distribution (heavier tails).
        student_t_df: Degrees of freedom for Student-t.

    Returns:
        P-values with same shape as z_scores.
    """
    z = np.asarray(z_scores, dtype=float)
    abs_z = np.abs(z)
    if use_student_t:
        if student_t_df <= 0:
            raise ValueError("student_t_df must be positive when use_student_t is True.")
        p = 2.0 * student_t.sf(abs_z, df=student_t_df)
    else:
        p = 2.0 * norm.sf(abs_z)
    return np.clip(p, 0.0, 1.0)
