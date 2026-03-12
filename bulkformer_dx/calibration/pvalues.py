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


def compute_stratified_calibration(
    p_values: np.ndarray,
    strata_labels: np.ndarray,
    *,
    n_bins_if_continuous: int = 5,
) -> dict[str, np.ndarray]:
    """Group p-values by strata labels and return dict of label -> p-values.

    If strata_labels are numeric with many unique values, they are binned into
    quantile-based buckets. Otherwise labels are used directly.

    Args:
        p_values: 1-D array of p-values.
        strata_labels: 1-D array of labels (same length as p_values).
        n_bins_if_continuous: Number of quantile bins for continuous labels.

    Returns:
        Dict mapping stratum name -> array of p-values for that stratum.
    """
    p = np.asarray(p_values, dtype=float).ravel()
    labels = np.asarray(strata_labels).ravel()
    if p.shape != labels.shape:
        raise ValueError("p_values and strata_labels must have the same length.")

    finite_mask = np.isfinite(p) & (p >= 0) & (p <= 1)
    p = p[finite_mask]
    labels = labels[finite_mask]

    # Decide whether to bin numeric labels
    try:
        num_labels = labels.astype(float)
        n_unique = len(np.unique(num_labels[np.isfinite(num_labels)]))
        if n_unique > n_bins_if_continuous * 2:
            # Quantile-bin
            edges = np.nanquantile(num_labels, np.linspace(0, 1, n_bins_if_continuous + 1))
            edges = np.unique(edges)
            bin_idx = np.digitize(num_labels, edges[1:-1])
            result: dict[str, np.ndarray] = {}
            for b in range(len(edges) - 1):
                mask = bin_idx == b
                if mask.sum() >= 2:
                    lo, hi = edges[b], edges[b + 1]
                    result[f"{lo:.2g}–{hi:.2g}"] = p[mask]
            return result
    except (ValueError, TypeError):
        pass

    # Categorical labels
    result = {}
    for label in sorted(set(labels)):
        mask = labels == label
        if mask.sum() >= 2:
            result[str(label)] = p[mask]
    return result
