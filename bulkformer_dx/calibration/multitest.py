"""Multiple-testing correction (BH, BY) scoped within-sample."""

from __future__ import annotations

import numpy as np


def benjamini_hochberg(p_values: np.ndarray | list[float]) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction (assumes independence).

    Args:
        p_values: 1D array of p-values in [0, 1].

    Returns:
        Adjusted p-values (q-values), same shape.
    """
    resolved = np.asarray(p_values, dtype=float)
    if resolved.ndim != 1:
        raise ValueError("Benjamini-Hochberg expects a 1D array of p-values.")
    if resolved.size == 0:
        return resolved.copy()
    finite_mask = np.isfinite(resolved)
    if np.any((resolved[finite_mask] < 0.0) | (resolved[finite_mask] > 1.0)):
        raise ValueError("P-values must lie in the interval [0, 1].")

    adjusted = np.full(resolved.shape, np.nan, dtype=float)
    finite_values = resolved[finite_mask]
    if finite_values.size == 0:
        return adjusted

    order = np.argsort(finite_values, kind="mergesort")
    ranked = finite_values[order]
    ranks = np.arange(1, ranked.size + 1, dtype=float)
    raw = ranked * ranked.size / ranks
    monotone = np.minimum.accumulate(raw[::-1])[::-1]
    reordered = np.empty_like(monotone)
    reordered[order] = np.clip(monotone, 0.0, 1.0)
    adjusted[finite_mask] = reordered
    return adjusted


def benjamini_yekutieli(p_values: np.ndarray | list[float]) -> np.ndarray:
    """Apply Benjamini-Yekutieli FDR correction (arbitrary dependence).

    Args:
        p_values: 1D array of p-values in [0, 1].

    Returns:
        Adjusted p-values (q-values), same shape.
    """
    resolved = np.asarray(p_values, dtype=float)
    if resolved.ndim != 1:
        raise ValueError("Benjamini-Yekutieli correction expects a 1D array of p-values.")
    if resolved.size == 0:
        return resolved.copy()
    finite_mask = np.isfinite(resolved)
    if np.any((resolved[finite_mask] < 0.0) | (resolved[finite_mask] > 1.0)):
        raise ValueError("P-values must lie in the interval [0, 1].")

    adjusted = np.full(resolved.shape, np.nan, dtype=float)
    finite_values = resolved[finite_mask]
    if finite_values.size == 0:
        return adjusted

    order = np.argsort(finite_values, kind="mergesort")
    ranked = finite_values[order]
    ranks = np.arange(1, ranked.size + 1, dtype=float)
    harmonic = float(np.sum(1.0 / ranks))
    raw = ranked * ranked.size * harmonic / ranks
    monotone = np.minimum.accumulate(raw[::-1])[::-1]
    reordered = np.empty_like(monotone)
    reordered[order] = np.clip(monotone, 0.0, 1.0)
    adjusted[finite_mask] = reordered
    return adjusted


def apply_within_sample(
    pvalue_matrix: np.ndarray,
    method: str = "BY",
) -> np.ndarray:
    """Apply multiple-testing correction per row (within-sample).

    Args:
        pvalue_matrix: (n_samples, n_tests) p-values.
        method: "BH" | "BY" | "none".

    Returns:
        Adjusted p-values, same shape.
    """
    pvalue_matrix = np.asarray(pvalue_matrix, dtype=float)
    if pvalue_matrix.ndim != 2:
        raise ValueError("pvalue_matrix must be 2D (samples x tests).")
    if method == "none":
        return pvalue_matrix.copy()
    if method == "BH":
        corrector = benjamini_hochberg
    elif method == "BY":
        corrector = benjamini_yekutieli
    else:
        raise ValueError(f"Unsupported method {method!r}. Use BH, BY, or none.")

    adjusted = np.empty_like(pvalue_matrix)
    for i in range(pvalue_matrix.shape[0]):
        adjusted[i] = corrector(pvalue_matrix[i])
    return adjusted
