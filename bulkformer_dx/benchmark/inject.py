"""Controlled outlier injection for benchmark evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class InjectionResult:
    """Result of outlier injection."""

    expression_perturbed: np.ndarray
    ground_truth_mask: np.ndarray
    injected_sample_idx: list[int]
    injected_gene_idx: list[int]
    directions: list[str]


def inject_outliers_log1p(
    expression: np.ndarray,
    valid_mask: np.ndarray,
    *,
    n_inject: int = 20,
    scale: float = 3.0,
    direction: str = "both",
    seed: int = 0,
) -> InjectionResult:
    """Inject controlled outliers into log1p(TPM) expression matrix.

    Args:
        expression: (n_samples, n_genes) float matrix.
        valid_mask: (n_samples, n_genes) bool - only inject into valid positions.
        n_inject: Number of (sample, gene) pairs to perturb.
        scale: Magnitude of perturbation in residual units (multiples of per-gene std).
        direction: "up", "down", or "both".
        seed: RNG seed.

    Returns:
        InjectionResult with perturbed matrix and ground truth.
    """
    expression = np.asarray(expression, dtype=float).copy()
    valid_mask = np.asarray(valid_mask, dtype=bool)
    rng = np.random.default_rng(seed)
    n_samples, n_genes = expression.shape

    valid_flat = valid_mask.ravel()
    valid_indices = np.flatnonzero(valid_flat)
    if len(valid_indices) < n_inject:
        n_inject = len(valid_indices)
    chosen = rng.choice(valid_indices, size=n_inject, replace=False)
    sample_idx = chosen // n_genes
    gene_idx = chosen % n_genes

    ground_truth = np.zeros((n_samples, n_genes), dtype=bool)
    directions_list: list[str] = []

    per_gene_std = np.nanstd(expression, axis=0)
    per_gene_std = np.where(per_gene_std > 1e-10, per_gene_std, 1.0)

    for i, (s, g) in enumerate(zip(sample_idx, gene_idx, strict=True)):
        delta = scale * per_gene_std[g]
        if direction == "up":
            expression[s, g] += delta
            directions_list.append("over")
        elif direction == "down":
            expression[s, g] -= delta
            expression[s, g] = max(0.0, expression[s, g])
            directions_list.append("under")
        else:
            sign = 1 if rng.random() > 0.5 else -1
            expression[s, g] += sign * delta
            expression[s, g] = max(0.0, expression[s, g])
            directions_list.append("over" if sign > 0 else "under")
        ground_truth[s, g] = True

    return InjectionResult(
        expression_perturbed=expression,
        ground_truth_mask=ground_truth,
        injected_sample_idx=sample_idx.tolist(),
        injected_gene_idx=gene_idx.tolist(),
        directions=directions_list,
    )
