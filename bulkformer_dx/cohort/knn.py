"""kNN local cohort selection in embedding space."""

from __future__ import annotations

import numpy as np


def select_knn_cohort(
    sample_ids: list[str],
    embedding: np.ndarray,
    *,
    k: int = 50,
    exclude_self: bool = True,
) -> dict[str, list[int]]:
    """Select k nearest neighbors per sample in embedding space.

    Args:
        sample_ids: Ordered sample identifiers.
        embedding: (n_samples, d) sample embeddings.
        k: Number of neighbors.
        exclude_self: If True, exclude the target sample from its cohort.

    Returns:
        sample_id -> list of indices of cohort members (k nearest neighbors).
    """
    embedding = np.asarray(embedding, dtype=float)
    if embedding.ndim != 2:
        raise ValueError("embedding must be 2D (n_samples, d).")
    n_samples = embedding.shape[0]
    if len(sample_ids) != n_samples:
        raise ValueError(
            f"sample_ids length {len(sample_ids)} != embedding rows {n_samples}."
        )
    if k >= n_samples and exclude_self:
        k = max(1, n_samples - 1)
    elif k >= n_samples:
        k = n_samples

    norms = np.sum(embedding**2, axis=1, keepdims=True)
    dist_sq = norms + norms.T - 2 * (embedding @ embedding.T)
    np.fill_diagonal(dist_sq, np.inf)

    result: dict[str, list[int]] = {}
    for i in range(n_samples):
        if exclude_self:
            neighbors = np.argsort(dist_sq[i])[:k].tolist()
        else:
            idx = np.argsort(dist_sq[i])[: k + 1]
            neighbors = idx.tolist()
        result[sample_ids[i]] = neighbors
    return result
