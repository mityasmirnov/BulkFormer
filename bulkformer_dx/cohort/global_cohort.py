"""Global cohort: use all samples."""

from __future__ import annotations

import numpy as np


def select_global_cohort(
    sample_ids: list[str],
    embedding: np.ndarray | None = None,
) -> dict[str, list[int]]:
    """Select global cohort: each sample uses all others as reference.

    Args:
        sample_ids: Ordered sample identifiers.
        embedding: Optional (n_samples, d) embeddings; ignored for global.

    Returns:
        sample_id -> list of indices of cohort members (all indices except self).
    """
    n = len(sample_ids)
    return {sample_ids[i]: [j for j in range(n) if j != i] for i in range(n)}
