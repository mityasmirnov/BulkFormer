"""Cohort selection for calibration (global vs kNN local)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bulkformer_dx.cohort import select_global_cohort, select_knn_cohort

if TYPE_CHECKING:
    pass


def get_cohort_indices(
    sample_ids: list[str],
    *,
    cohort_mode: str = "global",
    embedding: np.ndarray | None = None,
    knn_k: int = 50,
) -> dict[str, list[int]]:
    """Get cohort member indices per sample.

    Args:
        sample_ids: Ordered sample identifiers.
        cohort_mode: "global" | "knn_local".
        embedding: (n_samples, d) embeddings. Required for knn_local.
        knn_k: Number of neighbors for knn_local.

    Returns:
        sample_id -> list of cohort member indices (into sample_ids).
    """
    if cohort_mode == "global":
        return select_global_cohort(sample_ids, embedding=embedding)
    if cohort_mode == "knn_local":
        if embedding is None:
            raise ValueError(
                "knn_local cohort_mode requires embeddings. "
                "Provide --embedding-path or run calibration on NLL scoring output "
                "which saves embeddings."
            )
        return select_knn_cohort(
            sample_ids,
            embedding,
            k=knn_k,
            exclude_self=True,
        )
    raise ValueError(
        f"Unsupported cohort_mode {cohort_mode!r}. Use global or knn_local."
    )
