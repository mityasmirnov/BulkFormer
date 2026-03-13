"""Heterogeneity metrics for kNN-local cohort suitability."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_TISSUE_ENTROPY_THRESHOLD = 0.5
DEFAULT_BATCH_ENTROPY_THRESHOLD = 0.3


def tissue_entropy(sample_tissues: pd.Series) -> float:
    """Compute entropy of tissue distribution across samples.

    Low entropy indicates single-tissue or near-homogeneous cohort,
    where kNN-local may add noise rather than signal.

    Args:
        sample_tissues: Series of tissue labels indexed by sample_id.

    Returns:
        Entropy in nats. 0 = single tissue, higher = more diverse.
    """
    counts = sample_tissues.value_counts()
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs + 1e-10)))


def batch_entropy(sample_batches: pd.Series) -> float:
    """Compute entropy of batch distribution across samples.

    Low entropy indicates single-batch cohort.

    Args:
        sample_batches: Series of batch labels indexed by sample_id.

    Returns:
        Entropy in nats.
    """
    return tissue_entropy(sample_batches)


def suggest_knn_local(
    *,
    tissue_labels: pd.Series | None = None,
    batch_labels: pd.Series | None = None,
    tissue_entropy_threshold: float = DEFAULT_TISSUE_ENTROPY_THRESHOLD,
    batch_entropy_threshold: float = DEFAULT_BATCH_ENTROPY_THRESHOLD,
) -> tuple[bool, str]:
    """Suggest whether kNN-local is appropriate given cohort heterogeneity.

    Args:
        tissue_labels: Optional sample_id -> tissue_label.
        batch_labels: Optional sample_id -> batch_label.
        tissue_entropy_threshold: Min tissue entropy to recommend kNN_local.
        batch_entropy_threshold: Min batch entropy to recommend kNN_local.

    Returns:
        (recommend_knn_local, reason_message).
    """
    if tissue_labels is None and batch_labels is None:
        return True, "No metadata provided; cannot assess heterogeneity."
    reasons = []
    if tissue_labels is not None and len(tissue_labels) > 0:
        te = tissue_entropy(tissue_labels)
        if te < tissue_entropy_threshold:
            reasons.append(
                f"low tissue entropy ({te:.3f} < {tissue_entropy_threshold}); "
                "cohort may be single-tissue"
            )
    if batch_labels is not None and len(batch_labels) > 0:
        be = batch_entropy(batch_labels)
        if be < batch_entropy_threshold:
            reasons.append(
                f"low batch entropy ({be:.3f} < {batch_entropy_threshold}); "
                "cohort may be single-batch"
            )
    if reasons:
        return False, " ".join(reasons)
    return True, "Heterogeneity appears sufficient for kNN-local."
