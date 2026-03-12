"""Core data schemas for BulkFormer-DX anomaly and benchmarking pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class AlignedExpressionBundle:
    """Aligned expression data after preprocessing, ready for model inference.

    Attributes:
        expr_space: Expression space: "log1p_tpm", "tpm", or "counts".
        Y_obs: Observed expression matrix (n_samples, n_genes), float32.
        valid_mask: Boolean mask (n_samples, n_genes) for gene presence and usability.
        gene_ids: Ordered list of gene identifiers.
        sample_ids: Ordered list of sample identifiers.
        counts: Raw counts (n_samples, n_genes), optional. Required for NB tests.
        gene_length_kb: Gene lengths in kb (n_genes,), optional. Required for NB mapping.
        tpm_scaling_S: TPM scaling S_j per sample (n_samples,), optional.
        metadata: Optional sample-level covariates.
    """

    expr_space: str
    Y_obs: np.ndarray
    valid_mask: np.ndarray
    gene_ids: list[str]
    sample_ids: list[str]
    counts: np.ndarray | None = None
    gene_length_kb: np.ndarray | None = None
    tpm_scaling_S: np.ndarray | None = None
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class ModelPredictionBundle:
    """BulkFormer inference outputs for downstream scoring and calibration.

    Attributes:
        y_hat: Predicted mean expression (n_samples, n_genes), float32.
        sigma_hat: Optional predicted std (n_samples, n_genes). From sigma head or derived.
        embedding: Optional sample embeddings (n_samples, d) for kNN cohort selection.
        mc_samples: Optional MC samples (n_mc, n_samples, n_genes) from mc_predict.
    """

    y_hat: np.ndarray
    sigma_hat: np.ndarray | None = None
    embedding: np.ndarray | None = None
    mc_samples: np.ndarray | None = None
