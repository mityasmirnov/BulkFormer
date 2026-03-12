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


@dataclass(slots=True)
class MethodConfig:
    """Configuration for a single anomaly scoring/test method.

    Used by benchmark harness and scoring plugins. JSON/YAML serializable.
    """

    method_id: str
    space: str  # "log1p_tpm" | "counts"
    cohort_mode: str = "global"  # "global" | "knn_local"
    knn_k: int = 50
    uncertainty_source: str = "cohort_sigma"  # cohort_sigma | sigma_head | mc_variance | nb_dispersion
    distribution_family: str = "gaussian"  # gaussian | student_t | negative_binomial
    test_type: str = "zscore_2s"  # outrider_nb_2s | zscore_2s | empirical_tail | pseudo_likelihood
    multiple_testing: str = "BY"  # BY | BH | none
    alpha: float = 0.05
    mc_passes: int = 16
    mask_rate: float = 0.15
    seed: int = 0
    student_t_df: float = 5.0


@dataclass(slots=True)
class GeneOutlierRow:
    """Single row of gene-level outlier output (long format)."""

    sample_id: str
    gene_id: str
    y_obs: float
    y_hat: float
    residual: float
    score_gene: float
    p_raw: float | None = None
    p_adj: float | None = None
    direction: str | None = None  # "under" | "over"
    method_id: str = ""
    diagnostics_json: dict[str, Any] | None = None


@dataclass(slots=True)
class SampleOutlierRow:
    """Single row of sample-level outlier output."""

    sample_id: str
    score_sample: float
    cohort_mode: str = "global"
    method_id: str = ""
