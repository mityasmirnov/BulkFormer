"""Synthetic high-signal tests for normalized absolute outlier detection."""

from __future__ import annotations

import numpy as np
import torch

from bulkformer_dx.anomaly.calibration import compute_normalized_outliers


def test_synthetic_outlier_detection() -> None:
    """Injected +/-6 sigma genes should be recovered with strong precision and recall."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    num_genes = 10_000
    num_outliers = 50
    mu = torch.rand(1, num_genes, dtype=torch.float64) * 10.0
    sigma = torch.rand(1, num_genes, dtype=torch.float64) * 0.9 + 0.1
    observed = torch.normal(mean=mu, std=sigma)

    outlier_indices = rng.choice(num_genes, size=num_outliers, replace=False)
    direction = torch.where(
        torch.randn(num_outliers, dtype=torch.float64) >= 0,
        torch.tensor(1.0, dtype=torch.float64),
        torch.tensor(-1.0, dtype=torch.float64),
    )
    observed[0, outlier_indices] = mu[0, outlier_indices] + direction * 6.0 * sigma[0, outlier_indices]

    gene_names = [f"GENE_{idx}" for idx in range(num_genes)]
    sample_names = ["SAMPLE_1"]
    results = compute_normalized_outliers(
        observed_log1p_tpm=observed,
        expected_mu=mu,
        expected_sigma=sigma,
        gene_names=gene_names,
        sample_names=sample_names,
    )

    injected_genes = {f"GENE_{idx}" for idx in outlier_indices.tolist()}
    caught_genes = set(results.loc[results["is_significant"], "gene"].tolist())

    true_positives = len(injected_genes & caught_genes)
    false_positives = len(caught_genes - injected_genes)

    assert (true_positives / num_outliers) >= 0.95
    assert false_positives < 10

    first_injected_gene = f"GENE_{int(outlier_indices[0])}"
    test_z = float(results.loc[results["gene"] == first_injected_gene, "z_score"].iloc[0])
    assert abs(abs(test_z) - 6.0) < 0.5
