"""Tests for OUTRIDER-style NB test in count space."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bulkformer_dx.anomaly.nb_test import expected_counts_from_predicted_tpm
from bulkformer_dx.stats.dispersion import fit_nb_dispersion_mle, fit_nb_dispersion_moments
from bulkformer_dx.stats.nb import outrider_two_sided_nb_pvalue


def test_expected_counts_from_predicted_tpm_formula() -> None:
    """Expected count = pred_tpm * (S_j / 1e6) * L_g."""
    pred_tpm = np.array([[100.0, 50.0], [200.0, 100.0]])
    gene_lengths_kb = np.array([2.0, 1.0])
    S_j = np.array([1e6, 2e6])
    mu = expected_counts_from_predicted_tpm(pred_tpm, gene_lengths_kb, S_j)
    # mu[0,0] = 100 * (1e6/1e6) * 2 = 200
    assert np.isclose(mu[0, 0], 200.0)
    # mu[1,1] = 100 * (2e6/1e6) * 1 = 200
    assert np.isclose(mu[1, 1], 200.0)
    # mu[0,1] = 50 * 1 * 1 = 50
    assert np.isclose(mu[0, 1], 50.0)


def test_expected_counts_from_predicted_tpm_shape() -> None:
    """Output shape matches input."""
    pred_tpm = np.random.rand(5, 10)
    gene_lengths_kb = np.ones(10)
    S_j = np.ones(5) * 1e6
    mu = expected_counts_from_predicted_tpm(pred_tpm, gene_lengths_kb, S_j)
    assert mu.shape == (5, 10)


def test_outrider_two_sided_nb_pvalue_discrete_safe() -> None:
    """OUTRIDER formula: p_2s = 2 * min(0.5, p_le, p_ge), clamped to [0,1]."""
    # At k=mu, p-value should be high (not extreme)
    p = outrider_two_sided_nb_pvalue(k=10, mu=10.0, size=5.0)
    assert 0 <= p <= 1
    assert p > 0.3  # symmetric case

    # Extreme low count
    p_low = outrider_two_sided_nb_pvalue(k=0, mu=10.0, size=5.0)
    assert 0 <= p_low <= 1
    assert p_low < 0.5

    # Extreme high count
    p_high = outrider_two_sided_nb_pvalue(k=100, mu=10.0, size=5.0)
    assert 0 <= p_high <= 1
    assert p_high < 0.5


def test_outrider_pvalue_clamps_at_half() -> None:
    """Formula clamps min(0.5, p_le, p_ge) so p_2s <= 1."""
    p = outrider_two_sided_nb_pvalue(k=5, mu=5.0, size=10.0)
    assert p <= 1.0


def test_fit_nb_dispersion_mle() -> None:
    """MLE dispersion fit returns valid alpha and size."""
    mu = np.array([10.0, 20.0, 15.0, 12.0])
    k = np.array([8, 22, 14, 11])
    res = fit_nb_dispersion_mle(mu, k)
    assert res.alpha > 0
    assert res.size > 0
    assert res.size == pytest.approx(1.0 / res.alpha)
    assert res.n_obs == 4


def test_fit_nb_dispersion_moments() -> None:
    """Moments-based dispersion returns valid result."""
    mu = np.array([10.0, 20.0, 15.0, 12.0])
    k = np.array([8, 22, 14, 11])
    res = fit_nb_dispersion_moments(mu, k)
    assert res.alpha > 0
    assert res.size > 0


def test_nb_pvalues_reasonable_under_null() -> None:
    """Under null (k ~ NB(mu, size)), p-values should be in [0,1] and not degenerate."""
    np.random.seed(42)
    n = 200
    mu = 20.0
    size = 5.0
    k_vals = np.random.negative_binomial(n=size, p=size / (size + mu), size=n)
    p_vals = np.array(
        [outrider_two_sided_nb_pvalue(int(k), mu, size) for k in k_vals]
    )
    assert np.all((p_vals >= 0) & (p_vals <= 1))
    # Mean should be near 0.5 for two-sided under null (discrete can deviate)
    assert 0.3 < np.mean(p_vals) < 0.7


def test_compute_nb_outrider_for_calibration_requires_count_space_path(
    tmp_path: Path,
) -> None:
    """nb_outrider calibration requires count-space artifacts."""
    from bulkformer_dx.anomaly.nb_test import compute_nb_outrider_for_calibration

    ranked = {
        "s1": pd.DataFrame({
            "ensg_id": ["ENSG1", "ENSG2"],
            "anomaly_score": [1.0, 2.0],
            "mean_signed_residual": [0.1, -0.2],
            "rmse": [0.5, 0.6],
            "masked_count": [4, 4],
            "coverage_fraction": [1.0, 1.0],
            "observed_expression": [3.0, 4.0],
            "mean_predicted_expression": [2.8, 4.2],
        }),
        "s2": pd.DataFrame({
            "ensg_id": ["ENSG1", "ENSG2"],
            "anomaly_score": [1.5, 1.8],
            "mean_signed_residual": [0.2, -0.1],
            "rmse": [0.5, 0.5],
            "masked_count": [4, 4],
            "coverage_fraction": [1.0, 1.0],
            "observed_expression": [2.5, 3.8],
            "mean_predicted_expression": [2.6, 3.9],
        }),
    }

    with pytest.raises(FileNotFoundError, match="aligned_counts"):
        compute_nb_outrider_for_calibration(ranked, tmp_path)


def test_compute_nb_outrider_for_calibration_integration(tmp_path: Path) -> None:
    """Full nb_outrider calibration with mock count-space artifacts."""
    from bulkformer_dx.anomaly.nb_test import compute_nb_outrider_for_calibration

    # Create aligned_counts (samples x genes)
    counts_df = pd.DataFrame(
        {
            "ENSG1": [50, 60],
            "ENSG2": [100, 90],
        },
        index=["s1", "s2"],
    )
    counts_df.index.name = "sample_id"
    counts_df.to_csv(tmp_path / "aligned_counts.tsv", sep="\t")

    # gene_lengths_aligned
    pd.DataFrame({
        "ensg_id": ["ENSG1", "ENSG2"],
        "length_kb": [2.0, 1.5],
        "has_length": [1, 1],
    }).to_csv(tmp_path / "gene_lengths_aligned.tsv", sep="\t", index=False)

    # sample_scaling
    pd.DataFrame({
        "sample_id": ["s1", "s2"],
        "S_j": [1e6, 1.2e6],
    }).to_csv(tmp_path / "sample_scaling.tsv", sep="\t", index=False)

    ranked = {
        "s1": pd.DataFrame({
            "ensg_id": ["ENSG1", "ENSG2"],
            "anomaly_score": [1.0, 2.0],
            "mean_signed_residual": [0.1, -0.2],
            "rmse": [0.5, 0.6],
            "masked_count": [4, 4],
            "coverage_fraction": [1.0, 1.0],
            "observed_expression": [3.0, 4.0],
            "mean_predicted_expression": [2.8, 4.2],
        }),
        "s2": pd.DataFrame({
            "ensg_id": ["ENSG1", "ENSG2"],
            "anomaly_score": [1.5, 1.8],
            "mean_signed_residual": [0.2, -0.1],
            "rmse": [0.5, 0.5],
            "masked_count": [4, 4],
            "coverage_fraction": [1.0, 1.0],
            "observed_expression": [2.5, 3.8],
            "mean_predicted_expression": [2.6, 3.9],
        }),
    }

    result = compute_nb_outrider_for_calibration(ranked, tmp_path)

    assert "s1" in result
    assert "s2" in result
    assert "nb_outrider_p_raw" in result["s1"].columns
    assert "nb_outrider_p_adj" in result["s1"].columns
    assert "nb_outrider_direction" in result["s1"].columns
    assert "nb_outrider_expected_count" in result["s1"].columns
    for sample_id in ["s1", "s2"]:
        df = result[sample_id]
        assert np.all((df["nb_outrider_p_raw"] >= 0) | df["nb_outrider_p_raw"].isna())
        assert np.all((df["nb_outrider_p_raw"] <= 1) | df["nb_outrider_p_raw"].isna())
