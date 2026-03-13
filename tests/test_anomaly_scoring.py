"""Tests for Monte Carlo masking anomaly scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bulkformer_dx.anomaly.scoring import (
    MASK_TOKEN_VALUE,
    generate_deterministic_mask_plan,
    generate_mc_mask_plan,
    score_expression_anomalies,
)


def test_generate_deterministic_mask_plan_guarantees_K_target_coverage() -> None:
    """Deterministic plan guarantees min(masked_count) >= K_target for all valid genes."""
    valid_gene_flags = np.array([True, False, True, True, True], dtype=bool)
    mask_plan = generate_deterministic_mask_plan(
        valid_gene_flags,
        sample_count=2,
        K_target=5,
        mask_prob=0.5,
        seed=42,
    )
    n_valid = 4
    n_genes = 5
    m = max(1, int(np.ceil(n_valid * 0.5)))
    expected_passes = max(1, int(np.ceil(5 * n_valid / m)))
    assert mask_plan.shape == (2, expected_passes, n_genes)
    assert not mask_plan[:, :, 1].any()
    masked_counts = mask_plan.sum(axis=1)
    for s in range(2):
        for g in [0, 2, 3, 4]:
            assert masked_counts[s, g] >= 5, f"sample {s} gene {g} has {masked_counts[s, g]} < 5"
    assert mask_plan.sum(axis=2).min() >= 1


def test_generate_deterministic_mask_plan_reproducible() -> None:
    """Same seed produces identical mask plans."""
    valid = np.ones(100, dtype=bool)
    p1 = generate_deterministic_mask_plan(valid, sample_count=3, K_target=5, mask_prob=0.10, seed=0)
    p2 = generate_deterministic_mask_plan(valid, sample_count=3, K_target=5, mask_prob=0.10, seed=0)
    np.testing.assert_array_equal(p1, p2)


def test_generate_mc_mask_plan_masks_only_valid_genes() -> None:
    valid_gene_flags = np.array([True, False, True, True], dtype=bool)
    mask_plan = generate_mc_mask_plan(
        valid_gene_flags,
        sample_count=3,
        mc_passes=4,
        mask_prob=0.4,
        rng=np.random.default_rng(123),
    )

    assert mask_plan.shape == (3, 4, 4)
    assert not mask_plan[:, :, 1].any()
    assert np.all(mask_plan.sum(axis=2) == 2)
    assert np.all(mask_plan[:, :, [0, 2, 3]].sum(axis=2) == 2)


def test_score_expression_anomalies_aggregates_masked_residuals() -> None:
    expression = pd.DataFrame(
        [[10.0, 20.0, MASK_TOKEN_VALUE]],
        index=["sample_a"],
        columns=["ENSG1", "ENSG2", "ENSG3"],
    )
    valid_gene_mask = pd.DataFrame(
        {
            "ensg_id": ["ENSG1", "ENSG2", "ENSG3"],
            "is_valid": [1, 1, 0],
        }
    )
    mask_plan = np.array(
        [
            [
                [True, False, False],
                [True, False, False],
                [False, True, False],
            ]
        ],
        dtype=bool,
    )

    def predictor(masked_expression: np.ndarray, mask_fraction: float) -> np.ndarray:
        del mask_fraction
        predicted = masked_expression.copy()
        predicted[predicted == MASK_TOKEN_VALUE] = 0.0
        predicted[masked_expression[:, 0] == MASK_TOKEN_VALUE, 0] = 8.0
        predicted[masked_expression[:, 1] == MASK_TOKEN_VALUE, 1] = 23.0
        return predicted

    result = score_expression_anomalies(
        expression,
        valid_gene_mask,
        predictor=predictor,
        mc_passes=3,
        mask_prob=0.5,
        mask_plan=mask_plan,
    )

    cohort_row = result.cohort_scores.loc["sample_a"]
    ranked = result.ranked_gene_scores["sample_a"]
    gene_qc = result.gene_qc.set_index("ensg_id")

    assert cohort_row["masked_observations"] == 3
    assert cohort_row["genes_scored"] == 2
    assert np.isclose(cohort_row["gene_coverage_fraction"], 1.0)
    assert np.isclose(cohort_row["mean_abs_residual"], 7.0 / 3.0)

    assert ranked["ensg_id"].tolist() == ["ENSG2", "ENSG1"]
    assert ranked["masked_count"].tolist() == [1, 2]
    assert np.allclose(ranked["anomaly_score"], [3.0, 2.0])
    assert np.allclose(ranked["mean_signed_residual"], [-3.0, 2.0])

    assert gene_qc.loc["ENSG1", "masked_count"] == 2
    assert gene_qc.loc["ENSG2", "masked_count"] == 1
    assert np.isclose(gene_qc.loc["ENSG1", "mean_abs_residual"], 2.0)
    assert np.isclose(gene_qc.loc["ENSG2", "mean_abs_residual"], 3.0)


def test_score_expression_anomalies_passes_row_consistent_mask_fraction() -> None:
    expression = pd.DataFrame(
        [[10.0, 20.0, 30.0, MASK_TOKEN_VALUE]],
        index=["sample_a"],
        columns=["ENSG1", "ENSG2", "ENSG3", "ENSG4"],
    )
    valid_gene_mask = pd.DataFrame(
        {
            "ensg_id": ["ENSG1", "ENSG2", "ENSG3", "ENSG4"],
            "is_valid": [1, 1, 1, 0],
        }
    )
    mask_plan = np.array(
        [
            [
                [True, False, False, False],
                [True, True, False, False],
            ]
        ],
        dtype=bool,
    )

    def predictor(masked_expression: np.ndarray, mask_fraction: float) -> np.ndarray:
        actual_mask_fractions = (masked_expression == MASK_TOKEN_VALUE).mean(axis=1)
        assert np.allclose(actual_mask_fractions, mask_fraction)
        predicted = masked_expression.copy()
        predicted[predicted == MASK_TOKEN_VALUE] = 0.0
        return predicted

    score_expression_anomalies(
        expression,
        valid_gene_mask,
        predictor=predictor,
        mc_passes=2,
        mask_prob=0.5,
        mask_plan=mask_plan,
    )


def test_deterministic_nll_min_masked_count_ge_K_target() -> None:
    """With deterministic mask, all scored genes have masked_count >= K_target."""
    from bulkformer_dx.io.schemas import (
        AlignedExpressionBundle,
        MethodConfig,
        ModelPredictionBundle,
    )
    from bulkformer_dx.scoring.pseudolikelihood import compute_mc_masked_loglikelihood_scores

    n_samples, n_genes = 2, 8
    K_target = 5
    mask_prob = 0.5
    n_valid = n_genes
    m = max(1, int(np.ceil(n_valid * mask_prob)))
    mc_passes = max(1, int(np.ceil(K_target * n_valid / m)))
    Y = np.random.randn(n_samples, n_genes).astype(np.float32) + 5.0
    valid_mask = np.ones((n_samples, n_genes), dtype=bool)
    gene_ids = [f"ENSG{i}" for i in range(n_genes)]
    sample_ids = [f"s{i}" for i in range(n_samples)]
    bundle = AlignedExpressionBundle(
        expr_space="log1p_tpm",
        Y_obs=Y,
        valid_mask=valid_mask,
        gene_ids=gene_ids,
        sample_ids=sample_ids,
        counts=None,
        gene_length_kb=None,
        tpm_scaling_S=None,
        metadata=None,
    )
    mc_samples = np.random.randn(mc_passes, n_samples, n_genes).astype(np.float32) + 5.0
    preds = ModelPredictionBundle(
        y_hat=Y,
        sigma_hat=None,
        embedding=None,
        mc_samples=mc_samples,
    )
    config = MethodConfig(
        method_id="nll_det",
        space="log1p_tpm",
        mc_passes=mc_passes,
        mask_rate=mask_prob,
        mask_schedule="deterministic",
        K_target=K_target,
        seed=0,
        distribution_family="gaussian",
        uncertainty_source="cohort_sigma",
    )
    ranked, cohort = compute_mc_masked_loglikelihood_scores(
        bundle, preds, config=config
    )
    for sample_id, df in ranked.items():
        if df.empty:
            continue
        masked_counts = df["diagnostics_json"].apply(
            lambda x: x.get("masked_count", 0) if isinstance(x, dict) else 0
        )
        assert (masked_counts >= K_target).all(), (
            f"sample {sample_id}: min masked_count {int(masked_counts.min())} < K_target {K_target}"
        )


def test_finite_anomaly_scores() -> None:
    """Test that assert_finite_scores catches NaNs and score_expression_anomalies produces finite scores."""
    from bulkformer_dx.anomaly.scoring import assert_finite_scores

    # 1. Normal finite case
    ranked = {
        "s1": pd.DataFrame({"ensg_id": ["G1"], "anomaly_score": [1.0]}),
        "s2": pd.DataFrame({"ensg_id": ["G1"], "anomaly_score": [2.0]}),
    }
    assert_finite_scores(ranked)  # Should not raise

    # 2. Non-finite case
    ranked_bad = {
        "s1": pd.DataFrame({"ensg_id": ["G1"], "anomaly_score": [np.nan]}),
    }
    import pytest
    with pytest.raises(ValueError, match="Non-finite anomaly_score detected"):
        assert_finite_scores(ranked_bad)

    # 3. Check name variations (score_gene)
    ranked_alt = {
        "s1": pd.DataFrame({"gene_id": ["G1"], "score_gene": [np.inf]}),
    }
    with pytest.raises(ValueError, match="Non-finite anomaly_score detected"):
        assert_finite_scores(ranked_alt)
