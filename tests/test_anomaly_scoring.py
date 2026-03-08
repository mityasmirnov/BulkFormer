"""Tests for Monte Carlo masking anomaly scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bulkformer_dx.anomaly.scoring import (
    MASK_TOKEN_VALUE,
    generate_mc_mask_plan,
    score_expression_anomalies,
)


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
