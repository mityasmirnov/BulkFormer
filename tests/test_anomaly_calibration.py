"""Tests for anomaly calibration workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bulkformer_dx.anomaly import calibration
from bulkformer_dx.cli import build_parser


def _make_ranked_table(
    anomaly_score: float,
    observed_expression: float,
    mean_predicted_expression: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ensg_id": ["ENSG1", "ENSG2"],
            "anomaly_score": [anomaly_score, 1.0],
            "mean_signed_residual": [anomaly_score, -1.0],
            "rmse": [anomaly_score, 1.0],
            "masked_count": [4, 4],
            "coverage_fraction": [1.0, 1.0],
            "observed_expression": [observed_expression, 2.0],
            "mean_predicted_expression": [mean_predicted_expression, 2.5],
        }
    )


def test_benjamini_yekutieli_matches_known_values() -> None:
    p_values = np.array([0.01, 0.02, 0.5], dtype=float)

    adjusted = calibration.benjamini_yekutieli(p_values)

    assert np.allclose(adjusted, [0.055, 0.055, 0.9166666667])


def test_calibrate_ranked_gene_scores_adds_empirical_pvalues() -> None:
    ranked_gene_scores = {
        "sample_a": _make_ranked_table(10.0, 5.0, 1.0),
        "sample_b": _make_ranked_table(4.0, 2.0, 1.8),
        "sample_c": _make_ranked_table(3.0, 2.0, 1.9),
        "sample_d": _make_ranked_table(2.0, 2.0, 1.7),
        "sample_e": _make_ranked_table(1.0, 2.0, 1.6),
    }

    result = calibration.calibrate_ranked_gene_scores(ranked_gene_scores)
    sample_a = result.calibrated_ranked_gene_scores["sample_a"]

    assert "empirical_p_value" in sample_a.columns
    assert "by_q_value" in sample_a.columns
    assert np.isclose(sample_a.loc[0, "empirical_p_value"], 0.2)
    assert np.isclose(sample_a.loc[0, "by_q_value"], 0.6)
    assert np.isclose(sample_a.loc[1, "empirical_p_value"], 1.0)
    assert sample_a.loc[0, "empirical_p_value"] < sample_a.loc[1, "empirical_p_value"]


def test_calibrate_ranked_gene_scores_can_add_nb_approximation() -> None:
    ranked_gene_scores = {
        "sample_a": _make_ranked_table(10.0, 5.0, 1.0),
        "sample_b": _make_ranked_table(4.0, 2.0, 1.8),
        "sample_c": _make_ranked_table(3.0, 2.0, 1.9),
        "sample_d": _make_ranked_table(2.0, 2.0, 1.7),
        "sample_e": _make_ranked_table(1.0, 2.0, 1.6),
    }

    result = calibration.calibrate_ranked_gene_scores(
        ranked_gene_scores,
        count_space_method="nb_approx",
    )
    sample_a = result.calibrated_ranked_gene_scores["sample_a"]

    assert "nb_approx_p_value" in sample_a.columns
    assert "nb_approx_two_sided_p_value" in sample_a.columns
    assert 0.0 <= sample_a.loc[0, "nb_approx_p_value"] <= 1.0
    assert 0.0 <= sample_a.loc[0, "nb_approx_two_sided_p_value"] <= 1.0
    assert (
        sample_a.loc[0, "nb_approx_two_sided_p_value"]
        < sample_a.loc[1, "nb_approx_two_sided_p_value"]
    )


def test_calibrate_ranked_gene_scores_requires_multiple_samples() -> None:
    with pytest.raises(ValueError, match="at least two samples"):
        calibration.calibrate_ranked_gene_scores({"sample_a": _make_ranked_table(10.0, 5.0, 1.0)})


def test_calibrate_ranked_gene_scores_rejects_non_finite_values() -> None:
    ranked_gene_scores = {
        "sample_a": _make_ranked_table(10.0, 5.0, 1.0),
        "sample_b": _make_ranked_table(4.0, 2.0, 1.8),
    }
    ranked_gene_scores["sample_b"].loc[0, "anomaly_score"] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        calibration.calibrate_ranked_gene_scores(ranked_gene_scores)


def test_calibrate_parser_defaults_to_empirical_path() -> None:
    parser = build_parser()

    args = parser.parse_args(
        ["anomaly", "calibrate", "--scores", "scores_dir", "--output-dir", "out_dir"]
    )

    assert args.count_space_method == "none"


def test_run_writes_calibration_outputs(tmp_path: Path) -> None:
    scores_dir = tmp_path / "scores"
    ranked_dir = scores_dir / "ranked_genes"
    ranked_dir.mkdir(parents=True)

    for sample_id, anomaly_score, observed, predicted in [
        ("sample_a", 10.0, 5.0, 1.0),
        ("sample_b", 4.0, 2.0, 1.8),
        ("sample_c", 3.0, 2.0, 1.9),
        ("sample_d", 2.0, 2.0, 1.7),
        ("sample_e", 1.0, 2.0, 1.6),
    ]:
        _make_ranked_table(anomaly_score, observed, predicted).to_csv(
            ranked_dir / f"{sample_id}.tsv",
            sep="\t",
            index=False,
        )

    output_dir = tmp_path / "calibrated"
    exit_code = calibration.run(
        argparse.Namespace(
            scores=str(scores_dir),
            output_dir=str(output_dir),
            count_space_method="nb_approx",
        )
    )

    assert exit_code == 0
    assert (output_dir / "ranked_genes" / "sample_a.tsv").exists()
    assert (output_dir / "calibration_summary.tsv").exists()
    metadata = json.loads((output_dir / "calibration_run.json").read_text(encoding="utf-8"))
    assert metadata["count_space_method"] == "nb_approx"
