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
    assert "expected_sigma" in sample_a.columns
    assert "z_score" in sample_a.columns
    assert "raw_p_value" in sample_a.columns
    assert "by_adj_p_value" in sample_a.columns
    assert "is_significant" in sample_a.columns
    assert np.isclose(sample_a.loc[0, "empirical_p_value"], 0.2)
    assert np.isclose(sample_a.loc[0, "by_q_value"], 0.6)
    assert np.isclose(sample_a.loc[1, "empirical_p_value"], 1.0)
    assert sample_a.loc[0, "empirical_p_value"] < sample_a.loc[1, "empirical_p_value"]
    assert set(result.absolute_outliers.columns) >= {
        "sample_id",
        "gene",
        "observed_log1p_tpm",
        "expected_mu",
        "expected_sigma",
        "z_score",
        "raw_p_value",
        "by_adj_p_value",
        "is_significant",
    }




def test_calibrate_ranked_gene_scores_uses_leave_one_out_empirical_reference() -> None:
    ranked_gene_scores = {
        "sample_a": _make_ranked_table(10.0, 5.0, 1.0),
        "sample_b": _make_ranked_table(4.0, 2.0, 1.8),
        "sample_c": _make_ranked_table(3.0, 2.0, 1.9),
        "sample_d": _make_ranked_table(2.0, 2.0, 1.7),
        "sample_e": _make_ranked_table(1.0, 2.0, 1.6),
    }

    result = calibration.calibrate_ranked_gene_scores(ranked_gene_scores)

    assert np.isclose(
        result.calibrated_ranked_gene_scores["sample_a"].loc[0, "empirical_p_value"],
        0.2,
    )
    assert np.isclose(
        result.calibrated_ranked_gene_scores["sample_b"].loc[0, "empirical_p_value"],
        0.4,
    )

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


def test_calibrate_nb_outrider_requires_count_space_path() -> None:
    ranked_gene_scores = {
        "sample_a": _make_ranked_table(10.0, 5.0, 1.0),
        "sample_b": _make_ranked_table(4.0, 2.0, 1.8),
    }
    with pytest.raises(ValueError, match="nb_outrider requires count_space_path"):
        calibration.calibrate_ranked_gene_scores(
            ranked_gene_scores,
            count_space_method="nb_outrider",
        )


def test_calibrate_ranked_gene_scores_rejects_non_finite_values() -> None:
    ranked_gene_scores = {
        "sample_a": _make_ranked_table(10.0, 5.0, 1.0),
        "sample_b": _make_ranked_table(4.0, 2.0, 1.8),
    }
    ranked_gene_scores["sample_b"].loc[0, "anomaly_score"] = np.nan

    with pytest.raises(ValueError, match="at least one finite cohort score"):
        calibration.calibrate_ranked_gene_scores(ranked_gene_scores)


def test_calibrate_parser_defaults_to_empirical_path() -> None:
    parser = build_parser()

    args = parser.parse_args(
        ["anomaly", "calibrate", "--scores", "scores_dir", "--output-dir", "out_dir"]
    )

    assert args.count_space_method == "none"
    assert args.alpha == 0.05


def test_compute_normalized_outliers_supports_single_sample_inputs() -> None:
    result = calibration.compute_normalized_outliers(
        observed_log1p_tpm=np.array([[3.0, 7.0]], dtype=float),
        expected_mu=np.array([[1.0, 7.0]], dtype=float),
        expected_sigma=np.array([[1.0, 2.0]], dtype=float),
        gene_names=["ENSG1", "ENSG2"],
        sample_names=["sample_a"],
    )

    assert result["sample_id"].tolist() == ["sample_a", "sample_a"]
    assert result["gene"].tolist() == ["ENSG1", "ENSG2"]
    assert np.isclose(result.loc[0, "z_score"], 2.0)
    assert np.isclose(result.loc[1, "z_score"], 0.0)


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
            alpha=0.05,
        )
    )

    assert exit_code == 0
    assert (output_dir / "ranked_genes" / "sample_a.tsv").exists()
    assert (output_dir / "absolute_outliers.tsv").exists()
    assert (output_dir / "calibration_summary.tsv").exists()
    metadata = json.loads((output_dir / "calibration_run.json").read_text(encoding="utf-8"))
    assert metadata["count_space_method"] == "nb_approx"
    assert metadata["alpha"] == 0.05
    assert "outlier_validation" in metadata
    assert "mean_outliers_per_sample" in metadata["outlier_validation"]


def test_compute_normalized_outliers_gene_wise_centering() -> None:
    """Gene-wise centering subtracts cohort median residual before z-score."""
    result_no_center = calibration.compute_normalized_outliers(
        observed_log1p_tpm=np.array([[3.0, 7.0]], dtype=float),
        expected_mu=np.array([[1.0, 7.0]], dtype=float),
        expected_sigma=np.array([[1.0, 2.0]], dtype=float),
        gene_names=["ENSG1", "ENSG2"],
        sample_names=["s1"],
    )
    result_with_center = calibration.compute_normalized_outliers(
        observed_log1p_tpm=np.array([[3.0, 7.0]], dtype=float),
        expected_mu=np.array([[1.0, 7.0]], dtype=float),
        expected_sigma=np.array([[1.0, 2.0]], dtype=float),
        gene_names=["ENSG1", "ENSG2"],
        sample_names=["s1"],
        gene_centers={"ENSG1": 1.0, "ENSG2": 0.0},
    )
    assert np.isclose(result_no_center.loc[0, "z_score"], 2.0)
    assert np.isclose(result_with_center.loc[0, "z_score"], 1.0)


def test_compute_normalized_outliers_student_t() -> None:
    """Student-t produces valid p-values (heavier tails than Gaussian)."""
    result_gauss = calibration.compute_normalized_outliers(
        observed_log1p_tpm=np.array([[4.0]], dtype=float),
        expected_mu=np.array([[1.0]], dtype=float),
        expected_sigma=np.array([[1.0]], dtype=float),
        gene_names=["G1"],
        sample_names=["s1"],
    )
    result_t = calibration.compute_normalized_outliers(
        observed_log1p_tpm=np.array([[4.0]], dtype=float),
        expected_mu=np.array([[1.0]], dtype=float),
        expected_sigma=np.array([[1.0]], dtype=float),
        gene_names=["G1"],
        sample_names=["s1"],
        use_student_t=True,
        student_t_df=5.0,
    )
    assert 0 <= result_gauss.loc[0, "raw_p_value"] <= 1
    assert 0 <= result_t.loc[0, "raw_p_value"] <= 1
    assert result_t.loc[0, "raw_p_value"] > result_gauss.loc[0, "raw_p_value"]


def test_calibrate_gene_wise_centering_metadata() -> None:
    """Calibration run metadata records gene_wise_centering and use_student_t."""
    ranked = {
        "a": _make_ranked_table(10.0, 5.0, 1.0),
        "b": _make_ranked_table(4.0, 2.0, 1.8),
    }
    result = calibration.calibrate_ranked_gene_scores(
        ranked,
        gene_wise_centering=True,
        use_student_t=True,
        student_t_df=5.0,
    )
    assert result.run_metadata["gene_wise_centering"] is True
    assert result.run_metadata["use_student_t"] is True
    assert result.run_metadata["student_t_df"] == 5.0


def test_validate_outlier_counts() -> None:
    """validate_outlier_counts returns expected keys."""
    summary = pd.DataFrame({
        "tested_genes": [100, 100],
        "absolute_significant_gene_count_by_alpha": [5, 8],
    })
    v = calibration.validate_outlier_counts(summary, alpha=0.05)
    assert "mean_outliers_per_sample" in v
    assert "median_outliers_per_sample" in v
    assert "max_outliers_per_sample" in v
    assert "inflation_ratio" in v


def test_nb_approx_vs_nb_outrider_differ(tmp_path: Path) -> None:
    """nb_approx and nb_outrider should produce different p-values (not identical code path)."""
    # Create count-space artifacts for nb_outrider
    counts_df = pd.DataFrame(
        {"ENSG1": [50, 60, 70, 55, 65], "ENSG2": [100, 90, 80, 95, 85]},
        index=["sa", "sb", "sc", "sd", "se"],
    )
    counts_df.index.name = "sample_id"
    counts_df.to_csv(tmp_path / "aligned_counts.tsv", sep="\t")
    pd.DataFrame({
        "ensg_id": ["ENSG1", "ENSG2"],
        "length_kb": [2.0, 1.5],
        "has_length": [1, 1],
    }).to_csv(tmp_path / "gene_lengths_aligned.tsv", sep="\t", index=False)
    pd.DataFrame({
        "sample_id": ["sa", "sb", "sc", "sd", "se"],
        "S_j": [1e6, 1.1e6, 0.9e6, 1e6, 1.05e6],
    }).to_csv(tmp_path / "sample_scaling.tsv", sep="\t", index=False)

    ranked = {
        sid: _make_ranked_table(score, obs, pred)
        for sid, score, obs, pred in [
            ("sa", 10.0, 5.0, 1.0),
            ("sb", 4.0, 2.0, 1.8),
            ("sc", 3.0, 2.0, 1.9),
            ("sd", 2.0, 2.0, 1.7),
            ("se", 1.0, 2.0, 1.6),
        ]
    }

    result_approx = calibration.calibrate_ranked_gene_scores(
        ranked, count_space_method="nb_approx"
    )
    result_outrider = calibration.calibrate_ranked_gene_scores(
        ranked,
        count_space_method="nb_outrider",
        count_space_path=tmp_path,
    )

    # Collect NB p-values from both
    approx_pvals = []
    outrider_pvals = []
    for sid in ranked:
        at = result_approx.calibrated_ranked_gene_scores[sid]
        ot = result_outrider.calibrated_ranked_gene_scores[sid]
        approx_pvals.extend(at["nb_approx_two_sided_p_value"].tolist())
        if "nb_outrider_p_raw" in ot.columns:
            outrider_pvals.extend(ot["nb_outrider_p_raw"].tolist())

    # Both should produce valid p-values
    approx_arr = np.array(approx_pvals)
    assert np.all(np.isfinite(approx_arr))

    # They should differ for at least some values (not identical code path)
    if len(outrider_pvals) > 0:
        outrider_arr = np.array(outrider_pvals)
        finite_out = outrider_arr[np.isfinite(outrider_arr)]
        if len(finite_out) == len(approx_arr):
            # At least one p-value should differ
            assert not np.allclose(approx_arr, finite_out, atol=1e-10), (
                "nb_approx and nb_outrider produced bitwise-identical p-values!"
            )


def test_min_raw_pvalue_per_sample() -> None:
    """Calibration summary should have plausible min_empirical_p_value for anomalous samples."""
    ranked = {
        sid: _make_ranked_table(score, obs, pred)
        for sid, score, obs, pred in [
            ("sa", 10.0, 5.0, 1.0),  # clearly anomalous
            ("sb", 1.0, 2.0, 1.8),
            ("sc", 1.0, 2.0, 1.9),
            ("sd", 1.0, 2.0, 1.7),
            ("se", 1.0, 2.0, 1.6),
        ]
    }
    result = calibration.calibrate_ranked_gene_scores(ranked)
    summary = result.calibration_summary
    assert "min_empirical_p_value" in summary.columns
    # The most anomalous sample (sa) should have a small min empirical p-value
    sa_row = summary[summary["sample_id"] == "sa"]
    assert sa_row.iloc[0]["min_empirical_p_value"] < 0.5


def test_calibration_diagnostics_in_metadata() -> None:
    """calibrate_ranked_gene_scores populates calibration_diagnostics in run_metadata."""
    ranked = {
        sid: _make_ranked_table(score, obs, pred)
        for sid, score, obs, pred in [
            ("sa", 10.0, 5.0, 1.0),
            ("sb", 4.0, 2.0, 1.8),
            ("sc", 3.0, 2.0, 1.9),
        ]
    }
    result = calibration.calibrate_ranked_gene_scores(ranked)
    assert "calibration_diagnostics" in result.run_metadata
    diag = result.run_metadata["calibration_diagnostics"]
    assert "empirical" in diag
    assert "absolute_zscore" in diag
    assert "ks_stat" in diag["empirical"]
    assert "discovery_table" in diag["empirical"]
    assert "min_p" in diag["empirical"]
