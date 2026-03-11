"""Empirical and normalized absolute outlier calibration workflows."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm

SUPPORTED_COUNT_SPACE_METHODS = ("none", "nb_approx")
DEFAULT_COUNT_SPACE_METHOD = "none"
DEFAULT_ALPHA = 0.05
SIGMA_EPSILON = 1e-6
EMPIRICAL_PVALUE_COLUMN = "empirical_p_value"
BY_QVALUE_COLUMN = "by_q_value"
NB_PVALUE_COLUMN = "nb_approx_p_value"
NB_TWO_SIDED_PVALUE_COLUMN = "nb_approx_two_sided_p_value"
EXPECTED_MU_COLUMN = "expected_mu"
EXPECTED_SIGMA_COLUMN = "expected_sigma"
Z_SCORE_COLUMN = "z_score"
RAW_PVALUE_COLUMN = "raw_p_value"
ABSOLUTE_BY_QVALUE_COLUMN = "by_adj_p_value"
IS_SIGNIFICANT_COLUMN = "is_significant"

REQUIRED_RANKED_COLUMNS = {
    "ensg_id",
    "anomaly_score",
    "mean_signed_residual",
    "observed_expression",
    "mean_predicted_expression",
}


@dataclass(slots=True)
class CalibrationResult:
    """In-memory calibration outputs."""

    calibrated_ranked_gene_scores: dict[str, pd.DataFrame]
    absolute_outliers: pd.DataFrame
    calibration_summary: pd.DataFrame
    run_metadata: dict[str, Any]


@dataclass(slots=True)
class NegativeBinomialGeneParameters:
    """Gene-level parameters for the optional count-space approximation."""

    mean_observed_tpm: float
    variance_observed_tpm: float
    dispersion: float
    cohort_size: int


def _to_numpy_2d(values: Any, *, name: str) -> np.ndarray:
    """Convert tensors or arrays into a 2D float NumPy array."""
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    resolved = np.asarray(values, dtype=float)
    if resolved.ndim == 1:
        resolved = resolved[np.newaxis, :]
    if resolved.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got shape {resolved.shape}.")
    if not np.isfinite(resolved).all():
        raise ValueError(f"{name} must contain only finite values.")
    return resolved


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    separator = "\t" if suffix in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=separator)


def _resolve_ranked_dir(scores_path: Path) -> Path:
    if not scores_path.exists():
        raise ValueError(f"Scores path {scores_path} does not exist.")
    if scores_path.is_dir() and scores_path.name == "ranked_genes":
        return scores_path
    ranked_dir = scores_path / "ranked_genes"
    if ranked_dir.is_dir():
        return ranked_dir
    raise ValueError(
        "Scores path must point to an anomaly output directory containing "
        "'ranked_genes/' or to the ranked_genes directory itself."
    )


def _validate_ranked_gene_table(table: pd.DataFrame, *, sample_id: str) -> pd.DataFrame:
    missing_columns = REQUIRED_RANKED_COLUMNS - set(table.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Ranked gene table for sample {sample_id!r} is missing required columns: "
            f"{missing_list}."
        )
    resolved = table.copy()
    resolved["ensg_id"] = resolved["ensg_id"].astype(str)
    if not resolved["ensg_id"].is_unique:
        raise ValueError(f"Ranked gene table for sample {sample_id!r} contains duplicate genes.")
    numeric_columns = sorted(REQUIRED_RANKED_COLUMNS - {"ensg_id"})
    for column in numeric_columns:
        resolved[column] = pd.to_numeric(resolved[column], errors="raise")
        if not np.isfinite(resolved[column]).all():
            raise ValueError(
                f"Ranked gene table for sample {sample_id!r} contains non-finite values "
                f"in required column {column!r}."
            )
    return resolved


def load_ranked_gene_scores(scores_path: Path) -> dict[str, pd.DataFrame]:
    """Load per-sample ranked anomaly scores from a scoring output directory."""
    ranked_dir = _resolve_ranked_dir(scores_path)
    ranked_gene_scores: dict[str, pd.DataFrame] = {}
    file_paths = sorted(ranked_dir.glob("*.tsv")) + sorted(ranked_dir.glob("*.csv"))
    for file_path in file_paths:
        if file_path.stem in ranked_gene_scores:
            raise ValueError(f"Multiple ranked score files resolved to sample {file_path.stem!r}.")
        ranked_gene_scores[file_path.stem] = _validate_ranked_gene_table(
            _read_table(file_path),
            sample_id=file_path.stem,
        )
    if not ranked_gene_scores:
        raise ValueError(f"No ranked gene score files were found in {ranked_dir}.")
    return ranked_gene_scores


def benjamini_yekutieli(p_values: np.ndarray | list[float]) -> np.ndarray:
    """Apply Benjamini-Yekutieli FDR correction."""
    resolved = np.asarray(p_values, dtype=float)
    if resolved.ndim != 1:
        raise ValueError("Benjamini-Yekutieli correction expects a 1D array of p-values.")
    if resolved.size == 0:
        return resolved.copy()
    finite_mask = np.isfinite(resolved)
    if np.any((resolved[finite_mask] < 0.0) | (resolved[finite_mask] > 1.0)):
        raise ValueError("P-values must lie in the interval [0, 1].")

    adjusted = np.full(resolved.shape, np.nan, dtype=float)
    finite_values = resolved[finite_mask]
    if finite_values.size == 0:
        return adjusted

    order = np.argsort(finite_values, kind="mergesort")
    ranked = finite_values[order]
    ranks = np.arange(1, ranked.size + 1, dtype=float)
    harmonic = float(np.sum(1.0 / ranks))
    raw = ranked * ranked.size * harmonic / ranks
    monotone = np.minimum.accumulate(raw[::-1])[::-1]
    reordered = np.empty_like(monotone)
    reordered[order] = np.clip(monotone, 0.0, 1.0)
    adjusted[finite_mask] = reordered
    return adjusted


def compute_normalized_outliers(
    observed_log1p_tpm: Any,
    expected_mu: Any,
    expected_sigma: Any,
    gene_names: list[str],
    sample_names: list[str],
    *,
    alpha: float = DEFAULT_ALPHA,
    epsilon: float = SIGMA_EPSILON,
) -> pd.DataFrame:
    """Convert observed/predicted expression into z-scores and BY-adjusted calls.

    Parameters
    ----------
    observed_log1p_tpm
        Observed expression values with shape ``[samples, genes]``.
    expected_mu
        Expected mean expression values with shape ``[samples, genes]``.
    expected_sigma
        Expected standard deviation values with shape ``[samples, genes]``.
    gene_names
        Ordered gene names aligned to the columns of the input matrices.
    sample_names
        Ordered sample names aligned to the rows of the input matrices.
    alpha
        Significance threshold applied after Benjamini-Yekutieli correction.
    epsilon
        Minimum sigma value used for numerical stability.

    Returns
    -------
    pandas.DataFrame
        Flattened long-format normalized outlier table containing z-scores,
        two-sided p-values, BY-adjusted p-values, and significance calls.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in the interval (0, 1).")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    observed = _to_numpy_2d(observed_log1p_tpm, name="observed_log1p_tpm")
    mu = _to_numpy_2d(expected_mu, name="expected_mu")
    sigma = _to_numpy_2d(expected_sigma, name="expected_sigma")
    if observed.shape != mu.shape or observed.shape != sigma.shape:
        raise ValueError(
            "observed_log1p_tpm, expected_mu, and expected_sigma must share the same shape."
        )
    sample_count, gene_count = observed.shape
    if len(gene_names) != gene_count:
        raise ValueError(
            f"Expected {gene_count} gene names, received {len(gene_names)}."
        )
    if len(sample_names) != sample_count:
        raise ValueError(
            f"Expected {sample_count} sample names, received {len(sample_names)}."
        )

    sigma_safe = np.maximum(sigma, float(epsilon))
    z_scores = (observed - mu) / sigma_safe
    raw_p_values = 2.0 * norm.sf(np.abs(z_scores))
    by_adjusted = np.vstack(
        [benjamini_yekutieli(raw_p_values[row_idx]) for row_idx in range(sample_count)]
    )
    is_significant = by_adjusted < alpha

    flattened = pd.DataFrame(
        {
            "sample_id": np.repeat(np.asarray(sample_names, dtype=object), gene_count),
            "gene": np.tile(np.asarray(gene_names, dtype=object), sample_count),
            "observed_log1p_tpm": observed.reshape(-1),
            EXPECTED_MU_COLUMN: mu.reshape(-1),
            EXPECTED_SIGMA_COLUMN: sigma_safe.reshape(-1),
            Z_SCORE_COLUMN: z_scores.reshape(-1),
            RAW_PVALUE_COLUMN: raw_p_values.reshape(-1),
            ABSOLUTE_BY_QVALUE_COLUMN: by_adjusted.reshape(-1),
            IS_SIGNIFICANT_COLUMN: is_significant.reshape(-1),
        }
    )
    flattened["_abs_z_score"] = flattened[Z_SCORE_COLUMN].abs()
    flattened = flattened.sort_values(
        by=["sample_id", ABSOLUTE_BY_QVALUE_COLUMN, RAW_PVALUE_COLUMN, "_abs_z_score", "gene"],
        ascending=[True, True, True, False, True],
        kind="mergesort",
    ).drop(columns="_abs_z_score").reset_index(drop=True)
    return flattened


def _collect_gene_arrays(
    ranked_gene_scores: dict[str, pd.DataFrame],
    column: str,
    *,
    transform: Any | None = None,
) -> dict[str, np.ndarray]:
    gene_values: dict[str, list[float]] = {}
    for table in ranked_gene_scores.values():
        for gene_id, value in zip(table["ensg_id"], table[column], strict=True):
            gene_values.setdefault(str(gene_id), []).append(float(value))
    if transform is None:
        return {gene_id: np.asarray(values, dtype=float) for gene_id, values in gene_values.items()}
    return {
        gene_id: np.asarray(transform(np.asarray(values, dtype=float)), dtype=float)
        for gene_id, values in gene_values.items()
    }


def _estimate_empirical_sigma_by_gene(
    ranked_gene_scores: dict[str, pd.DataFrame],
) -> dict[str, float]:
    """Estimate a robust residual scale per gene across the cohort."""
    residuals_by_gene = _collect_gene_arrays(
        ranked_gene_scores,
        "mean_signed_residual",
    )
    sigma_by_gene: dict[str, float] = {}
    for gene_id, residuals in residuals_by_gene.items():
        center = float(np.nanmedian(residuals))
        mad = float(np.nanmedian(np.abs(residuals - center)))
        sigma = 1.4826 * mad
        if not np.isfinite(sigma) or sigma <= SIGMA_EPSILON:
            ddof = 1 if residuals.size > 1 else 0
            sigma = float(np.nanstd(residuals, ddof=ddof))
        if not np.isfinite(sigma) or sigma <= SIGMA_EPSILON:
            sigma = SIGMA_EPSILON
        sigma_by_gene[gene_id] = sigma
    return sigma_by_gene


def _expression_to_tpm(values: np.ndarray) -> np.ndarray:
    tpm = np.expm1(np.asarray(values, dtype=float))
    return np.clip(tpm, a_min=0.0, a_max=None)


def _leave_one_out_empirical_p_value(
    *,
    distribution: np.ndarray,
    observed_value: float,
) -> float:
    """Estimate an upper-tail empirical p-value against the cohort background.

    The current sample should not be part of its own reference distribution; this helper
    applies a +1 pseudo-count so finite p-values remain available even for small cohorts.
    """
    finite_distribution = np.asarray(distribution, dtype=float)
    finite_distribution = finite_distribution[np.isfinite(finite_distribution)]
    if finite_distribution.size == 0:
        raise ValueError("Empirical calibration requires at least one finite cohort score.")
    exceedances = float(np.count_nonzero(finite_distribution >= observed_value))
    return (exceedances + 1.0) / (finite_distribution.size + 1.0)


def _estimate_negative_binomial_parameters(
    ranked_gene_scores: dict[str, pd.DataFrame],
) -> dict[str, NegativeBinomialGeneParameters]:
    observed_tpm_by_gene = _collect_gene_arrays(
        ranked_gene_scores,
        "observed_expression",
        transform=_expression_to_tpm,
    )
    parameters: dict[str, NegativeBinomialGeneParameters] = {}
    for gene_id, observed_tpm in observed_tpm_by_gene.items():
        mean_observed_tpm = float(np.mean(observed_tpm))
        variance_observed_tpm = float(np.var(observed_tpm, ddof=1)) if observed_tpm.size > 1 else 0.0
        if mean_observed_tpm <= 0.0:
            dispersion = 1.0
        else:
            dispersion = max(
                (variance_observed_tpm - mean_observed_tpm) / max(mean_observed_tpm**2, 1e-12),
                1e-8,
            )
        parameters[gene_id] = NegativeBinomialGeneParameters(
            mean_observed_tpm=mean_observed_tpm,
            variance_observed_tpm=max(variance_observed_tpm, mean_observed_tpm),
            dispersion=float(dispersion),
            cohort_size=int(observed_tpm.size),
        )
    return parameters


def _negative_binomial_p_values(
    row: pd.Series,
    gene_parameters: NegativeBinomialGeneParameters,
) -> tuple[float, float]:
    predicted_tpm = float(_expression_to_tpm(np.array([row["mean_predicted_expression"]]))[0])
    observed_tpm = float(_expression_to_tpm(np.array([row["observed_expression"]]))[0])
    observed_count = int(np.round(observed_tpm))
    mean_count = max(predicted_tpm, 1e-8)
    dispersion = max(gene_parameters.dispersion, 1e-8)
    size = 1.0 / dispersion
    probability = size / (size + mean_count)
    lower_tail = float(nbinom.cdf(observed_count, size, probability))
    upper_tail = float(nbinom.sf(observed_count - 1, size, probability))
    one_sided = upper_tail if observed_tpm >= mean_count else lower_tail
    two_sided = min(1.0, 2.0 * min(lower_tail, upper_tail))
    return one_sided, two_sided


def calibrate_ranked_gene_scores(
    ranked_gene_scores: dict[str, pd.DataFrame],
    *,
    count_space_method: str = DEFAULT_COUNT_SPACE_METHOD,
    alpha: float = DEFAULT_ALPHA,
) -> CalibrationResult:
    """Add empirical calibration, normalized outlier calls, and optional NB approximations."""
    if not ranked_gene_scores:
        raise ValueError("At least one ranked gene score table is required for calibration.")
    if count_space_method not in SUPPORTED_COUNT_SPACE_METHODS:
        supported = ", ".join(SUPPORTED_COUNT_SPACE_METHODS)
        raise ValueError(f"Unsupported count-space method {count_space_method!r}. Use one of: {supported}.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in the interval (0, 1).")

    validated_scores = {
        sample_id: _validate_ranked_gene_table(table, sample_id=sample_id)
        for sample_id, table in ranked_gene_scores.items()
    }
    if len(validated_scores) < 2:
        raise ValueError("Empirical cohort calibration requires at least two samples.")
    anomaly_score_lookup = {
        sample_id: dict(zip(table["ensg_id"].astype(str), table["anomaly_score"], strict=True))
        for sample_id, table in validated_scores.items()
    }
    empirical_sigma_by_gene = _estimate_empirical_sigma_by_gene(validated_scores)
    nb_parameters = (
        _estimate_negative_binomial_parameters(validated_scores)
        if count_space_method == "nb_approx"
        else {}
    )

    calibrated_ranked_gene_scores: dict[str, pd.DataFrame] = {}
    absolute_outlier_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for sample_id, table in validated_scores.items():
        calibrated = table.copy()
        empirical_p_values = np.array(
            [
                _leave_one_out_empirical_p_value(
                    distribution=np.array(
                        [
                            score_by_gene[str(gene_id)]
                            for other_sample_id, score_by_gene in anomaly_score_lookup.items()
                            if other_sample_id != sample_id and str(gene_id) in score_by_gene
                        ],
                        dtype=float,
                    ),
                    observed_value=float(anomaly_score),
                )
                for gene_id, anomaly_score in zip(
                    calibrated["ensg_id"],
                    calibrated["anomaly_score"],
                    strict=True,
                )
            ],
            dtype=float,
        )
        calibrated[EMPIRICAL_PVALUE_COLUMN] = empirical_p_values
        calibrated[BY_QVALUE_COLUMN] = benjamini_yekutieli(empirical_p_values)
        expected_sigma = np.array(
            [empirical_sigma_by_gene[str(gene_id)] for gene_id in calibrated["ensg_id"]],
            dtype=float,
        )
        absolute_outliers = compute_normalized_outliers(
            calibrated["observed_expression"].to_numpy(dtype=float, copy=True),
            calibrated["mean_predicted_expression"].to_numpy(dtype=float, copy=True),
            expected_sigma,
            calibrated["ensg_id"].astype(str).tolist(),
            [sample_id],
            alpha=alpha,
        )
        absolute_outlier_tables.append(absolute_outliers)
        absolute_columns = absolute_outliers.rename(columns={"gene": "ensg_id"})
        calibrated = calibrated.merge(
            absolute_columns[
                [
                    "ensg_id",
                    EXPECTED_MU_COLUMN,
                    EXPECTED_SIGMA_COLUMN,
                    Z_SCORE_COLUMN,
                    RAW_PVALUE_COLUMN,
                    ABSOLUTE_BY_QVALUE_COLUMN,
                    IS_SIGNIFICANT_COLUMN,
                ]
            ],
            on="ensg_id",
            how="left",
            validate="one_to_one",
        )

        if count_space_method == "nb_approx":
            nb_one_sided: list[float] = []
            nb_two_sided: list[float] = []
            for _, row in calibrated.iterrows():
                one_sided, two_sided = _negative_binomial_p_values(
                    row,
                    nb_parameters[str(row["ensg_id"])],
                )
                nb_one_sided.append(one_sided)
                nb_two_sided.append(two_sided)
            calibrated[NB_PVALUE_COLUMN] = np.asarray(nb_one_sided, dtype=float)
            calibrated[NB_TWO_SIDED_PVALUE_COLUMN] = np.asarray(nb_two_sided, dtype=float)

        calibrated = calibrated.sort_values(
            by=[BY_QVALUE_COLUMN, EMPIRICAL_PVALUE_COLUMN, "anomaly_score", "ensg_id"],
            ascending=[True, True, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        calibrated_ranked_gene_scores[sample_id] = calibrated

        summary_row: dict[str, Any] = {
            "sample_id": sample_id,
            "tested_genes": int(len(calibrated)),
            "min_empirical_p_value": float(np.nanmin(calibrated[EMPIRICAL_PVALUE_COLUMN])),
            "min_by_q_value": float(np.nanmin(calibrated[BY_QVALUE_COLUMN])),
            "significant_gene_count_by_0_05": int(np.count_nonzero(calibrated[BY_QVALUE_COLUMN] <= 0.05)),
            "absolute_significant_gene_count_by_alpha": int(
                np.count_nonzero(calibrated[IS_SIGNIFICANT_COLUMN])
            ),
            "min_absolute_by_adj_p_value": float(
                np.nanmin(calibrated[ABSOLUTE_BY_QVALUE_COLUMN])
            ),
        }
        if count_space_method == "nb_approx":
            summary_row["min_nb_approx_two_sided_p_value"] = float(
                np.nanmin(calibrated[NB_TWO_SIDED_PVALUE_COLUMN])
            )
        summary_rows.append(summary_row)

    calibration_summary = (
        pd.DataFrame(summary_rows)
        .sort_values(by=["min_by_q_value", "sample_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    absolute_outliers = pd.concat(absolute_outlier_tables, ignore_index=True)
    absolute_outliers = absolute_outliers.sort_values(
        by=["sample_id", ABSOLUTE_BY_QVALUE_COLUMN, RAW_PVALUE_COLUMN, "gene"],
        ascending=[True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    run_metadata: dict[str, Any] = {
        "samples": len(calibrated_ranked_gene_scores),
        "scored_genes": int(sum(len(table) for table in calibrated_ranked_gene_scores.values())),
        "count_space_method": count_space_method,
        "empirical_method": "cohort upper-tail fraction over anomaly_score",
        "absolute_outlier_method": "z=(Y-mu)/(sigma+eps), two-sided normal p-values, BY correction",
        "alpha": alpha,
    }
    if count_space_method == "nb_approx":
        run_metadata["count_space_note"] = (
            "Negative-binomial values are a TPM-derived approximation for ranking support only; "
            "they are not raw-count inference and do not reproduce OUTRIDER."
        )
    return CalibrationResult(
        calibrated_ranked_gene_scores=calibrated_ranked_gene_scores,
        absolute_outliers=absolute_outliers,
        calibration_summary=calibration_summary,
        run_metadata=run_metadata,
    )


def write_calibration_outputs(result: CalibrationResult, output_dir: Path) -> None:
    """Write calibrated ranked tables, summaries, and metadata to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked_dir = output_dir / "ranked_genes"
    ranked_dir.mkdir(parents=True, exist_ok=True)

    for sample_id, table in result.calibrated_ranked_gene_scores.items():
        table.to_csv(ranked_dir / f"{sample_id}.tsv", sep="\t", index=False)
    result.absolute_outliers.to_csv(output_dir / "absolute_outliers.tsv", sep="\t", index=False)
    result.calibration_summary.to_csv(output_dir / "calibration_summary.tsv", sep="\t", index=False)
    with (output_dir / "calibration_run.json").open("w", encoding="utf-8") as handle:
        json.dump(result.run_metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run(args: argparse.Namespace) -> int:
    """Calibrate ranked anomaly scores across the cohort."""
    ranked_gene_scores = load_ranked_gene_scores(Path(args.scores))
    result = calibrate_ranked_gene_scores(
        ranked_gene_scores,
        count_space_method=args.count_space_method,
        alpha=getattr(args, "alpha", DEFAULT_ALPHA),
    )
    output_dir = Path(args.output_dir)
    write_calibration_outputs(result, output_dir)
    print(f"Wrote calibrated anomaly outputs to {output_dir}")
    print(
        "Samples: {samples} | scored genes: {genes} | method: {method}".format(
            samples=result.run_metadata["samples"],
            genes=result.run_metadata["scored_genes"],
            method=result.run_metadata["count_space_method"],
        )
    )
    return 0
