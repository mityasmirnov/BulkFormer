"""Empirical and normalized absolute outlier calibration workflows."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm, t as student_t

from bulkformer_dx.calibration.multitest import benjamini_yekutieli
from bulkformer_dx.calibration.pvalues import empirical_tail_pvalue

SUPPORTED_COUNT_SPACE_METHODS = ("none", "nb_approx", "nb_outrider")
DEFAULT_COUNT_SPACE_METHOD = "none"
DEFAULT_ALPHA = 0.05
SIGMA_EPSILON = 1e-6
DEFAULT_STUDENT_T_DF = 5.0
EMPIRICAL_PVALUE_COLUMN = "empirical_p_value"
BY_QVALUE_COLUMN = "by_q_value"
NB_PVALUE_COLUMN = "nb_approx_p_value"
NB_TWO_SIDED_PVALUE_COLUMN = "nb_approx_two_sided_p_value"
NB_OUTRIDER_PVALUE_COLUMN = "nb_outrider_p_raw"
NB_OUTRIDER_PADJ_COLUMN = "nb_outrider_p_adj"
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
    numeric_columns = sorted(REQUIRED_RANKED_COLUMNS - {"ensg_id"})
    for column in numeric_columns:
        resolved[column] = pd.to_numeric(resolved[column], errors="coerce")
    # Drop rows with non-finite required numeric values (NLL can produce inf/nan for edge cases)
    finite_mask = np.ones(len(resolved), dtype=bool)
    for column in numeric_columns:
        finite_mask &= np.isfinite(resolved[column])
    n_dropped = (~finite_mask).sum()
    if n_dropped > 0:
        resolved = resolved.loc[finite_mask].copy()
    if not resolved["ensg_id"].is_unique:
        raise ValueError(f"Ranked gene table for sample {sample_id!r} contains duplicate genes.")
    if len(resolved) == 0:
        raise ValueError(
            f"Ranked gene table for sample {sample_id!r} has no rows with finite anomaly_score."
        )
    return resolved


def load_embeddings_from_scores_dir(
    scores_path: Path,
) -> tuple[np.ndarray, list[str]] | None:
    """Load sample embeddings from a scoring output directory if present.

    Looks for embeddings.npy and cohort_scores.tsv to establish sample order.
    Returns (embeddings, sample_ids) or None if not available.
    """
    base = scores_path if scores_path.is_dir() and scores_path.name != "ranked_genes" else scores_path.parent
    emb_path = base / "embeddings.npy"
    cohort_path = base / "cohort_scores.tsv"
    if not emb_path.exists() or not cohort_path.exists():
        return None
    try:
        embeddings = np.load(emb_path, allow_pickle=False)
        cohort = _read_table(cohort_path)
        sample_col = "sample_id" if "sample_id" in cohort.columns else str(cohort.columns[0])
        sample_ids = cohort[sample_col].astype(str).tolist()
        if embeddings.shape[0] != len(sample_ids):
            return None
        return np.asarray(embeddings, dtype=float), sample_ids
    except (OSError, ValueError, KeyError):
        return None


def load_embeddings_from_path(
    path: Path,
    sample_ids: list[str],
) -> np.ndarray | None:
    """Load embeddings from a file path.

    Supports .npy (shape n_samples x d) and .npz with 'embeddings' key.
    Sample order must match sample_ids.
    """
    if not path.exists():
        return None
    try:
        if path.suffix.lower() == ".npy":
            arr = np.load(path, allow_pickle=False)
        elif path.suffix.lower() == ".npz":
            data = np.load(path, allow_pickle=False)
            arr = data["embeddings"] if "embeddings" in data else data[list(data.keys())[0]]
        else:
            return None
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != len(sample_ids):
            return None
        return arr
    except (OSError, ValueError, KeyError):
        return None


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


def compute_normalized_outliers(
    observed_log1p_tpm: Any,
    expected_mu: Any,
    expected_sigma: Any,
    gene_names: list[str],
    sample_names: list[str],
    *,
    alpha: float = DEFAULT_ALPHA,
    epsilon: float = SIGMA_EPSILON,
    gene_centers: dict[str, float] | None = None,
    use_student_t: bool = False,
    student_t_df: float = DEFAULT_STUDENT_T_DF,
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
    gene_centers
        Optional cohort median residual per gene. When provided, residuals are
        centered before z-score computation (gene-wise centering).
    use_student_t
        If True, use Student-t distribution for p-values instead of Gaussian.
        Heavier tails often improve calibration when residuals are non-normal.
    student_t_df
        Degrees of freedom for Student-t when use_student_t is True.

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
    if use_student_t and student_t_df <= 0:
        raise ValueError("student_t_df must be positive when use_student_t is True.")

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
    residuals = observed - mu
    if gene_centers is not None:
        center_array = np.array(
            [gene_centers.get(g, 0.0) for g in gene_names],
            dtype=float,
        )
        residuals = residuals - center_array[np.newaxis, :]
    z_scores = residuals / sigma_safe
    if use_student_t:
        raw_p_values = 2.0 * student_t.sf(np.abs(z_scores), df=student_t_df)
    else:
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


def _format_absolute_outlier_method(
    *,
    gene_wise_centering: bool,
    use_student_t: bool,
    student_t_df: float,
) -> str:
    """Build human-readable description of the absolute outlier method."""
    parts = ["z=(Y-mu"]
    if gene_wise_centering:
        parts[0] += "-center_g"
    parts[0] += ")/(sigma+eps)"
    dist = f"Student-t(df={student_t_df})" if use_student_t else "normal"
    parts.append(f"two-sided {dist} p-values")
    parts.append("BY correction")
    return ", ".join(parts)


def _compute_gene_wise_residual_centers(
    ranked_gene_scores: dict[str, pd.DataFrame],
    *,
    cohort_indices: dict[str, list[int]] | None = None,
    sample_id_list: list[str] | None = None,
) -> dict[str, float] | dict[str, dict[str, float]]:
    """Compute cohort median residual per gene for gene-wise centering.

    If cohort_indices is provided, returns per-sample: sample_id -> gene_id -> center.
    Otherwise returns global: gene_id -> center.
    """
    sample_ids = sample_id_list or list(ranked_gene_scores.keys())

    if cohort_indices is None:
        residuals_by_gene = _collect_gene_arrays(
            ranked_gene_scores,
            "mean_signed_residual",
        )
        return {
            gene_id: float(np.nanmedian(residuals))
            for gene_id, residuals in residuals_by_gene.items()
        }

    centers_by_sample: dict[str, dict[str, float]] = {}
    for target_id, neighbor_indices in cohort_indices.items():
        neighbor_ids = [sample_ids[j] for j in neighbor_indices if j < len(sample_ids)]
        sub_scores = {sid: ranked_gene_scores[sid] for sid in neighbor_ids if sid in ranked_gene_scores}
        if len(sub_scores) < 2:
            sub_scores = ranked_gene_scores
        sub_residuals = _collect_gene_arrays(sub_scores, "mean_signed_residual")
        centers_by_sample[target_id] = {
            gene_id: float(np.nanmedian(residuals))
            for gene_id, residuals in sub_residuals.items()
        }
    return centers_by_sample


def _estimate_empirical_sigma_by_gene(
    ranked_gene_scores: dict[str, pd.DataFrame],
    *,
    cohort_indices: dict[str, list[int]] | None = None,
    sample_id_list: list[str] | None = None,
) -> dict[str, float] | dict[str, dict[str, float]]:
    """Estimate a robust residual scale per gene across the cohort.

    If cohort_indices is provided, returns per-sample sigma: sample_id -> gene_id -> sigma.
    Otherwise returns global sigma: gene_id -> sigma.
    """
    residuals_by_gene = _collect_gene_arrays(
        ranked_gene_scores,
        "mean_signed_residual",
    )
    sample_ids = sample_id_list or list(ranked_gene_scores.keys())

    if cohort_indices is None:
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

    sample_id_to_idx = {s: i for i, s in enumerate(sample_ids)}
    sigma_by_sample_gene: dict[str, dict[str, float]] = {}
    for target_id, neighbor_indices in cohort_indices.items():
        neighbor_ids = [sample_ids[j] for j in neighbor_indices if j < len(sample_ids)]
        sub_scores = {sid: ranked_gene_scores[sid] for sid in neighbor_ids if sid in ranked_gene_scores}
        if len(sub_scores) < 2:
            sub_scores = ranked_gene_scores
        sub_residuals = _collect_gene_arrays(sub_scores, "mean_signed_residual")
        sigma_by_gene: dict[str, float] = {}
        for gene_id, residuals in sub_residuals.items():
            center = float(np.nanmedian(residuals))
            mad = float(np.nanmedian(np.abs(residuals - center)))
            sigma = 1.4826 * mad
            if not np.isfinite(sigma) or sigma <= SIGMA_EPSILON:
                ddof = 1 if residuals.size > 1 else 0
                sigma = float(np.nanstd(residuals, ddof=ddof))
            if not np.isfinite(sigma) or sigma <= SIGMA_EPSILON:
                sigma = SIGMA_EPSILON
            sigma_by_gene[gene_id] = sigma
        sigma_by_sample_gene[target_id] = sigma_by_gene
    return sigma_by_sample_gene


def _expression_to_tpm(values: np.ndarray) -> np.ndarray:
    tpm = np.expm1(np.asarray(values, dtype=float))
    return np.clip(tpm, a_min=0.0, a_max=None)


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
    count_space_path: Path | None = None,
    nb_cache_dir: Path | None = None,
    alpha: float = DEFAULT_ALPHA,
    gene_wise_centering: bool = True,
    use_student_t: bool = False,
    student_t_df: float = DEFAULT_STUDENT_T_DF,
    cohort_mode: str = "global",
    knn_k: int = 50,
    embeddings: np.ndarray | None = None,
) -> CalibrationResult:
    """Add empirical calibration, normalized outlier calls, and optional NB approximations."""
    if not ranked_gene_scores:
        raise ValueError("At least one ranked gene score table is required for calibration.")
    if count_space_method not in SUPPORTED_COUNT_SPACE_METHODS:
        supported = ", ".join(SUPPORTED_COUNT_SPACE_METHODS)
        raise ValueError(f"Unsupported count-space method {count_space_method!r}. Use one of: {supported}.")
    if count_space_method == "nb_outrider" and (count_space_path is None or not count_space_path.exists()):
        raise ValueError(
            "nb_outrider requires count_space_path pointing to preprocess output "
            "(aligned_counts.tsv, gene_lengths_aligned.tsv, sample_scaling.tsv)."
        )
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in the interval (0, 1).")
    if cohort_mode not in ("global", "knn_local"):
        raise ValueError(f"cohort_mode must be 'global' or 'knn_local', got {cohort_mode!r}.")

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
    sample_id_list = list(validated_scores.keys())
    cohort_indices = None
    if cohort_mode == "knn_local" and embeddings is not None:
        from bulkformer_dx.calibration.cohort import get_cohort_indices
        cohort_indices = get_cohort_indices(
            sample_id_list,
            cohort_mode=cohort_mode,
            embedding=embeddings,
            knn_k=min(knn_k, len(sample_id_list) - 1),
        )
    sigma_result = _estimate_empirical_sigma_by_gene(
        validated_scores,
        cohort_indices=cohort_indices,
        sample_id_list=sample_id_list,
    )
    centers_result = (
        _compute_gene_wise_residual_centers(
            validated_scores,
            cohort_indices=cohort_indices,
            sample_id_list=sample_id_list,
        )
        if gene_wise_centering
        else None
    )
    nb_parameters = (
        _estimate_negative_binomial_parameters(validated_scores)
        if count_space_method == "nb_approx"
        else {}
    )
    nb_outrider_tables: dict[str, pd.DataFrame] = {}
    if count_space_method == "nb_outrider" and count_space_path is not None:
        from bulkformer_dx.anomaly.nb_test import compute_nb_outrider_for_calibration

        nb_outrider_tables = compute_nb_outrider_for_calibration(
            validated_scores,
            count_space_path,
            cache_dir=nb_cache_dir,
            multiple_testing="BY",
        )

    calibrated_ranked_gene_scores: dict[str, pd.DataFrame] = {}
    absolute_outlier_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for sample_id, table in validated_scores.items():
        calibrated = table.copy()
        empirical_p_values = np.array(
            [
                empirical_tail_pvalue(
                    np.array(
                        [
                            score_by_gene[str(gene_id)]
                            for other_sample_id, score_by_gene in anomaly_score_lookup.items()
                            if other_sample_id != sample_id and str(gene_id) in score_by_gene
                        ],
                        dtype=float,
                    ),
                    float(anomaly_score),
                    upper_tail=True,
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
        if isinstance(sigma_result, dict) and sample_id in sigma_result:
            sig_map = sigma_result[sample_id]
            expected_sigma = np.array(
                [sig_map.get(str(gene_id), SIGMA_EPSILON) for gene_id in calibrated["ensg_id"]],
                dtype=float,
            )
        else:
            expected_sigma = np.array(
                [sigma_result.get(str(gene_id), SIGMA_EPSILON) for gene_id in calibrated["ensg_id"]],
                dtype=float,
            )
        gene_centers = None
        if centers_result is not None:
            if isinstance(centers_result, dict) and sample_id in centers_result:
                gene_centers = centers_result[sample_id]
            else:
                gene_centers = centers_result
        absolute_outliers = compute_normalized_outliers(
            calibrated["observed_expression"].to_numpy(dtype=float, copy=True),
            calibrated["mean_predicted_expression"].to_numpy(dtype=float, copy=True),
            expected_sigma,
            calibrated["ensg_id"].astype(str).tolist(),
            [sample_id],
            alpha=alpha,
            gene_centers=gene_centers,
            use_student_t=use_student_t,
            student_t_df=student_t_df,
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

        if count_space_method == "nb_outrider" and sample_id in nb_outrider_tables:
            nb_table = nb_outrider_tables[sample_id]
            for col in ("nb_outrider_p_raw", "nb_outrider_p_adj", "nb_outrider_direction", "nb_outrider_expected_count"):
                if col in nb_table.columns:
                    calibrated[col] = nb_table[col].values

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
        if count_space_method == "nb_outrider" and NB_OUTRIDER_PVALUE_COLUMN in calibrated.columns:
            summary_row["min_nb_outrider_p_value"] = float(
                np.nanmin(calibrated[NB_OUTRIDER_PVALUE_COLUMN])
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
        "absolute_outlier_method": _format_absolute_outlier_method(
            gene_wise_centering=gene_wise_centering,
            use_student_t=use_student_t,
            student_t_df=student_t_df,
        ),
        "alpha": alpha,
        "gene_wise_centering": gene_wise_centering,
        "use_student_t": use_student_t,
        "cohort_mode": cohort_mode,
    }
    if cohort_mode == "knn_local":
        run_metadata["knn_k"] = knn_k
    if use_student_t:
        run_metadata["student_t_df"] = student_t_df
    if count_space_method == "nb_approx":
        run_metadata["count_space_note"] = (
            "Negative-binomial values are a TPM-derived approximation for ranking support only; "
            "they are not raw-count inference and do not reproduce OUTRIDER."
        )
    if count_space_method == "nb_outrider":
        run_metadata["count_space_note"] = (
            "OUTRIDER-style NB test in count space using BulkFormer as the mean model; "
            "dispersion fitted per gene or with shrinkage."
        )
        if count_space_path is not None:
            run_metadata["count_space_path"] = str(count_space_path)
    return CalibrationResult(
        calibrated_ranked_gene_scores=calibrated_ranked_gene_scores,
        absolute_outliers=absolute_outliers,
        calibration_summary=calibration_summary,
        run_metadata=run_metadata,
    )


def validate_outlier_counts(
    calibration_summary: pd.DataFrame,
    *,
    alpha: float = DEFAULT_ALPHA,
    tested_genes_col: str = "tested_genes",
    absolute_outliers_col: str = "absolute_significant_gene_count_by_alpha",
) -> dict[str, Any]:
    """Validate that outlier counts per sample are plausible.

    Returns summary statistics for downstream checks. Under the null, expected
    significant genes per sample is alpha * tested_genes; heavy inflation may
    indicate calibration issues.

    Returns
    -------
    dict
        Keys: mean_outliers, median_outliers, max_outliers, expected_under_null,
        inflation_ratio (mean / expected).
    """
    if absolute_outliers_col not in calibration_summary.columns:
        return {}
    outliers = calibration_summary[absolute_outliers_col]
    tested = (
        calibration_summary[tested_genes_col]
        if tested_genes_col in calibration_summary.columns
        else None
    )
    expected = (
        float(alpha * tested.mean())
        if tested is not None and tested.size > 0
        else None
    )
    result: dict[str, Any] = {
        "mean_outliers_per_sample": float(outliers.mean()),
        "median_outliers_per_sample": float(outliers.median()),
        "max_outliers_per_sample": int(outliers.max()),
    }
    if expected is not None and expected > 0:
        result["expected_under_null"] = expected
        result["inflation_ratio"] = float(outliers.mean() / expected)
    return result


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
    scores_path = Path(args.scores)
    ranked_gene_scores = load_ranked_gene_scores(scores_path)
    sample_ids = list(ranked_gene_scores.keys())
    alpha = getattr(args, "alpha", DEFAULT_ALPHA)
    count_space_path = getattr(args, "count_space_path", None)
    if count_space_path is not None:
        count_space_path = Path(count_space_path)
    nb_cache_dir = getattr(args, "nb_cache_dir", None)
    if nb_cache_dir is not None:
        nb_cache_dir = Path(nb_cache_dir)
    elif args.count_space_method == "nb_outrider":
        nb_cache_dir = Path(args.output_dir) / "nb_params_cache"
    cohort_mode = getattr(args, "cohort_mode", "global")
    knn_k = getattr(args, "knn_k", 50)
    embeddings = None
    if cohort_mode == "knn_local":
        embedding_path = getattr(args, "embedding_path", None)
        if embedding_path is not None:
            embeddings = load_embeddings_from_path(Path(embedding_path), sample_ids)
        if embeddings is None:
            loaded = load_embeddings_from_scores_dir(scores_path)
            if loaded is not None:
                emb_arr, emb_sample_ids = loaded
                idx_map = {s: i for i, s in enumerate(emb_sample_ids)}
                if all(s in idx_map for s in sample_ids):
                    embeddings = np.array(
                        [emb_arr[idx_map[s]] for s in sample_ids],
                        dtype=np.float32,
                    )
        if embeddings is None:
            raise ValueError(
                "cohort_mode knn_local requires embeddings. Either provide --embedding-path "
                "pointing to a .npy/.npz file, or run calibration on NLL scoring output "
                "(anomaly score --score-type nll) which saves embeddings to the scores directory."
            )
    result = calibrate_ranked_gene_scores(
        ranked_gene_scores,
        count_space_method=args.count_space_method,
        count_space_path=count_space_path,
        nb_cache_dir=nb_cache_dir,
        alpha=alpha,
        gene_wise_centering=getattr(args, "gene_wise_centering", True),
        use_student_t=getattr(args, "use_student_t", False),
        student_t_df=getattr(args, "student_t_df", DEFAULT_STUDENT_T_DF),
        cohort_mode=cohort_mode,
        knn_k=knn_k,
        embeddings=embeddings,
    )
    validation = validate_outlier_counts(
        result.calibration_summary,
        alpha=alpha,
    )
    if validation:
        result.run_metadata["outlier_validation"] = validation

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
    if validation:
        print(
            "Outlier counts: mean={mean:.1f} | median={median:.0f} | max={max} | "
            "inflation={infl:.2f}x".format(
                mean=validation["mean_outliers_per_sample"],
                median=validation["median_outliers_per_sample"],
                max=validation["max_outliers_per_sample"],
                infl=validation.get("inflation_ratio", float("nan")),
            )
        )
    return 0
