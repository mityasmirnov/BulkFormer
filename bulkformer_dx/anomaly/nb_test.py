"""OUTRIDER-style NB test in count space using BulkFormer as the mean model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bulkformer_dx.anomaly.calibration import benjamini_yekutieli
from bulkformer_dx.io.schemas import AlignedExpressionBundle, MethodConfig, ModelPredictionBundle
from bulkformer_dx.stats.dispersion import (
    DispersionFitResult,
    fit_deseq2_trend,
    fit_nb_dispersion_mle,
    shrink_dispersion_to_trend,
)
from bulkformer_dx.stats.nb import outrider_two_sided_nb_pvalue

NB_PARAMS_CACHE_FILENAME = "nb_params.parquet"
NB_PARAMS_METADATA_FILENAME = "nb_params_metadata.json"
MEAN_EPS = 1e-8
DISPERSION_EPS = 1e-10


def expected_counts_from_predicted_tpm(
    pred_tpm: np.ndarray,
    gene_lengths_kb: np.ndarray,
    S_j: np.ndarray,
) -> np.ndarray:
    """Map predicted TPM to expected count mean.

    Formula: mu_count = pred_tpm * (S_j / 1e6) * L_g

    Args:
        pred_tpm: Predicted TPM (n_samples, n_genes).
        gene_lengths_kb: Gene lengths in kb (n_genes,).
        S_j: TPM scaling per sample (n_samples,).

    Returns:
        Expected counts matrix (n_samples, n_genes).
    """
    pred_tpm = np.asarray(pred_tpm, dtype=float)
    gene_lengths_kb = np.asarray(gene_lengths_kb, dtype=float)
    S_j = np.asarray(S_j, dtype=float)

    if pred_tpm.ndim != 2:
        raise ValueError("pred_tpm must be 2D (samples x genes).")
    n_samples, n_genes = pred_tpm.shape
    if gene_lengths_kb.shape != (n_genes,):
        raise ValueError(
            f"gene_lengths_kb must have shape (n_genes,)={n_genes}, got {gene_lengths_kb.shape}."
        )
    if S_j.shape != (n_samples,):
        raise ValueError(
            f"S_j must have shape (n_samples,)={n_samples}, got {S_j.shape}."
        )

    L = np.maximum(gene_lengths_kb, 1e-6)
    tpm_safe = np.maximum(pred_tpm, 0.0)
    S_j_safe = np.maximum(S_j, 1e-6)
    mu_count = tpm_safe * (S_j_safe[:, np.newaxis] / 1e6) * L[np.newaxis, :]
    return mu_count


def _log1p_tpm_to_tpm(log1p_tpm: np.ndarray) -> np.ndarray:
    """Convert log1p(TPM) to TPM."""
    return np.maximum(np.expm1(np.asarray(log1p_tpm, dtype=float)), 0.0)


def _fit_dispersions(
    counts: np.ndarray,
    mu: np.ndarray,
    gene_ids: list[str],
    valid_mask: np.ndarray,
    *,
    dispersion_method: str = "mle",
    use_shrinkage: bool = False,
) -> dict[str, np.ndarray]:
    """Fit per-gene NB dispersion. Returns alpha and size arrays."""
    n_samples, n_genes = counts.shape
    alpha_by_gene: dict[str, float] = {}
    mean_counts_by_gene: dict[str, float] = {}

    for g in range(n_genes):
        gid = gene_ids[g]
        valid = valid_mask[:, g]
        if valid.sum() < 2:
            alpha_by_gene[gid] = 0.1
            mean_counts_by_gene[gid] = float(np.mean(mu[:, g]))
            continue
        k_g = counts[valid, g]
        mu_g = np.maximum(mu[valid, g], MEAN_EPS)
        if dispersion_method == "mle":
            res = fit_nb_dispersion_mle(mu_g, k_g)
        else:
            from bulkformer_dx.stats.dispersion import fit_nb_dispersion_moments

            res = fit_nb_dispersion_moments(mu_g, k_g)
        alpha_by_gene[gid] = max(res.alpha, DISPERSION_EPS)
        mean_counts_by_gene[gid] = float(np.mean(mu[:, g]))

    if use_shrinkage and len(alpha_by_gene) >= 2:
        means = np.array([mean_counts_by_gene[gid] for gid in gene_ids])
        alphas = np.array([alpha_by_gene[gid] for gid in gene_ids])
        valid_for_trend = (means > MEAN_EPS) & (alphas > 0)
        if valid_for_trend.sum() >= 2:
            asympt_disp, extra_pois = fit_deseq2_trend(
                means[valid_for_trend],
                alphas[valid_for_trend],
            )
            for i, gid in enumerate(gene_ids):
                alpha_by_gene[gid] = shrink_dispersion_to_trend(
                    alpha_by_gene[gid],
                    mean_counts_by_gene[gid],
                    asympt_disp,
                    extra_pois,
                    prior_weight=0.3,
                )

    alpha_arr = np.array([alpha_by_gene.get(gid, 0.1) for gid in gene_ids])
    size_arr = 1.0 / np.maximum(alpha_arr, DISPERSION_EPS)
    return {"alpha": alpha_arr, "size": size_arr}


def _load_or_fit_dispersions(
    counts: np.ndarray,
    mu: np.ndarray,
    gene_ids: list[str],
    valid_mask: np.ndarray,
    cache_dir: Path | None,
    *,
    dispersion_method: str = "mle",
    use_shrinkage: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load dispersion from cache or fit and optionally cache."""
    cache_path = cache_dir / NB_PARAMS_CACHE_FILENAME if cache_dir else None

    if cache_path is not None and cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
            if "ensg_id" in cached.columns and "alpha" in cached.columns:
                cached = cached.set_index("ensg_id")
                alpha_arr = np.array(
                    [cached.loc[gid, "alpha"] if gid in cached.index else 0.1 for gid in gene_ids]
                )
                size_arr = 1.0 / np.maximum(alpha_arr, DISPERSION_EPS)
                return alpha_arr, size_arr
        except Exception:
            pass

    result = _fit_dispersions(
        counts,
        mu,
        gene_ids,
        valid_mask,
        dispersion_method=dispersion_method,
        use_shrinkage=use_shrinkage,
    )

    if cache_path is not None and cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {
                "ensg_id": gene_ids,
                "alpha": result["alpha"],
                "size": result["size"],
            }
        )
        df.to_parquet(cache_path, index=False)
        meta = {
            "dispersion_method": dispersion_method,
            "use_shrinkage": use_shrinkage,
            "n_genes": len(gene_ids),
        }
        with (cache_dir / NB_PARAMS_METADATA_FILENAME).open("w") as f:
            json.dump(meta, f, indent=2)

    return result["alpha"], result["size"]


@dataclass(slots=True)
class NBOutriderResult:
    """In-memory NB OUTRIDER test outputs."""

    p_raw: np.ndarray  # (n_samples, n_genes)
    p_adj: np.ndarray
    direction: np.ndarray  # "under" | "over" | ""
    mu_expected: np.ndarray
    alpha: np.ndarray
    gene_ids: list[str]
    sample_ids: list[str]
    valid_mask: np.ndarray


def run_nb_outrider_test(
    bundle: AlignedExpressionBundle,
    preds: ModelPredictionBundle,
    method_config: MethodConfig | None = None,
    *,
    dispersion_method: str = "mle",
    use_shrinkage: bool = False,
    cache_dir: Path | None = None,
    multiple_testing: str = "BY",
) -> NBOutriderResult:
    """Run OUTRIDER-style two-sided NB test using BulkFormer predictions.

    Args:
        bundle: Aligned expression with counts, gene_length_kb, tpm_scaling_S.
        preds: Model predictions (y_hat in log1p(TPM)).
        method_config: Optional method config; defaults used if None.
        dispersion_method: "mle" or "moments".
        use_shrinkage: Whether to shrink dispersions to DESeq2-like trend.
        cache_dir: Directory to cache dispersion parameters.
        multiple_testing: "BY" or "BH" or "none".

    Returns:
        NBOutriderResult with p-values, adjusted p-values, per (sample, gene).
    """
    if bundle.counts is None:
        raise ValueError("AlignedExpressionBundle must have counts for NB test.")
    if bundle.gene_length_kb is None:
        raise ValueError("AlignedExpressionBundle must have gene_length_kb for NB test.")
    if bundle.tpm_scaling_S is None:
        raise ValueError("AlignedExpressionBundle must have tpm_scaling_S for NB test.")

    counts = np.asarray(bundle.counts, dtype=float)
    gene_length_kb = np.asarray(bundle.gene_length_kb, dtype=float)
    S_j = np.asarray(bundle.tpm_scaling_S, dtype=float)
    y_hat = np.asarray(preds.y_hat, dtype=float)
    valid_mask = np.asarray(bundle.valid_mask, dtype=bool)

    n_samples, n_genes = counts.shape
    pred_tpm = _log1p_tpm_to_tpm(y_hat)
    mu_expected = expected_counts_from_predicted_tpm(pred_tpm, gene_length_kb, S_j)

    alpha_arr, size_arr = _load_or_fit_dispersions(
        counts,
        mu_expected,
        bundle.gene_ids,
        valid_mask,
        cache_dir,
        dispersion_method=dispersion_method,
        use_shrinkage=use_shrinkage,
    )

    p_raw = np.full((n_samples, n_genes), np.nan, dtype=float)
    direction = np.full((n_samples, n_genes), "", dtype=object)

    for i in range(n_samples):
        for g in range(n_genes):
            if not valid_mask[i, g]:
                continue
            k = int(np.floor(counts[i, g]))
            mu = max(float(mu_expected[i, g]), MEAN_EPS)
            size = max(float(size_arr[g]), 1e-6)
            p_raw[i, g] = outrider_two_sided_nb_pvalue(k, mu, size)
            if k >= mu:
                direction[i, g] = "over"
            else:
                direction[i, g] = "under"

    if multiple_testing == "BY":
        p_adj = np.full_like(p_raw, np.nan)
        for i in range(n_samples):
            valid_g = valid_mask[i, :]
            if valid_g.sum() > 0:
                p_adj[i, valid_g] = benjamini_yekutieli(p_raw[i, valid_g])
    elif multiple_testing == "BH":
        from scipy.stats import false_discovery_control

        p_adj = np.full_like(p_raw, np.nan)
        for i in range(n_samples):
            valid_g = valid_mask[i, :]
            if valid_g.sum() > 0:
                p_adj[i, valid_g] = false_discovery_control(p_raw[i, valid_g])
    else:
        p_adj = p_raw.copy()

    return NBOutriderResult(
        p_raw=p_raw,
        p_adj=p_adj,
        direction=direction,
        mu_expected=mu_expected,
        alpha=alpha_arr,
        gene_ids=bundle.gene_ids,
        sample_ids=bundle.sample_ids,
        valid_mask=valid_mask,
    )


def nb_outrider_result_to_ranked_tables(
    result: NBOutriderResult,
    ranked_gene_scores: dict[str, pd.DataFrame],
    *,
    count_column: str = "observed_count",
    pred_column: str = "mean_predicted_expression",
) -> dict[str, pd.DataFrame]:
    """Merge NB OUTRIDER p-values into per-sample ranked tables.

    ranked_gene_scores keys are sample_ids; each table has ensg_id, anomaly_score,
    mean_predicted_expression, observed_expression, etc.
    """
    merged: dict[str, pd.DataFrame] = {}
    sample_id_to_idx = {s: i for i, s in enumerate(result.sample_ids)}
    gene_id_to_idx = {g: i for i, g in enumerate(result.gene_ids)}

    for sample_id, table in ranked_gene_scores.items():
        if sample_id not in sample_id_to_idx:
            continue
        idx_s = sample_id_to_idx[sample_id]
        df = table.copy()
        p_raw_list = []
        p_adj_list = []
        direction_list = []
        mu_list = []
        for _, row in df.iterrows():
            gid = str(row["ensg_id"])
            if gid in gene_id_to_idx:
                idx_g = gene_id_to_idx[gid]
                if result.valid_mask[idx_s, idx_g]:
                    p_raw_list.append(result.p_raw[idx_s, idx_g])
                    p_adj_list.append(result.p_adj[idx_s, idx_g])
                    direction_list.append(result.direction[idx_s, idx_g])
                    mu_list.append(result.mu_expected[idx_s, idx_g])
                else:
                    p_raw_list.append(np.nan)
                    p_adj_list.append(np.nan)
                    direction_list.append("")
                    mu_list.append(np.nan)
            else:
                p_raw_list.append(np.nan)
                p_adj_list.append(np.nan)
                direction_list.append("")
                mu_list.append(np.nan)

        df["nb_outrider_p_raw"] = p_raw_list
        df["nb_outrider_p_adj"] = p_adj_list
        df["nb_outrider_direction"] = direction_list
        df["nb_outrider_expected_count"] = mu_list
        merged[sample_id] = df

    return merged


def compute_nb_outrider_for_calibration(
    ranked_gene_scores: dict[str, pd.DataFrame],
    count_space_path: Path,
    *,
    dispersion_method: str = "mle",
    use_shrinkage: bool = False,
    cache_dir: Path | None = None,
    multiple_testing: str = "BY",
) -> dict[str, pd.DataFrame]:
    """Compute NB OUTRIDER p-values for calibration from ranked tables + count-space artifacts.

    Loads aligned_counts, gene_lengths_aligned, sample_scaling from count_space_path.
    Returns per-sample tables with nb_outrider_p_raw, nb_outrider_p_adj, etc.
    Only genes present in ranked tables (masked genes) are tested.
    """
    from bulkformer_dx.anomaly.scoring import load_aligned_expression

    counts_path = count_space_path / "aligned_counts.tsv"
    lengths_path = count_space_path / "gene_lengths_aligned.tsv"
    scaling_path = count_space_path / "sample_scaling.tsv"
    for p, name in [
        (counts_path, "aligned_counts.tsv"),
        (lengths_path, "gene_lengths_aligned.tsv"),
        (scaling_path, "sample_scaling.tsv"),
    ]:
        if not p.exists():
            raise FileNotFoundError(
                f"Count-space path must contain {name}. Missing: {p}"
            )

    aligned_counts = load_aligned_expression(counts_path)

    gene_lengths = pd.read_csv(lengths_path, sep="\t")
    if "ensg_id" not in gene_lengths.columns or "length_kb" not in gene_lengths.columns:
        raise ValueError("gene_lengths_aligned must have ensg_id and length_kb columns.")
    lengths_df = gene_lengths.set_index("ensg_id")

    sample_scaling = pd.read_csv(scaling_path, sep="\t")
    if "sample_id" in sample_scaling.columns:
        sample_scaling = sample_scaling.set_index("sample_id")
    elif sample_scaling.index.name is None and sample_scaling.shape[1] >= 2:
        sample_scaling = sample_scaling.set_index(sample_scaling.columns[0])
    if "S_j" not in sample_scaling.columns:
        raise ValueError("sample_scaling must have S_j column.")

    sample_ids = [s for s in ranked_gene_scores if s in aligned_counts.index]
    if not sample_ids:
        raise ValueError(
            "No sample IDs from ranked tables found in aligned_counts. "
            "Ensure preprocess and anomaly scoring used the same sample set."
        )

    gene_ids = list(aligned_counts.columns)
    counts = aligned_counts.loc[sample_ids, gene_ids].to_numpy(dtype=float)
    gene_length_kb = np.array(
        [lengths_df.loc[g, "length_kb"] if g in lengths_df.index else 1.0 for g in gene_ids],
        dtype=float,
    )
    S_j = sample_scaling.loc[sample_ids, "S_j"].to_numpy(dtype=float)

    pred_log1p = np.full((len(sample_ids), len(gene_ids)), np.nan, dtype=float)
    sample_id_to_idx = {s: i for i, s in enumerate(sample_ids)}
    gene_id_to_idx = {g: i for i, g in enumerate(gene_ids)}

    for sample_id, table in ranked_gene_scores.items():
        if sample_id not in sample_id_to_idx:
            continue
        idx_s = sample_id_to_idx[sample_id]
        for _, row in table.iterrows():
            gid = str(row["ensg_id"])
            if gid in gene_id_to_idx:
                pred_log1p[idx_s, gene_id_to_idx[gid]] = float(row["mean_predicted_expression"])

    pred_tpm = _log1p_tpm_to_tpm(pred_log1p)
    mu_expected = expected_counts_from_predicted_tpm(pred_tpm, gene_length_kb, S_j)

    valid_mask = (
        np.isfinite(mu_expected)
        & (mu_expected > MEAN_EPS)
        & (counts >= 0)
        & np.isfinite(counts)
    )
    valid_mask &= ~np.isnan(pred_log1p)

    alpha_arr, size_arr = _load_or_fit_dispersions(
        counts,
        mu_expected,
        gene_ids,
        valid_mask,
        cache_dir,
        dispersion_method=dispersion_method,
        use_shrinkage=use_shrinkage,
    )

    p_raw = np.full((len(sample_ids), len(gene_ids)), np.nan, dtype=float)
    direction = np.full((len(sample_ids), len(gene_ids)), "", dtype=object)

    for i in range(len(sample_ids)):
        for g in range(len(gene_ids)):
            if not valid_mask[i, g]:
                continue
            k = int(np.floor(counts[i, g]))
            mu = max(float(mu_expected[i, g]), MEAN_EPS)
            size = max(float(size_arr[g]), 1e-6)
            p_raw[i, g] = outrider_two_sided_nb_pvalue(k, mu, size)
            direction[i, g] = "over" if k >= mu else "under"

    if multiple_testing == "BY":
        p_adj = np.full_like(p_raw, np.nan)
        for i in range(len(sample_ids)):
            valid_g = valid_mask[i, :]
            if valid_g.sum() > 0:
                p_adj[i, valid_g] = benjamini_yekutieli(p_raw[i, valid_g])
    else:
        p_adj = p_raw.copy()

    result_tables: dict[str, pd.DataFrame] = {}
    for sample_id, table in ranked_gene_scores.items():
        if sample_id not in sample_id_to_idx:
            result_tables[sample_id] = table.copy()
            continue
        idx_s = sample_id_to_idx[sample_id]
        df = table.copy()
        p_raw_list = []
        p_adj_list = []
        direction_list = []
        mu_list = []
        for _, row in df.iterrows():
            gid = str(row["ensg_id"])
            if gid in gene_id_to_idx:
                idx_g = gene_id_to_idx[gid]
                if valid_mask[idx_s, idx_g]:
                    p_raw_list.append(p_raw[idx_s, idx_g])
                    p_adj_list.append(p_adj[idx_s, idx_g])
                    direction_list.append(direction[idx_s, idx_g])
                    mu_list.append(mu_expected[idx_s, idx_g])
                else:
                    p_raw_list.append(np.nan)
                    p_adj_list.append(np.nan)
                    direction_list.append("")
                    mu_list.append(np.nan)
            else:
                p_raw_list.append(np.nan)
                p_adj_list.append(np.nan)
                direction_list.append("")
                mu_list.append(np.nan)

        df["nb_outrider_p_raw"] = p_raw_list
        df["nb_outrider_p_adj"] = p_adj_list
        df["nb_outrider_direction"] = direction_list
        df["nb_outrider_expected_count"] = mu_list
        result_tables[sample_id] = df

    return result_tables
