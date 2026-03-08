"""Monte Carlo masking anomaly scoring for BulkFormer-aligned inputs."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from bulkformer_dx.bulkformer_model import load_bulkformer_model, predict_expression

MASK_TOKEN_VALUE = -10.0
DEFAULT_MASK_PROB = 0.15
DEFAULT_MC_PASSES = 16
DEFAULT_RANDOM_SEED = 0


@dataclass(slots=True)
class AnomalyScoringResult:
    """In-memory anomaly scoring outputs."""

    cohort_scores: pd.DataFrame
    gene_qc: pd.DataFrame
    ranked_gene_scores: dict[str, pd.DataFrame]
    run_metadata: dict[str, Any]


Predictor = Callable[[np.ndarray, float], np.ndarray]


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    separator = "\t" if suffix in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=separator)


def load_aligned_expression(path: Path) -> pd.DataFrame:
    """Load a sample-by-gene aligned expression matrix from disk."""
    expression = _read_table(path)
    if expression.empty:
        raise ValueError("The aligned expression table is empty.")
    sample_column = str(expression.columns[0])
    expression = expression.set_index(sample_column)
    return expression.astype(float)


def load_valid_gene_mask(path: Path) -> pd.DataFrame:
    """Load the BulkFormer valid-gene mask table."""
    valid_gene_mask = _read_table(path)
    required_columns = {"ensg_id", "is_valid"}
    missing_columns = required_columns - set(valid_gene_mask.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Valid-gene mask is missing required columns: {missing_list}."
        )
    valid_gene_mask = valid_gene_mask.loc[:, ["ensg_id", "is_valid"]].copy()
    valid_gene_mask["ensg_id"] = valid_gene_mask["ensg_id"].astype(str)
    valid_gene_mask["is_valid"] = valid_gene_mask["is_valid"].astype(int)
    return valid_gene_mask


def resolve_valid_gene_flags(
    valid_gene_mask: pd.DataFrame,
    expected_genes: pd.Index,
) -> np.ndarray:
    """Align the valid-gene mask to the expression matrix column order."""
    gene_mask_indexed = valid_gene_mask.drop_duplicates(subset=["ensg_id"]).set_index("ensg_id")
    resolved = gene_mask_indexed.reindex(expected_genes)
    if resolved["is_valid"].isna().any():
        missing = resolved.index[resolved["is_valid"].isna()].tolist()
        preview = ", ".join(missing[:5])
        raise ValueError(
            "Valid-gene mask did not cover every input gene column. Missing genes: "
            f"{preview}"
        )
    return resolved["is_valid"].to_numpy(dtype=bool)


def generate_mc_mask_plan(
    valid_gene_flags: np.ndarray,
    *,
    sample_count: int,
    mc_passes: int,
    mask_prob: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a sample-by-pass-by-gene masking plan over valid genes only."""
    valid_gene_flags = np.asarray(valid_gene_flags, dtype=bool)
    if valid_gene_flags.ndim != 1:
        raise ValueError("Valid gene flags must be a 1D boolean array.")
    if sample_count <= 0:
        raise ValueError("Sample count must be positive.")
    if mc_passes <= 0:
        raise ValueError("Monte Carlo passes must be positive.")
    if not 0 < mask_prob <= 1:
        raise ValueError("Mask probability must be in the interval (0, 1].")

    valid_gene_indices = np.flatnonzero(valid_gene_flags)
    valid_gene_count = int(valid_gene_indices.size)
    if valid_gene_count == 0:
        raise ValueError("At least one valid BulkFormer gene is required for anomaly scoring.")

    genes_to_mask_per_pass = min(
        valid_gene_count,
        max(1, int(np.ceil(valid_gene_count * mask_prob))),
    )
    mask_plan = np.zeros(
        (sample_count, mc_passes, valid_gene_flags.shape[0]),
        dtype=bool,
    )
    for sample_idx in range(sample_count):
        for pass_idx in range(mc_passes):
            masked_gene_indices = rng.choice(
                valid_gene_indices,
                size=genes_to_mask_per_pass,
                replace=False,
            )
            mask_plan[sample_idx, pass_idx, masked_gene_indices] = True
    return mask_plan


def _validate_mask_plan(
    mask_plan: np.ndarray,
    *,
    sample_count: int,
    mc_passes: int,
    gene_count: int,
    valid_gene_flags: np.ndarray,
) -> np.ndarray:
    resolved_mask_plan = np.asarray(mask_plan, dtype=bool)
    expected_shape = (sample_count, mc_passes, gene_count)
    if resolved_mask_plan.shape != expected_shape:
        raise ValueError(
            f"Mask plan shape {resolved_mask_plan.shape} did not match {expected_shape}."
        )
    invalid_gene_masking = resolved_mask_plan[:, :, ~valid_gene_flags]
    if invalid_gene_masking.any():
        raise ValueError("Mask plan attempted to Monte Carlo mask genes marked as invalid.")
    if (resolved_mask_plan.sum(axis=2) == 0).any():
        raise ValueError("Each Monte Carlo pass must mask at least one valid gene.")
    return resolved_mask_plan


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full(numerator.shape, np.nan, dtype=float)
    np.divide(
        numerator,
        denominator,
        out=result,
        where=denominator > 0,
    )
    return result


def _sanitize_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return sanitized or "sample"


def _predict_masked_expression(
    masked_expression: np.ndarray,
    *,
    predictor: Predictor,
    fill_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    row_mask_fractions = (masked_expression == fill_value).mean(axis=1)
    predicted = np.empty_like(masked_expression, dtype=float)
    for mask_fraction in np.unique(row_mask_fractions):
        row_selector = np.isclose(row_mask_fractions, mask_fraction)
        predicted_subset = np.asarray(
            predictor(masked_expression[row_selector], float(mask_fraction)),
            dtype=float,
        )
        expected_shape = masked_expression[row_selector].shape
        if predicted_subset.shape != expected_shape:
            raise ValueError(
                "Predictor returned an array with shape "
                f"{predicted_subset.shape}, expected {expected_shape}."
            )
        predicted[row_selector] = predicted_subset
    return predicted, row_mask_fractions


def score_expression_anomalies(
    expression: pd.DataFrame,
    valid_gene_mask: pd.DataFrame,
    *,
    predictor: Predictor,
    mc_passes: int = DEFAULT_MC_PASSES,
    mask_prob: float = DEFAULT_MASK_PROB,
    fill_value: float = MASK_TOKEN_VALUE,
    random_seed: int = DEFAULT_RANDOM_SEED,
    mask_plan: np.ndarray | None = None,
) -> AnomalyScoringResult:
    """Score anomalies by repeatedly masking valid genes and aggregating residuals."""
    if expression.empty:
        raise ValueError("Expression matrix must contain at least one sample.")

    observed = expression.to_numpy(dtype=float, copy=True)
    sample_ids = [str(sample_id) for sample_id in expression.index]
    gene_ids = [str(gene_id) for gene_id in expression.columns]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Sample IDs must be unique for ranked anomaly output generation.")
    sample_count, gene_count = observed.shape
    valid_gene_flags = resolve_valid_gene_flags(valid_gene_mask, expression.columns)
    valid_gene_count = int(valid_gene_flags.sum())
    rng = np.random.default_rng(random_seed)
    resolved_mask_plan = (
        generate_mc_mask_plan(
            valid_gene_flags,
            sample_count=sample_count,
            mc_passes=mc_passes,
            mask_prob=mask_prob,
            rng=rng,
        )
        if mask_plan is None
        else _validate_mask_plan(
            mask_plan,
            sample_count=sample_count,
            mc_passes=mc_passes,
            gene_count=gene_count,
            valid_gene_flags=valid_gene_flags,
        )
    )

    masked_expression = np.repeat(observed[:, None, :], mc_passes, axis=1)
    masked_expression[resolved_mask_plan] = fill_value
    flattened_expression = masked_expression.reshape(sample_count * mc_passes, gene_count)
    predicted, row_mask_fractions = _predict_masked_expression(
        flattened_expression,
        predictor=predictor,
        fill_value=fill_value,
    )
    predicted = predicted.reshape(sample_count, mc_passes, gene_count)
    residuals = observed[:, None, :] - predicted
    mask_counts = resolved_mask_plan.sum(axis=1)
    mask_float = resolved_mask_plan.astype(float)
    abs_residual_sum = (np.abs(residuals) * mask_float).sum(axis=1)
    signed_residual_sum = (residuals * mask_float).sum(axis=1)
    squared_residual_sum = ((residuals**2) * mask_float).sum(axis=1)
    predicted_sum = (predicted * mask_float).sum(axis=1)

    mean_abs_residual = _safe_divide(abs_residual_sum, mask_counts)
    mean_signed_residual = _safe_divide(signed_residual_sum, mask_counts)
    rmse = np.sqrt(_safe_divide(squared_residual_sum, mask_counts))
    mean_predicted_expression = _safe_divide(predicted_sum, mask_counts)

    masked_observations = mask_counts.sum(axis=1)
    genes_scored = (mask_counts > 0).sum(axis=1)
    sample_abs_residual_sum = abs_residual_sum.sum(axis=1)
    sample_sq_residual_sum = squared_residual_sum.sum(axis=1)
    mean_abs_residual_by_sample = _safe_divide(sample_abs_residual_sum, masked_observations)
    rmse_by_sample = np.sqrt(_safe_divide(sample_sq_residual_sum, masked_observations))
    median_gene_score = np.array(
        [
            float(np.nanmedian(mean_abs_residual[sample_idx][mask_counts[sample_idx] > 0]))
            for sample_idx in range(sample_count)
        ],
        dtype=float,
    )

    cohort_scores = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "valid_gene_count": valid_gene_count,
            "mc_passes": mc_passes,
            "masked_observations": masked_observations.astype(int),
            "genes_scored": genes_scored.astype(int),
            "gene_coverage_fraction": genes_scored / valid_gene_count,
            "mean_abs_residual": mean_abs_residual_by_sample,
            "rmse": rmse_by_sample,
            "median_gene_score": median_gene_score,
        }
    ).set_index("sample_id")

    ranked_gene_scores: dict[str, pd.DataFrame] = {}
    for sample_idx, sample_id in enumerate(sample_ids):
        sample_ranked = pd.DataFrame(
            {
                "ensg_id": gene_ids,
                "anomaly_score": mean_abs_residual[sample_idx],
                "mean_signed_residual": mean_signed_residual[sample_idx],
                "rmse": rmse[sample_idx],
                "masked_count": mask_counts[sample_idx].astype(int),
                "coverage_fraction": mask_counts[sample_idx] / mc_passes,
                "observed_expression": observed[sample_idx],
                "mean_predicted_expression": mean_predicted_expression[sample_idx],
            }
        )
        sample_ranked = sample_ranked.loc[sample_ranked["masked_count"] > 0].copy()
        sample_ranked = sample_ranked.sort_values(
            by=["anomaly_score", "masked_count", "ensg_id"],
            ascending=[False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        ranked_gene_scores[sample_id] = sample_ranked

    total_masked_count = mask_counts.sum(axis=0)
    sample_count_scored = (mask_counts > 0).sum(axis=0)
    gene_qc = pd.DataFrame(
        {
            "ensg_id": gene_ids,
            "is_valid": valid_gene_flags.astype(int),
            "masked_count": total_masked_count.astype(int),
            "sample_count_scored": sample_count_scored.astype(int),
            "sample_coverage_fraction": sample_count_scored / sample_count,
            "mean_abs_residual": _safe_divide(abs_residual_sum.sum(axis=0), total_masked_count),
            "rmse": np.sqrt(_safe_divide(squared_residual_sum.sum(axis=0), total_masked_count)),
        }
    )

    run_metadata = {
        "samples": sample_count,
        "genes": gene_count,
        "valid_gene_count": valid_gene_count,
        "invalid_gene_count": int((~valid_gene_flags).sum()),
        "mc_passes": mc_passes,
        "mask_prob": mask_prob,
        "fill_value": fill_value,
        "random_seed": random_seed,
        "mask_fractions": np.unique(row_mask_fractions).astype(float).tolist(),
    }
    return AnomalyScoringResult(
        cohort_scores=cohort_scores,
        gene_qc=gene_qc,
        ranked_gene_scores=ranked_gene_scores,
        run_metadata=run_metadata,
    )


def write_anomaly_outputs(result: AnomalyScoringResult, output_dir: Path) -> None:
    """Write anomaly scoring tables and QC outputs to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked_dir = output_dir / "ranked_genes"
    ranked_dir.mkdir(parents=True, exist_ok=True)

    result.cohort_scores.to_csv(output_dir / "cohort_scores.tsv", sep="\t")
    result.gene_qc.to_csv(output_dir / "gene_qc.tsv", sep="\t", index=False)
    written_filenames: set[str] = set()
    for sample_id, ranked_scores in result.ranked_gene_scores.items():
        filename = f"{_sanitize_filename(sample_id)}.tsv"
        if filename in written_filenames:
            raise ValueError(
                f"Multiple sample IDs resolved to the same ranked-gene filename {filename!r}."
            )
        written_filenames.add(filename)
        ranked_scores.to_csv(
            ranked_dir / filename,
            sep="\t",
            index=False,
        )

    with (output_dir / "anomaly_run.json").open("w", encoding="utf-8") as handle:
        json.dump(result.run_metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run(args: argparse.Namespace) -> int:
    """Execute Monte Carlo masking anomaly scoring."""
    expression = load_aligned_expression(Path(args.input))
    valid_gene_mask = load_valid_gene_mask(Path(args.valid_gene_mask))
    model_kwargs: dict[str, Any] = {
        "variant": args.variant,
        "checkpoint_path": args.checkpoint_path,
        "device": args.device,
    }
    if args.graph_path is not None:
        model_kwargs["graph_path"] = args.graph_path
    if args.graph_weights_path is not None:
        model_kwargs["graph_weights_path"] = args.graph_weights_path
    if args.gene_embedding_path is not None:
        model_kwargs["gene_embedding_path"] = args.gene_embedding_path
    if args.gene_info_path is not None:
        model_kwargs["gene_info_path"] = args.gene_info_path
    loaded_model = load_bulkformer_model(**model_kwargs)

    def predictor(masked_expression: np.ndarray, mask_fraction: float) -> np.ndarray:
        return predict_expression(
            loaded_model.model,
            masked_expression,
            batch_size=args.batch_size,
            mask_prob=mask_fraction,
            device=loaded_model.device,
        )

    result = score_expression_anomalies(
        expression,
        valid_gene_mask,
        predictor=predictor,
        mc_passes=args.mc_passes,
        mask_prob=args.mask_prob,
        fill_value=args.fill_value,
        random_seed=args.random_seed,
    )
    output_dir = Path(args.output_dir)
    write_anomaly_outputs(result, output_dir)
    print(f"Wrote anomaly scoring outputs to {output_dir}")
    print(
        "Samples: {samples} | valid genes: {valid_genes} | MC passes: {mc_passes} | "
        "mean cohort abs residual: {mean_abs:.4f}".format(
            samples=result.run_metadata["samples"],
            valid_genes=result.run_metadata["valid_gene_count"],
            mc_passes=result.run_metadata["mc_passes"],
            mean_abs=float(result.cohort_scores["mean_abs_residual"].mean()),
        )
    )
    return 0
