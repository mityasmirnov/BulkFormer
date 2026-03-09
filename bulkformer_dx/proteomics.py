"""Frozen-backbone proteomics training and prediction workflows."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from math import erfc, sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bulkformer_dx.anomaly.calibration import benjamini_yekutieli
from bulkformer_dx.anomaly.scoring import load_aligned_expression
from bulkformer_dx.bulkformer_model import extract_sample_embeddings, load_bulkformer_model
from bulkformer_dx.tissue import resolve_selected_gene_ids

SUPPORTED_MODES = ("train", "predict")
SUPPORTED_HEAD_TYPES = ("linear", "mlp")
DEFAULT_HEAD_TYPE = "linear"
DEFAULT_AGGREGATION = "mean"
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_HIDDEN_DIM = 256
DEFAULT_VAL_FRACTION = 0.2
DEFAULT_PATIENCE = 10
DEFAULT_ALPHA = 0.05


@dataclass(slots=True)
class TrainedProteomicsHead:
    """Trained proteomics head plus compact training metadata."""

    head_type: str
    model: nn.Module
    metrics: dict[str, Any]
    input_dim: int
    output_dim: int
    hidden_dim: int | None


class ProteinHeadLinear(nn.Module):
    """Single-layer mapping from transcriptome embedding to protein intensities."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features)


class ProteinHeadMLP(nn.Module):
    """Shallow two-layer MLP head for proteomics prediction."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    separator = "\t" if suffix in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=separator)


def load_proteomics_table(path: Path) -> pd.DataFrame:
    """Load a sample-by-protein intensity table keyed by sample_id."""
    table = _read_table(path)
    if table.empty:
        raise ValueError("The proteomics table is empty.")
    sample_column = str(table.columns[0])
    table = table.set_index(sample_column)
    table.index = table.index.astype(str)
    return table.apply(pd.to_numeric, errors="coerce")


def align_expression_and_proteomics(
    expression: pd.DataFrame,
    proteomics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align RNA and proteomics tables on the shared sample IDs."""
    shared_samples = [sample_id for sample_id in expression.index if sample_id in proteomics.index]
    if not shared_samples:
        raise ValueError("RNA and proteomics tables did not share any sample IDs.")
    aligned_expression = expression.loc[shared_samples].copy()
    aligned_proteomics = proteomics.loc[shared_samples].copy()
    return aligned_expression, aligned_proteomics


def _positive_only_log2(values: np.ndarray) -> np.ndarray:
    logged = np.full(values.shape, np.nan, dtype=np.float32)
    finite_positive = np.isfinite(values) & (values > 0)
    logged[finite_positive] = np.log2(values[finite_positive]).astype(np.float32, copy=False)
    return logged


def transform_proteomics_targets(
    proteomics: pd.DataFrame,
    *,
    log2_transform: bool = False,
    already_log2: bool = False,
    center_scale: bool = False,
    fit_statistics: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert proteomics intensities into the numeric target space used by the head."""
    if log2_transform and already_log2:
        raise ValueError("Use either log2_transform or already_log2, not both.")

    values = proteomics.to_numpy(dtype=np.float32, copy=True)
    if fit_statistics is None:
        resolved_log2 = bool(log2_transform)
        if resolved_log2:
            values = _positive_only_log2(values)
        means = np.nanmean(values, axis=0)
        means = np.where(np.isfinite(means), means, 0.0).astype(np.float32, copy=False)
        stds = np.nanstd(values, axis=0)
        stds = np.where(np.isfinite(stds) & (stds > 1e-6), stds, 1.0).astype(np.float32, copy=False)
        stats = {
            "log2_transform": resolved_log2,
            "already_log2": bool(already_log2),
            "center_scale": bool(center_scale),
            "means": means.tolist(),
            "stds": stds.tolist(),
        }
    else:
        resolved_log2 = bool(fit_statistics.get("log2_transform", False))
        if resolved_log2:
            values = _positive_only_log2(values)
        means = np.asarray(fit_statistics["means"], dtype=np.float32)
        stds = np.asarray(fit_statistics["stds"], dtype=np.float32)
        stats = dict(fit_statistics)

    if center_scale or bool(stats.get("center_scale", False)):
        values = (values - means) / stds

    transformed = pd.DataFrame(values, index=proteomics.index, columns=proteomics.columns)
    return transformed, stats


def invert_transformed_targets(
    values: np.ndarray,
    *,
    transform_stats: dict[str, Any],
) -> np.ndarray:
    """Map model outputs back into the final reported log2 protein space."""
    restored = np.asarray(values, dtype=np.float32)
    if bool(transform_stats.get("center_scale", False)):
        means = np.asarray(transform_stats["means"], dtype=np.float32)
        stds = np.asarray(transform_stats["stds"], dtype=np.float32)
        restored = restored * stds + means
    return restored


def masked_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error over finite protein targets only."""
    squared_error = (predictions - targets).square() * target_mask
    denominator = target_mask.sum().clamp_min(1.0)
    return squared_error.sum() / denominator


def _build_head(
    *,
    input_dim: int,
    output_dim: int,
    head_type: str,
    hidden_dim: int,
) -> nn.Module:
    if head_type == "linear":
        return ProteinHeadLinear(input_dim=input_dim, output_dim=output_dim)
    if head_type == "mlp":
        return ProteinHeadMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
        )
    supported = ", ".join(SUPPORTED_HEAD_TYPES)
    raise ValueError(f"Unsupported proteomics head type {head_type!r}. Use one of: {supported}.")


def _split_train_val_indices(
    sample_count: int,
    *,
    val_fraction: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must lie in the interval [0, 1).")
    if sample_count < 3 or val_fraction == 0:
        indices = np.arange(sample_count, dtype=int)
        return indices, np.empty(0, dtype=int)

    rng = np.random.default_rng(random_seed)
    indices = rng.permutation(sample_count)
    val_count = min(sample_count - 1, max(1, int(np.floor(sample_count * val_fraction))))
    val_indices = np.sort(indices[:val_count])
    train_indices = np.sort(indices[val_count:])
    return train_indices, val_indices


def train_proteomics_head(
    sample_embeddings: np.ndarray,
    targets: np.ndarray,
    *,
    head_type: str = DEFAULT_HEAD_TYPE,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    batch_size: int = DEFAULT_BATCH_SIZE,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    patience: int = DEFAULT_PATIENCE,
    random_seed: int = 0,
    device: str | torch.device = "cpu",
) -> TrainedProteomicsHead:
    """Fit a shallow proteomics head with masked loss and early stopping."""
    features = np.asarray(sample_embeddings, dtype=np.float32)
    target_values = np.asarray(targets, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("sample_embeddings must be a 2D matrix.")
    if target_values.ndim != 2:
        raise ValueError("targets must be a 2D sample-by-protein matrix.")
    if features.shape[0] != target_values.shape[0]:
        raise ValueError("sample_embeddings and targets must contain the same number of samples.")
    if features.shape[0] == 0:
        raise ValueError("At least one training sample is required.")
    if head_type not in SUPPORTED_HEAD_TYPES:
        supported = ", ".join(SUPPORTED_HEAD_TYPES)
        raise ValueError(f"Unsupported proteomics head type {head_type!r}. Use one of: {supported}.")

    torch.manual_seed(random_seed)
    resolved_device = torch.device(device)
    input_dim = int(features.shape[1])
    output_dim = int(target_values.shape[1])
    model = _build_head(
        input_dim=input_dim,
        output_dim=output_dim,
        head_type=head_type,
        hidden_dim=hidden_dim,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    train_indices, val_indices = _split_train_val_indices(
        features.shape[0],
        val_fraction=val_fraction,
        random_seed=random_seed,
    )
    feature_tensor = torch.as_tensor(features, dtype=torch.float32)
    target_tensor = torch.as_tensor(np.nan_to_num(target_values, nan=0.0), dtype=torch.float32)
    target_mask = torch.as_tensor(np.isfinite(target_values), dtype=torch.float32)

    generator = torch.Generator()
    generator.manual_seed(random_seed)
    train_dataset = TensorDataset(
        feature_tensor[train_indices],
        target_tensor[train_indices],
        target_mask[train_indices],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),
        shuffle=True,
        generator=generator,
    )

    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_epoch = 0
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    epochs_without_improvement = 0

    for epoch_idx in range(epochs):
        model.train()
        epoch_losses: list[float] = []
        for batch_features, batch_targets, batch_mask in train_loader:
            batch_features = batch_features.to(resolved_device)
            batch_targets = batch_targets.to(resolved_device)
            batch_mask = batch_mask.to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_features)
            loss = masked_mse_loss(predictions, batch_targets, batch_mask)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        train_loss = float(np.mean(epoch_losses))
        monitor_loss = train_loss
        if val_indices.size > 0:
            model.eval()
            with torch.inference_mode():
                val_predictions = model(feature_tensor[val_indices].to(resolved_device))
                val_loss = float(
                    masked_mse_loss(
                        val_predictions,
                        target_tensor[val_indices].to(resolved_device),
                        target_mask[val_indices].to(resolved_device),
                    ).item()
                )
            monitor_loss = val_loss
        else:
            val_loss = float("nan")

        if monitor_loss < best_val_loss - 1e-8:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch_idx + 1
            best_val_loss = monitor_loss
            best_train_loss = train_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if val_indices.size > 0 and epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    model = model.cpu().eval()

    metrics: dict[str, Any] = {
        "head_type": head_type,
        "samples": int(features.shape[0]),
        "proteins": output_dim,
        "input_dim": input_dim,
        "train_samples": int(train_indices.size),
        "val_samples": int(val_indices.size),
        "best_epoch": int(best_epoch),
        "best_train_loss": float(best_train_loss),
        "best_val_loss": None if not np.isfinite(best_val_loss) else float(best_val_loss),
    }
    return TrainedProteomicsHead(
        head_type=head_type,
        model=model,
        metrics=metrics,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=None if head_type == "linear" else int(hidden_dim),
    )


def predict_proteomics_targets(
    model: nn.Module,
    sample_embeddings: np.ndarray,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Run a trained proteomics head over sample embeddings."""
    features = torch.as_tensor(np.asarray(sample_embeddings, dtype=np.float32))
    resolved_device = torch.device(device)
    model = model.to(resolved_device).eval()
    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for start_idx in range(0, features.shape[0], batch_size):
            batch = features[start_idx : start_idx + batch_size].to(resolved_device)
            outputs.append(model(batch).detach().cpu())
    return torch.cat(outputs, dim=0).numpy()


def _proteinwise_robust_statistics(residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centers = np.nanmedian(residuals, axis=0)
    mad = np.nanmedian(np.abs(residuals - centers), axis=0)
    scales = 1.4826 * mad
    finite_scale = np.isfinite(scales) & (scales > 1e-8)
    fallback = np.nanstd(residuals, axis=0)
    fallback = np.where(np.isfinite(fallback) & (fallback > 1e-8), fallback, 1.0)
    scales = np.where(finite_scale, scales, fallback)
    centers = np.where(np.isfinite(centers), centers, 0.0)
    return centers.astype(np.float32), scales.astype(np.float32)


def calibrate_proteomics_residuals(
    residuals: pd.DataFrame,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute protein-wise robust p-values and per-sample BY-adjusted q-values."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in the interval (0, 1).")

    residual_values = residuals.to_numpy(dtype=np.float32, copy=True)
    centers, scales = _proteinwise_robust_statistics(residual_values)
    z_scores = (residual_values - centers) / scales
    p_values = np.full(z_scores.shape, np.nan, dtype=np.float32)
    finite_mask = np.isfinite(z_scores)
    p_values[finite_mask] = np.asarray(
        [erfc(abs(float(z_value)) / sqrt(2.0)) for z_value in z_scores[finite_mask]],
        dtype=np.float32,
    )
    q_values = np.full(p_values.shape, np.nan, dtype=np.float32)
    for row_idx in range(p_values.shape[0]):
        finite_row = np.isfinite(p_values[row_idx])
        if finite_row.any():
            q_values[row_idx, finite_row] = benjamini_yekutieli(p_values[row_idx, finite_row])
    return (
        pd.DataFrame(p_values, index=residuals.index, columns=residuals.columns),
        pd.DataFrame(q_values, index=residuals.index, columns=residuals.columns),
    )


def build_ranked_protein_tables(
    predicted: pd.DataFrame,
    observed: pd.DataFrame | None,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame | None, pd.DataFrame | None]:
    """Build per-sample ranked protein residual tables."""
    if observed is None:
        ranked = {
            str(sample_id): pd.DataFrame(
                {
                    "protein_id": predicted.columns.astype(str),
                    "predicted_log2_intensity": predicted.loc[sample_id].to_numpy(dtype=float),
                }
            )
            for sample_id in predicted.index
        }
        return ranked, None, None

    residuals = observed - predicted
    p_values, q_values = calibrate_proteomics_residuals(residuals, alpha=alpha)
    ranked_tables: dict[str, pd.DataFrame] = {}
    for sample_id in predicted.index:
        sample_ranked = pd.DataFrame(
            {
                "protein_id": predicted.columns.astype(str),
                "predicted_log2_intensity": predicted.loc[sample_id].to_numpy(dtype=float),
                "observed_log2_intensity": observed.loc[sample_id].to_numpy(dtype=float),
                "residual": residuals.loc[sample_id].to_numpy(dtype=float),
                "abs_residual": residuals.loc[sample_id].abs().to_numpy(dtype=float),
                "p_value": p_values.loc[sample_id].to_numpy(dtype=float),
                "padj": q_values.loc[sample_id].to_numpy(dtype=float),
            }
        )
        sample_ranked["call"] = (
            sample_ranked["padj"].notna() & (sample_ranked["padj"] <= alpha)
        ).astype(int)
        sample_ranked = sample_ranked.sort_values(
            by=["abs_residual", "protein_id"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        sample_ranked["rank_within_sample"] = np.arange(1, len(sample_ranked) + 1)
        ranked_tables[str(sample_id)] = sample_ranked
    return ranked_tables, residuals, q_values


def _optional_path_string(value: object) -> str | None:
    if value is None:
        return None
    return str(Path(value))


def _resolve_model_contract(
    loaded_model: Any,
    model_kwargs: dict[str, Any],
) -> dict[str, str | None]:
    assets = getattr(loaded_model, "assets", None)
    if assets is None:
        return {
            "variant": None if model_kwargs.get("variant") is None else str(model_kwargs.get("variant")),
            "checkpoint_path": _optional_path_string(model_kwargs.get("checkpoint_path")),
            "graph_path": _optional_path_string(model_kwargs.get("graph_path")),
            "graph_weights_path": _optional_path_string(model_kwargs.get("graph_weights_path")),
            "gene_embedding_path": _optional_path_string(model_kwargs.get("gene_embedding_path")),
            "gene_info_path": _optional_path_string(model_kwargs.get("gene_info_path")),
        }
    return {
        "variant": str(getattr(assets, "variant", model_kwargs.get("variant"))),
        "checkpoint_path": _optional_path_string(getattr(assets, "checkpoint_path", None)),
        "graph_path": _optional_path_string(getattr(assets, "graph_path", None)),
        "graph_weights_path": _optional_path_string(getattr(assets, "graph_weights_path", None)),
        "gene_embedding_path": _optional_path_string(getattr(assets, "gene_embedding_path", None)),
        "gene_info_path": _optional_path_string(getattr(assets, "gene_info_path", None)),
    }


def _resolve_prediction_model_kwargs(
    args: argparse.Namespace,
    stored_contract: dict[str, str | None],
) -> dict[str, Any]:
    resolved: dict[str, Any] = {"device": getattr(args, "device", "cpu")}
    for field_name in (
        "variant",
        "checkpoint_path",
        "graph_path",
        "graph_weights_path",
        "gene_embedding_path",
        "gene_info_path",
    ):
        explicit_value = getattr(args, field_name, None)
        stored_value = stored_contract.get(field_name)
        if explicit_value is not None and stored_value is not None and str(explicit_value) != stored_value:
            raise ValueError(
                f"Prediction {field_name}={explicit_value!r} does not match the training artifact "
                f"value {stored_value!r}."
            )
        resolved_value = explicit_value if explicit_value is not None else stored_value
        if resolved_value is not None:
            resolved[field_name] = resolved_value
    return resolved


def extract_proteomics_embeddings(
    expression: pd.DataFrame,
    *,
    selected_gene_ids: list[str],
    aggregation: str,
    batch_size: int,
    model_kwargs: dict[str, Any],
) -> tuple[np.ndarray, dict[str, str | None]]:
    """Load BulkFormer once and derive sample-level transcriptome embeddings."""
    loaded_model = load_bulkformer_model(**model_kwargs)
    gene_indices = expression.columns.get_indexer(selected_gene_ids).tolist()
    if any(gene_index < 0 for gene_index in gene_indices):
        raise ValueError("Selected proteomics genes could not be aligned to the expression matrix.")
    embeddings = extract_sample_embeddings(
        loaded_model.model,
        expression,
        batch_size=batch_size,
        aggregation=aggregation,
        device=loaded_model.device,
        gene_indices=gene_indices,
    )
    return embeddings, _resolve_model_contract(loaded_model, model_kwargs)


def save_proteomics_artifact(
    head_result: TrainedProteomicsHead,
    output_dir: Path,
    *,
    protein_ids: list[str],
    selected_gene_ids: list[str],
    aggregation: str,
    transform_stats: dict[str, Any],
    model_contract: dict[str, str | None],
) -> Path:
    """Persist the trained proteomics head and its feature contract."""
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "protein_head.pt"
    torch.save(
        {
            "head_type": head_result.head_type,
            "input_dim": head_result.input_dim,
            "output_dim": head_result.output_dim,
            "hidden_dim": head_result.hidden_dim,
            "metrics": head_result.metrics,
            "state_dict": head_result.model.state_dict(),
            "feature_spec": {
                "protein_ids": protein_ids,
                "selected_gene_ids": selected_gene_ids,
                "aggregation": aggregation,
                "transform_stats": transform_stats,
                "model_contract": model_contract,
            },
        },
        artifact_path,
    )
    return artifact_path


def load_proteomics_artifact(path: Path) -> dict[str, Any]:
    """Load a saved proteomics artifact bundle."""
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(artifact, dict) or "state_dict" not in artifact or "feature_spec" not in artifact:
        raise ValueError(f"Proteomics artifact at {path} did not contain the expected keys.")
    return artifact


def build_model_from_artifact(artifact: dict[str, Any]) -> nn.Module:
    """Recreate a proteomics head from its serialized metadata."""
    model = _build_head(
        input_dim=int(artifact["input_dim"]),
        output_dim=int(artifact["output_dim"]),
        head_type=str(artifact["head_type"]),
        hidden_dim=int(artifact.get("hidden_dim") or DEFAULT_HIDDEN_DIM),
    )
    model.load_state_dict(artifact["state_dict"])
    return model.eval()


def write_proteomics_outputs(
    *,
    output_dir: Path,
    predicted: pd.DataFrame,
    observed: pd.DataFrame | None,
    ranked_tables: dict[str, pd.DataFrame],
    artifact_path: Path | None,
    metrics: dict[str, Any],
    residuals: pd.DataFrame | None,
    q_values: pd.DataFrame | None,
) -> None:
    """Write proteomics predictions, rankings, and metadata to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked_dir = output_dir / "ranked_proteins"
    ranked_dir.mkdir(parents=True, exist_ok=True)

    predicted.to_csv(output_dir / "predicted_proteomics.tsv", sep="\t")
    if observed is not None:
        observed.to_csv(output_dir / "observed_proteomics.tsv", sep="\t")
    if residuals is not None:
        residuals.to_csv(output_dir / "residuals.tsv", sep="\t")
    if q_values is not None:
        q_values.to_csv(output_dir / "q_values.tsv", sep="\t")

    summary_rows: list[dict[str, Any]] = []
    for sample_id, ranked_table in ranked_tables.items():
        ranked_table.to_csv(ranked_dir / f"{sample_id}.tsv", sep="\t", index=False)
        summary_rows.append(
            {
                "sample_id": sample_id,
                "proteins_ranked": int(len(ranked_table)),
                "top_abs_residual": None
                if "abs_residual" not in ranked_table.columns or ranked_table["abs_residual"].isna().all()
                else float(ranked_table["abs_residual"].max()),
                "top_significant_proteins": int(ranked_table.get("call", pd.Series(dtype=int)).sum())
                if "call" in ranked_table.columns
                else 0,
            }
        )
    pd.DataFrame(summary_rows).to_csv(output_dir / "proteomics_summary.tsv", sep="\t", index=False)

    run_metadata = dict(metrics)
    run_metadata["artifact_path"] = None if artifact_path is None else str(artifact_path)
    with (output_dir / "prediction_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _prepare_model_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    model_kwargs: dict[str, Any] = {
        "variant": getattr(args, "variant", None),
        "checkpoint_path": getattr(args, "checkpoint_path", None),
        "device": getattr(args, "device", "cpu"),
    }
    for attr_name in ("graph_path", "graph_weights_path", "gene_embedding_path", "gene_info_path"):
        attr_value = getattr(args, attr_name, None)
        if attr_value is not None:
            model_kwargs[attr_name] = attr_value
    return model_kwargs


def _run_prediction_from_artifact(
    expression: pd.DataFrame,
    observed_proteomics: pd.DataFrame | None,
    *,
    artifact: dict[str, Any],
    artifact_path: Path,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, pd.DataFrame], dict[str, Any], pd.DataFrame | None, pd.DataFrame | None]:
    feature_spec = artifact["feature_spec"]
    selected_gene_ids = [str(gene_id) for gene_id in feature_spec["selected_gene_ids"]]
    aggregation = str(feature_spec.get("aggregation", DEFAULT_AGGREGATION))
    transform_stats = dict(feature_spec["transform_stats"])
    model_kwargs = _resolve_prediction_model_kwargs(args, feature_spec.get("model_contract", {}))
    embeddings, _ = extract_proteomics_embeddings(
        expression,
        selected_gene_ids=selected_gene_ids,
        aggregation=aggregation,
        batch_size=args.batch_size,
        model_kwargs=model_kwargs,
    )
    model = build_model_from_artifact(artifact)
    predicted_values = predict_proteomics_targets(
        model,
        embeddings,
        batch_size=args.batch_size,
        device=args.device,
    )
    predicted_values = invert_transformed_targets(predicted_values, transform_stats=transform_stats)
    protein_ids = [str(protein_id) for protein_id in feature_spec["protein_ids"]]
    predicted = pd.DataFrame(predicted_values, index=expression.index, columns=protein_ids)

    observed_log2: pd.DataFrame | None = None
    if observed_proteomics is not None:
        aligned_proteomics = observed_proteomics.reindex(expression.index)[protein_ids]
        observed_log2, _ = transform_proteomics_targets(
            aligned_proteomics,
            fit_statistics=transform_stats,
        )
        observed_log2 = pd.DataFrame(
            invert_transformed_targets(observed_log2.to_numpy(dtype=np.float32), transform_stats=transform_stats),
            index=aligned_proteomics.index,
            columns=aligned_proteomics.columns,
        )

    ranked_tables, residuals, q_values = build_ranked_protein_tables(
        predicted,
        observed_log2,
        alpha=args.alpha,
    )
    metrics = {
        **artifact.get("metrics", {}),
        "samples": int(predicted.shape[0]),
        "proteins": int(predicted.shape[1]),
        "mode": "predict",
        "artifact_path": str(artifact_path),
    }
    return predicted, observed_log2, ranked_tables, metrics, residuals, q_values


def register_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the proteomics command group."""
    parser = subparsers.add_parser(
        "proteomics",
        help="Train or run proteomics heads on BulkFormer embeddings.",
        description=(
            "Use frozen BulkFormer transcriptome embeddings to predict proteomics, "
            "rank protein residual outliers, and optionally calibrate them with "
            "per-sample BY correction."
        ),
    )
    parser.add_argument("mode", choices=SUPPORTED_MODES, help="Whether to train or predict.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a BulkFormer-aligned sample-by-gene RNA expression table.",
    )
    parser.add_argument(
        "--proteomics",
        help="Path to a sample-by-protein intensity table. Required for train mode.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where model artifacts and proteomics outputs should be written.",
    )
    parser.add_argument(
        "--artifact-path",
        help="Path to a saved protein_head.pt artifact. Required for predict mode.",
    )
    parser.add_argument(
        "--valid-gene-mask",
        help="Optional valid_gene_mask.tsv file to restrict BulkFormer embeddings to observed genes.",
    )
    parser.add_argument(
        "--aggregation",
        default=DEFAULT_AGGREGATION,
        help="Sample embedding aggregation. Defaults to mean.",
    )
    parser.add_argument(
        "--head-type",
        choices=SUPPORTED_HEAD_TYPES,
        default=DEFAULT_HEAD_TYPE,
        help="Proteomics head architecture.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=DEFAULT_HIDDEN_DIM,
        help="Hidden dimension for the MLP head.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate for AdamW.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="Weight decay for AdamW.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size used for embedding extraction and head training.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=DEFAULT_VAL_FRACTION,
        help="Validation fraction used for early stopping during training.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help="Early-stopping patience in epochs.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed for training and sample splitting.",
    )
    parser.add_argument(
        "--log2-transform",
        action="store_true",
        help="Log2-transform positive proteomics intensities before training or scoring.",
    )
    parser.add_argument(
        "--already-log2",
        action="store_true",
        help="Declare that the proteomics table is already in log2 space.",
    )
    parser.add_argument(
        "--center-scale",
        action="store_true",
        help="Center and scale proteins using training-set statistics.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="BY-adjusted significance threshold for protein calls.",
    )
    parser.add_argument("--variant", help="Preferred BulkFormer checkpoint variant.")
    parser.add_argument("--checkpoint-path", help="Explicit BulkFormer checkpoint path.")
    parser.add_argument("--graph-path", help="Optional BulkFormer graph asset path.")
    parser.add_argument("--graph-weights-path", help="Optional BulkFormer graph weight path.")
    parser.add_argument("--gene-embedding-path", help="Optional BulkFormer gene embedding path.")
    parser.add_argument("--gene-info-path", help="Optional BulkFormer gene info path.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for embedding extraction and prediction.",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    """Execute proteomics training or prediction."""
    if args.mode not in SUPPORTED_MODES:
        supported = ", ".join(SUPPORTED_MODES)
        raise ValueError(f"Unsupported proteomics mode {args.mode!r}. Use one of: {supported}.")

    expression = load_aligned_expression(Path(args.input))
    if args.mode == "train" and args.proteomics is None:
        raise ValueError("A proteomics intensity table is required for train mode.")

    model_kwargs = _prepare_model_kwargs(args)

    if args.mode == "train":
        proteomics = load_proteomics_table(Path(args.proteomics))
        expression, proteomics = align_expression_and_proteomics(expression, proteomics)
        selected_gene_ids = resolve_selected_gene_ids(
            expression,
            valid_gene_mask_path=getattr(args, "valid_gene_mask", None),
        )
        sample_embeddings, model_contract = extract_proteomics_embeddings(
            expression,
            selected_gene_ids=selected_gene_ids,
            aggregation=args.aggregation,
            batch_size=args.batch_size,
            model_kwargs=model_kwargs,
        )
        transformed_targets, transform_stats = transform_proteomics_targets(
            proteomics,
            log2_transform=args.log2_transform,
            already_log2=args.already_log2,
            center_scale=args.center_scale,
        )
        head_result = train_proteomics_head(
            sample_embeddings,
            transformed_targets.to_numpy(dtype=np.float32),
            head_type=args.head_type,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            val_fraction=args.val_fraction,
            patience=args.patience,
            random_seed=args.random_seed,
            device=args.device,
        )
        artifact_path = save_proteomics_artifact(
            head_result,
            Path(args.output_dir),
            protein_ids=proteomics.columns.astype(str).tolist(),
            selected_gene_ids=selected_gene_ids,
            aggregation=args.aggregation,
            transform_stats=transform_stats,
            model_contract=model_contract,
        )
        predicted_values = predict_proteomics_targets(
            head_result.model,
            sample_embeddings,
            batch_size=args.batch_size,
            device=args.device,
        )
        predicted_values = invert_transformed_targets(predicted_values, transform_stats=transform_stats)
        observed_log2 = pd.DataFrame(
            invert_transformed_targets(
                transformed_targets.to_numpy(dtype=np.float32),
                transform_stats=transform_stats,
            ),
            index=proteomics.index,
            columns=proteomics.columns,
        )
        predicted = pd.DataFrame(predicted_values, index=proteomics.index, columns=proteomics.columns)
        ranked_tables, residuals, q_values = build_ranked_protein_tables(
            predicted,
            observed_log2,
            alpha=args.alpha,
        )
        metrics = {
            **head_result.metrics,
            "mode": "train",
            "aggregation": args.aggregation,
            "selected_genes": len(selected_gene_ids),
            "alpha": args.alpha,
        }
        write_proteomics_outputs(
            output_dir=Path(args.output_dir),
            predicted=predicted,
            observed=observed_log2,
            ranked_tables=ranked_tables,
            artifact_path=artifact_path,
            metrics=metrics,
            residuals=residuals,
            q_values=q_values,
        )
        print(f"Wrote proteomics head artifact to {artifact_path}")
        print(f"Wrote proteomics outputs to {Path(args.output_dir)}")
        return 0

    if args.artifact_path is None:
        raise ValueError("An artifact path is required for predict mode.")
    artifact_path = Path(args.artifact_path)
    artifact = load_proteomics_artifact(artifact_path)
    observed_proteomics = (
        None if args.proteomics is None else load_proteomics_table(Path(args.proteomics))
    )
    if observed_proteomics is not None:
        expression, observed_proteomics = align_expression_and_proteomics(expression, observed_proteomics)
    predicted, observed, ranked_tables, metrics, residuals, q_values = _run_prediction_from_artifact(
        expression,
        observed_proteomics,
        artifact=artifact,
        artifact_path=artifact_path,
        args=args,
    )
    write_proteomics_outputs(
        output_dir=Path(args.output_dir),
        predicted=predicted,
        observed=observed,
        ranked_tables=ranked_tables,
        artifact_path=artifact_path,
        metrics=metrics,
        residuals=residuals,
        q_values=q_values,
    )
    print(f"Wrote proteomics predictions to {Path(args.output_dir)}")
    return 0
