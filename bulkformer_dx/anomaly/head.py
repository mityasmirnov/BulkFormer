"""Small anomaly heads trained on top of frozen BulkFormer embeddings."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bulkformer_dx.bulkformer_model import (
    extract_gene_embeddings,
    load_bulkformer_model,
    predict_expression,
)

from .scoring import load_aligned_expression, load_valid_gene_mask, resolve_valid_gene_flags

DEFAULT_HEAD_MODE = "sigma_nll"
SUPPORTED_HEAD_MODES = ("sigma_nll", "injected_outlier")
DEFAULT_HIDDEN_DIM = 128
DEFAULT_EPOCHS = 30
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_MIN_SIGMA = 1e-3
DEFAULT_INJECTION_RATE = 0.05
DEFAULT_OUTLIER_SCALE = 3.0


@dataclass(slots=True)
class TrainedAnomalyHead:
    """Trained anomaly head plus compact training metadata."""

    mode: str
    model: nn.Module
    metrics: dict[str, float | int | str]
    input_dim: int
    hidden_dim: int


class MLPHead(nn.Module):
    """Small MLP used for both sigma/NLL and injected-outlier heads."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def _validate_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in SUPPORTED_HEAD_MODES:
        supported = ", ".join(SUPPORTED_HEAD_MODES)
        raise ValueError(f"Unsupported anomaly head mode {mode!r}. Expected one of: {supported}.")
    return normalized


def gaussian_nll_loss(
    predicted_mean: torch.Tensor,
    predicted_log_sigma: torch.Tensor,
    targets: torch.Tensor,
    *,
    min_sigma: float = DEFAULT_MIN_SIGMA,
) -> torch.Tensor:
    """Mean Gaussian NLL without the constant term."""
    if min_sigma <= 0:
        raise ValueError("min_sigma must be positive.")
    sigma = torch.exp(predicted_log_sigma).clamp_min(float(min_sigma))
    stabilized_log_sigma = torch.log(sigma)
    standardized_error = (targets - predicted_mean) / sigma
    per_example = 0.5 * (standardized_error.square() + 2.0 * stabilized_log_sigma)
    return per_example.mean()


def inject_synthetic_outliers(
    expression: np.ndarray,
    valid_gene_flags: np.ndarray,
    *,
    injection_rate: float = DEFAULT_INJECTION_RATE,
    outlier_scale: float = DEFAULT_OUTLIER_SCALE,
    random_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject synthetic outliers into valid genes only."""
    if not 0 < injection_rate <= 1:
        raise ValueError("injection_rate must be in the interval (0, 1].")
    if outlier_scale <= 0:
        raise ValueError("outlier_scale must be positive.")

    observed = np.asarray(expression, dtype=np.float32)
    if observed.ndim != 2:
        raise ValueError("Expression must be a 2D sample-by-gene matrix.")
    valid_gene_flags = np.asarray(valid_gene_flags, dtype=bool)
    if valid_gene_flags.ndim != 1 or valid_gene_flags.shape[0] != observed.shape[1]:
        raise ValueError("valid_gene_flags must be a 1D boolean vector aligned to expression columns.")

    valid_gene_indices = np.flatnonzero(valid_gene_flags)
    if valid_gene_indices.size == 0:
        raise ValueError("At least one valid gene is required to inject synthetic outliers.")

    rng = np.random.default_rng(random_seed)
    candidate_positions = [
        (sample_idx, gene_idx)
        for sample_idx in range(observed.shape[0])
        for gene_idx in valid_gene_indices
    ]
    injection_count = max(1, int(np.ceil(len(candidate_positions) * injection_rate)))
    selected_position_indices = rng.choice(
        len(candidate_positions),
        size=injection_count,
        replace=False,
    )

    perturbed = observed.copy()
    labels = np.zeros_like(observed, dtype=bool)
    gene_scales = observed[:, valid_gene_indices].std(axis=0, ddof=0)
    gene_scales = np.where(gene_scales > 0, gene_scales, 1.0)
    scale_by_gene_index = dict(zip(valid_gene_indices.tolist(), gene_scales.tolist(), strict=True))

    for position_idx in np.atleast_1d(selected_position_indices):
        sample_idx, gene_idx = candidate_positions[int(position_idx)]
        direction = float(rng.choice((-1.0, 1.0)))
        delta = direction * float(outlier_scale) * float(scale_by_gene_index[gene_idx])
        perturbed[sample_idx, gene_idx] = perturbed[sample_idx, gene_idx] + delta
        labels[sample_idx, gene_idx] = True

    return perturbed, labels


def _flatten_valid_gene_examples(
    embeddings: np.ndarray,
    values: np.ndarray,
    valid_gene_flags: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if embeddings.ndim != 3:
        raise ValueError("Embeddings must have shape [samples, genes, embedding_dim].")
    valid_embeddings = embeddings[:, valid_gene_flags, :]
    valid_values = values[:, valid_gene_flags]
    return (
        valid_embeddings.reshape(-1, valid_embeddings.shape[-1]).astype(np.float32, copy=False),
        valid_values.reshape(-1).astype(np.float32, copy=False),
    )


def prepare_sigma_nll_training_data(
    expression: np.ndarray,
    valid_gene_flags: np.ndarray,
    *,
    loaded_model: Any,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare flattened frozen-backbone features and residual targets."""
    embeddings = extract_gene_embeddings(
        loaded_model.model,
        expression,
        batch_size=batch_size,
        device=loaded_model.device,
    )
    predicted_expression = predict_expression(
        loaded_model.model,
        expression,
        batch_size=batch_size,
        device=loaded_model.device,
    )
    residual_targets = np.asarray(expression, dtype=np.float32) - np.asarray(
        predicted_expression,
        dtype=np.float32,
    )
    return _flatten_valid_gene_examples(embeddings, residual_targets, valid_gene_flags)


def prepare_injected_outlier_training_data(
    expression: np.ndarray,
    valid_gene_flags: np.ndarray,
    *,
    loaded_model: Any,
    batch_size: int,
    injection_rate: float,
    outlier_scale: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare frozen-backbone features and synthetic labels for injected outliers."""
    perturbed_expression, labels = inject_synthetic_outliers(
        expression,
        valid_gene_flags,
        injection_rate=injection_rate,
        outlier_scale=outlier_scale,
        random_seed=random_seed,
    )
    embeddings = extract_gene_embeddings(
        loaded_model.model,
        perturbed_expression,
        batch_size=batch_size,
        device=loaded_model.device,
    )
    return _flatten_valid_gene_examples(embeddings, labels.astype(np.float32), valid_gene_flags)


def train_head_model(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    mode: str = DEFAULT_HEAD_MODE,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    batch_size: int = 256,
    random_seed: int = 0,
    device: str | torch.device = "cpu",
    min_sigma: float = DEFAULT_MIN_SIGMA,
) -> TrainedAnomalyHead:
    """Train a small head on frozen BulkFormer features."""
    resolved_mode = _validate_mode(mode)
    feature_matrix = np.asarray(features, dtype=np.float32)
    target_vector = np.asarray(targets, dtype=np.float32).reshape(-1)

    if feature_matrix.ndim != 2:
        raise ValueError("features must be a 2D matrix.")
    if feature_matrix.shape[0] == 0:
        raise ValueError("features must contain at least one example.")
    if feature_matrix.shape[0] != target_vector.shape[0]:
        raise ValueError("features and targets must contain the same number of examples.")
    if hidden_dim <= 0 or epochs <= 0 or batch_size <= 0:
        raise ValueError("hidden_dim, epochs, and batch_size must be positive.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative.")

    torch.manual_seed(random_seed)
    resolved_device = torch.device(device)
    input_dim = int(feature_matrix.shape[1])
    output_dim = 2 if resolved_mode == "sigma_nll" else 1
    model = MLPHead(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(
        resolved_device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    feature_tensor = torch.as_tensor(feature_matrix, dtype=torch.float32)
    target_tensor = torch.as_tensor(target_vector, dtype=torch.float32)
    dataset = TensorDataset(feature_tensor, target_tensor)
    generator = torch.Generator()
    generator.manual_seed(random_seed)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
    )

    for _epoch_idx in range(epochs):
        model.train()
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(resolved_device)
            batch_targets = batch_targets.to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_features)
            if resolved_mode == "sigma_nll":
                predicted_mean = outputs[:, 0]
                predicted_log_sigma = outputs[:, 1]
                loss = gaussian_nll_loss(
                    predicted_mean,
                    predicted_log_sigma,
                    batch_targets,
                    min_sigma=min_sigma,
                )
            else:
                logits = outputs[:, 0]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch_targets)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.inference_mode():
        all_features = feature_tensor.to(resolved_device)
        outputs = model(all_features)
        if resolved_mode == "sigma_nll":
            predicted_mean = outputs[:, 0]
            predicted_log_sigma = outputs[:, 1]
            sigma = torch.exp(predicted_log_sigma).clamp_min(min_sigma)
            metrics: dict[str, float | int | str] = {
                "mode": resolved_mode,
                "train_examples": int(feature_matrix.shape[0]),
                "mean_nll": float(
                    gaussian_nll_loss(
                        predicted_mean,
                        predicted_log_sigma,
                        target_tensor.to(resolved_device),
                        min_sigma=min_sigma,
                    ).item()
                ),
                "train_rmse": float(
                    torch.sqrt(
                        torch.mean((predicted_mean - target_tensor.to(resolved_device)).square())
                    ).item()
                ),
                "mean_predicted_sigma": float(sigma.mean().item()),
            }
        else:
            logits = outputs[:, 0]
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).to(torch.float32)
            train_targets = target_tensor.to(resolved_device)
            metrics = {
                "mode": resolved_mode,
                "train_examples": int(feature_matrix.shape[0]),
                "train_accuracy": float((predictions == train_targets).to(torch.float32).mean().item()),
                "positive_rate": float(train_targets.mean().item()),
            }

    return TrainedAnomalyHead(
        mode=resolved_mode,
        model=model.cpu(),
        metrics=metrics,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
    )


def _artifact_filename_for_mode(mode: str) -> str:
    return f"{mode}_head.pt"


def save_trained_head(head_result: TrainedAnomalyHead, output_dir: Path) -> tuple[Path, Path]:
    """Persist the trained head checkpoint plus JSON metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / _artifact_filename_for_mode(head_result.mode)
    metrics_path = output_dir / "training_metrics.json"
    torch.save(
        {
            "mode": head_result.mode,
            "input_dim": head_result.input_dim,
            "hidden_dim": head_result.hidden_dim,
            "metrics": head_result.metrics,
            "state_dict": head_result.model.state_dict(),
        },
        checkpoint_path,
    )
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(head_result.metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return checkpoint_path, metrics_path


def run(args: argparse.Namespace) -> int:
    """Train an anomaly head on frozen BulkFormer features."""
    input_path = getattr(args, "input", None)
    if input_path is None:
        raise ValueError("An aligned expression input table is required.")
    valid_gene_mask_path = getattr(args, "valid_gene_mask", None)
    if valid_gene_mask_path is None:
        raise ValueError("A valid_gene_mask table is required.")

    expression = load_aligned_expression(Path(input_path))
    valid_gene_mask = load_valid_gene_mask(Path(valid_gene_mask_path))
    valid_gene_flags = resolve_valid_gene_flags(valid_gene_mask, expression.columns)

    model_kwargs: dict[str, Any] = {
        "variant": getattr(args, "variant", None),
        "checkpoint_path": getattr(args, "checkpoint_path", None),
        "device": getattr(args, "device", "cpu"),
    }
    for attr_name in ("graph_path", "graph_weights_path", "gene_embedding_path", "gene_info_path"):
        attr_value = getattr(args, attr_name, None)
        if attr_value is not None:
            model_kwargs[attr_name] = attr_value
    loaded_model = load_bulkformer_model(**model_kwargs)

    resolved_mode = _validate_mode(getattr(args, "mode", DEFAULT_HEAD_MODE))
    observed_expression = expression.to_numpy(dtype=np.float32, copy=True)
    if resolved_mode == "sigma_nll":
        features, targets = prepare_sigma_nll_training_data(
            observed_expression,
            valid_gene_flags,
            loaded_model=loaded_model,
            batch_size=args.batch_size,
        )
    else:
        features, targets = prepare_injected_outlier_training_data(
            observed_expression,
            valid_gene_flags,
            loaded_model=loaded_model,
            batch_size=args.batch_size,
            injection_rate=args.injection_rate,
            outlier_scale=args.outlier_scale,
            random_seed=args.random_seed,
        )

    head_result = train_head_model(
        features,
        targets,
        mode=resolved_mode,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        random_seed=args.random_seed,
        device=args.device,
        min_sigma=args.min_sigma,
    )
    checkpoint_path, metrics_path = save_trained_head(head_result, Path(args.output_dir))
    print(f"Wrote {resolved_mode} head checkpoint to {checkpoint_path}")
    print(f"Wrote training metrics to {metrics_path}")
    return 0
