"""Tissue train/predict workflows driven by BulkFormer sample embeddings."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from bulkformer_dx.anomaly.scoring import (
    load_aligned_expression,
    load_valid_gene_mask,
    resolve_valid_gene_flags,
)
from bulkformer_dx.bulkformer_model import extract_sample_embeddings, load_bulkformer_model

DEFAULT_AGGREGATION = "mean"
DEFAULT_N_ESTIMATORS = 256
SUPPORTED_MODES = ("train", "predict")


@dataclass(slots=True)
class TrainedTissueClassifier:
    """Trained sklearn pipeline plus compact training metadata."""

    pipeline: Pipeline
    metrics: dict[str, Any]


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    separator = "\t" if suffix in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=separator)


def _optional_path_string(value: object) -> str | None:
    if value is None:
        return None
    return str(Path(value))


def load_tissue_labels(path: Path) -> pd.Series:
    """Load sample-to-tissue labels keyed by sample ID."""
    labels = _read_table(path)
    required_columns = {"sample_id", "tissue_label"}
    missing_columns = required_columns - set(labels.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Tissue label table is missing required columns: {missing_list}.")
    if labels["sample_id"].duplicated().any():
        raise ValueError("Tissue label table contains duplicate sample IDs.")

    resolved = labels.loc[:, ["sample_id", "tissue_label"]].copy()
    resolved["sample_id"] = resolved["sample_id"].astype(str)
    resolved["tissue_label"] = resolved["tissue_label"].astype(str).str.strip()
    if (resolved["tissue_label"] == "").any():
        raise ValueError("Tissue label table contains empty tissue labels.")
    return resolved.set_index("sample_id")["tissue_label"]


def align_labels_to_expression(expression: pd.DataFrame, labels: pd.Series) -> np.ndarray:
    """Align sample labels to the expression matrix row order."""
    resolved = labels.reindex(expression.index)
    if resolved.isna().any():
        missing_samples = resolved.index[resolved.isna()].tolist()
        preview = ", ".join(missing_samples[:5])
        raise ValueError(
            "Tissue label table did not cover every expression sample. Missing labels for: "
            f"{preview}"
        )
    return resolved.to_numpy(dtype=object)


def resolve_selected_gene_ids(
    expression: pd.DataFrame,
    *,
    valid_gene_mask_path: str | None = None,
    selected_gene_ids: list[str] | None = None,
) -> list[str]:
    """Resolve the exact ordered gene set used to build sample embeddings."""
    if selected_gene_ids is not None:
        resolved_gene_ids = [str(gene_id) for gene_id in selected_gene_ids]
        missing_gene_ids = [gene_id for gene_id in resolved_gene_ids if gene_id not in expression.columns]
        if missing_gene_ids:
            preview = ", ".join(missing_gene_ids[:5])
            raise ValueError(
                "The saved tissue artifact expects genes that are absent from the input "
                f"expression matrix: {preview}"
            )
    elif valid_gene_mask_path is not None:
        valid_gene_mask = load_valid_gene_mask(Path(valid_gene_mask_path))
        valid_gene_flags = resolve_valid_gene_flags(valid_gene_mask, expression.columns)
        resolved_gene_ids = expression.columns[valid_gene_flags].astype(str).tolist()
    else:
        resolved_gene_ids = expression.columns.astype(str).tolist()

    if not resolved_gene_ids:
        raise ValueError("At least one gene must be selected for tissue embedding extraction.")
    return resolved_gene_ids


def train_tissue_classifier(
    sample_embeddings: np.ndarray,
    labels: np.ndarray | list[str],
    *,
    classifier_type: str = "random_forest",
    pca_components: int | None = None,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    max_depth: int | None = None,
    random_seed: int = 0,
) -> TrainedTissueClassifier:
    """Fit an optional PCA plus classifier on sample embeddings.

    Args:
        sample_embeddings: (n_samples, n_features) embedding matrix.
        labels: Tissue labels per sample.
        classifier_type: "random_forest" (default) or "tabpfn" (few-shot friendly).
        pca_components: Optional PCA dimensionality reduction.
        n_estimators: For random_forest only.
        max_depth: For random_forest only.
        random_seed: Random seed.

    Returns:
        TrainedTissueClassifier with pipeline and metrics.
    """
    features = np.asarray(sample_embeddings, dtype=np.float32)
    resolved_labels = np.asarray(labels, dtype=object).reshape(-1)
    if features.ndim != 2:
        raise ValueError("sample_embeddings must be a 2D matrix.")
    if features.shape[0] == 0:
        raise ValueError("sample_embeddings must contain at least one sample.")
    if features.shape[0] != resolved_labels.shape[0]:
        raise ValueError("sample_embeddings and labels must contain the same number of rows.")
    if n_estimators <= 0 and classifier_type == "random_forest":
        raise ValueError("n_estimators must be positive.")

    unique_labels = np.unique(resolved_labels)
    if unique_labels.size < 2:
        raise ValueError("At least two tissue classes are required for training.")

    steps: list[tuple[str, Any]] = []
    resolved_pca_components = None if pca_components in (None, 0) else int(pca_components)
    if resolved_pca_components is not None:
        if resolved_pca_components <= 0:
            raise ValueError("pca_components must be positive when provided.")
        max_components = min(features.shape[0], features.shape[1])
        if resolved_pca_components > max_components:
            raise ValueError(
                f"pca_components={resolved_pca_components} exceeds the allowable maximum "
                f"of {max_components} for the current training matrix."
            )
        steps.append(("pca", PCA(n_components=resolved_pca_components)))

    if classifier_type == "tabpfn":
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as e:
            raise ImportError(
                "TabPFN classifier requires the tabpfn package. Install with: pip install tabpfn"
            ) from e
        clf = TabPFNClassifier()
        steps.append(("classifier", clf))
    else:
        steps.append(
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_seed,
                    n_jobs=-1,
                ),
            )
        )
    pipeline = Pipeline(steps)
    pipeline.fit(features, resolved_labels)
    train_predictions = pipeline.predict(features)
    train_accuracy = float(np.mean(train_predictions == resolved_labels))
    metrics = {
        "samples": int(features.shape[0]),
        "embedding_dim": int(features.shape[1]),
        "class_count": int(unique_labels.size),
        "classes": unique_labels.tolist(),
        "classifier_type": classifier_type,
        "pca_components": resolved_pca_components,
        "n_estimators": int(n_estimators) if classifier_type == "random_forest" else None,
        "max_depth": None if max_depth is None else int(max_depth),
        "train_accuracy": train_accuracy,
    }
    return TrainedTissueClassifier(pipeline=pipeline, metrics=metrics)


def save_tissue_artifacts(
    trained_classifier: TrainedTissueClassifier,
    output_dir: Path,
    *,
    aggregation: str,
    selected_gene_ids: list[str],
    model_contract: dict[str, str | None],
) -> tuple[Path, Path]:
    """Persist the trained sklearn pipeline and JSON summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "tissue_model.joblib"
    summary_path = output_dir / "training_summary.json"
    bundle = {
        "pipeline": trained_classifier.pipeline,
        "metrics": {
            **trained_classifier.metrics,
            "aggregation": aggregation,
        },
        "feature_spec": {
            "aggregation": aggregation,
            "selected_gene_ids": selected_gene_ids,
            "model_contract": model_contract,
        },
    }
    joblib.dump(bundle, artifact_path)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(bundle["metrics"], handle, indent=2, sort_keys=True)
        handle.write("\n")
    return artifact_path, summary_path


def load_tissue_artifacts(path: Path) -> dict[str, Any]:
    """Load the persisted tissue classifier bundle from disk."""
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle or "metrics" not in bundle:
        raise ValueError(f"Tissue artifact at {path} did not contain the expected bundle keys.")
    bundle.setdefault("feature_spec", {})
    return bundle


def predict_tissue_labels(
    sample_embeddings: np.ndarray,
    pipeline: Pipeline,
) -> pd.DataFrame:
    """Predict tissue labels and per-class probabilities for each sample embedding."""
    features = np.asarray(sample_embeddings, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("sample_embeddings must be a 2D matrix.")
    predicted_labels = pipeline.predict(features)
    probability_frame = pd.DataFrame(index=np.arange(features.shape[0]))
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(features)
        classifier = pipeline.named_steps["classifier"]
        for class_label, class_probabilities in zip(classifier.classes_, probabilities.T, strict=True):
            probability_frame[f"probability_{class_label}"] = class_probabilities
    probability_frame.insert(0, "predicted_tissue_label", predicted_labels)
    return probability_frame


def _resolve_model_contract(
    loaded_model: Any,
    model_kwargs: dict[str, Any],
) -> dict[str, str | None]:
    assets = getattr(loaded_model, "assets", None)
    if assets is not None:
        return {
            "variant": str(getattr(assets, "variant", model_kwargs.get("variant"))),
            "checkpoint_path": _optional_path_string(getattr(assets, "checkpoint_path", None)),
            "graph_path": _optional_path_string(getattr(assets, "graph_path", None)),
            "graph_weights_path": _optional_path_string(getattr(assets, "graph_weights_path", None)),
            "gene_embedding_path": _optional_path_string(getattr(assets, "gene_embedding_path", None)),
            "gene_info_path": _optional_path_string(getattr(assets, "gene_info_path", None)),
        }
    return {
        "variant": None if model_kwargs.get("variant") is None else str(model_kwargs.get("variant")),
        "checkpoint_path": _optional_path_string(model_kwargs.get("checkpoint_path")),
        "graph_path": _optional_path_string(model_kwargs.get("graph_path")),
        "graph_weights_path": _optional_path_string(model_kwargs.get("graph_weights_path")),
        "gene_embedding_path": _optional_path_string(model_kwargs.get("gene_embedding_path")),
        "gene_info_path": _optional_path_string(model_kwargs.get("gene_info_path")),
    }


def _resolve_prediction_model_kwargs(
    args: argparse.Namespace,
    stored_contract: dict[str, str | None],
) -> dict[str, Any]:
    resolved_kwargs: dict[str, Any] = {"device": getattr(args, "device", "cpu")}
    field_names = (
        "variant",
        "checkpoint_path",
        "graph_path",
        "graph_weights_path",
        "gene_embedding_path",
        "gene_info_path",
    )
    for field_name in field_names:
        explicit_value = getattr(args, field_name, None)
        stored_value = stored_contract.get(field_name)
        if explicit_value is not None and stored_value is not None and str(explicit_value) != stored_value:
            raise ValueError(
                f"Prediction {field_name}={explicit_value!r} does not match the training artifact "
                f"value {stored_value!r}."
            )
        resolved_value = explicit_value if explicit_value is not None else stored_value
        if resolved_value is not None:
            resolved_kwargs[field_name] = resolved_value
    return resolved_kwargs


def extract_tissue_embeddings(
    expression: pd.DataFrame,
    *,
    selected_gene_ids: list[str],
    aggregation: str,
    batch_size: int,
    model_kwargs: dict[str, Any],
) -> tuple[np.ndarray, dict[str, str | None]]:
    """Load BulkFormer once and derive per-sample embeddings."""
    loaded_model = load_bulkformer_model(**model_kwargs)
    gene_indices = expression.columns.get_indexer(selected_gene_ids).tolist()
    if any(gene_index < 0 for gene_index in gene_indices):
        raise ValueError("Selected tissue genes could not be aligned to the expression matrix.")
    return extract_sample_embeddings(
        loaded_model.model,
        expression,
        batch_size=batch_size,
        aggregation=aggregation,
        device=loaded_model.device,
        gene_indices=gene_indices,
    ), _resolve_model_contract(loaded_model, model_kwargs)


def write_prediction_outputs(
    predictions: pd.DataFrame,
    sample_ids: pd.Index,
    output_dir: Path,
    *,
    artifact_path: Path,
) -> tuple[Path, Path]:
    """Persist tissue predictions plus compact run metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "tissue_predictions.tsv"
    summary_path = output_dir / "prediction_summary.json"
    resolved_predictions = predictions.copy()
    resolved_predictions.insert(0, "sample_id", sample_ids.astype(str))
    resolved_predictions.to_csv(predictions_path, sep="\t", index=False)

    summary = {
        "samples": int(len(resolved_predictions)),
        "artifact_path": str(artifact_path),
        "predicted_classes": sorted(resolved_predictions["predicted_tissue_label"].unique().tolist()),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return predictions_path, summary_path


def register_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the tissue validation command group."""
    parser = subparsers.add_parser(
        "tissue",
        help="Train or run tissue validation models.",
        description=(
            "Train tissue classifiers from BulkFormer sample embeddings or predict "
            "tissue labels with serialized sklearn artifacts."
        ),
    )
    parser.add_argument(
        "mode",
        choices=SUPPORTED_MODES,
        help="Whether to fit a classifier or predict tissue labels.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a BulkFormer-aligned sample-by-gene expression table.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where model artifacts or predictions should be written.",
    )
    parser.add_argument(
        "--labels",
        help="Path to a sample_id/tissue_label table. Required for train mode.",
    )
    parser.add_argument(
        "--artifact-path",
        help="Path to a serialized tissue_model.joblib bundle. Required for predict mode.",
    )
    parser.add_argument(
        "--valid-gene-mask",
        help="Optional valid_gene_mask.tsv file to restrict embeddings to observed genes.",
    )
    parser.add_argument(
        "--aggregation",
        default=DEFAULT_AGGREGATION,
        help="Sample embedding aggregation to use. Defaults to mean.",
    )
    parser.add_argument(
        "--classifier-type",
        choices=("random_forest", "tabpfn"),
        default="random_forest",
        help="Classifier backend: random_forest (default) or tabpfn (few-shot friendly).",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        help="Optional PCA dimensionality applied before classifier training.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=DEFAULT_N_ESTIMATORS,
        help="Number of trees for the RandomForest classifier.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        help="Optional maximum tree depth for the RandomForest classifier.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed used for classifier fitting.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="BulkFormer batch size used for embedding extraction.",
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
        help="Torch device for BulkFormer embedding extraction. Defaults to cpu.",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    """Execute the tissue train or predict workflow."""
    resolved_mode = str(args.mode).strip().lower()
    if resolved_mode not in SUPPORTED_MODES:
        supported = ", ".join(SUPPORTED_MODES)
        raise ValueError(f"Unsupported tissue mode {args.mode!r}. Expected one of: {supported}.")

    expression = load_aligned_expression(Path(args.input))
    model_kwargs: dict[str, Any] = {
        "variant": getattr(args, "variant", None),
        "checkpoint_path": getattr(args, "checkpoint_path", None),
        "device": getattr(args, "device", "cpu"),
    }
    for attr_name in ("graph_path", "graph_weights_path", "gene_embedding_path", "gene_info_path"):
        attr_value = getattr(args, attr_name, None)
        if attr_value is not None:
            model_kwargs[attr_name] = attr_value

    if resolved_mode == "train":
        labels_path = getattr(args, "labels", None)
        if labels_path is None:
            raise ValueError("A tissue label table is required for train mode.")
        labels = load_tissue_labels(Path(labels_path))
        aligned_labels = align_labels_to_expression(expression, labels)
        selected_gene_ids = resolve_selected_gene_ids(
            expression,
            valid_gene_mask_path=getattr(args, "valid_gene_mask", None),
        )
        sample_embeddings, model_contract = extract_tissue_embeddings(
            expression,
            selected_gene_ids=selected_gene_ids,
            aggregation=args.aggregation,
            batch_size=args.batch_size,
            model_kwargs=model_kwargs,
        )
        trained_classifier = train_tissue_classifier(
            sample_embeddings,
            aligned_labels,
            classifier_type=getattr(args, "classifier_type", "random_forest"),
            pca_components=args.pca_components,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_seed=args.random_seed,
        )
        artifact_path, summary_path = save_tissue_artifacts(
            trained_classifier,
            Path(args.output_dir),
            aggregation=args.aggregation,
            selected_gene_ids=selected_gene_ids,
            model_contract=model_contract,
        )
        print(f"Wrote tissue model bundle to {artifact_path}")
        print(f"Wrote tissue training summary to {summary_path}")
        return 0

    artifact_path = getattr(args, "artifact_path", None)
    if artifact_path is None:
        raise ValueError("An artifact path is required for predict mode.")
    artifact_bundle = load_tissue_artifacts(Path(artifact_path))
    feature_spec = artifact_bundle.get("feature_spec", {})
    stored_selected_gene_ids = feature_spec.get("selected_gene_ids")
    if getattr(args, "valid_gene_mask", None) is not None and stored_selected_gene_ids is not None:
        requested_gene_ids = resolve_selected_gene_ids(
            expression,
            valid_gene_mask_path=getattr(args, "valid_gene_mask", None),
        )
        if requested_gene_ids != list(stored_selected_gene_ids):
            raise ValueError(
                "Prediction valid_gene_mask selection does not match the gene set saved in "
                "the training artifact."
            )
    selected_gene_ids = resolve_selected_gene_ids(
        expression,
        valid_gene_mask_path=None if stored_selected_gene_ids is not None else getattr(args, "valid_gene_mask", None),
        selected_gene_ids=None if stored_selected_gene_ids is None else list(stored_selected_gene_ids),
    )
    aggregation = str(feature_spec.get("aggregation", artifact_bundle["metrics"].get("aggregation", args.aggregation)))
    prediction_model_kwargs = _resolve_prediction_model_kwargs(
        args,
        feature_spec.get("model_contract", {}),
    )
    sample_embeddings, _ = extract_tissue_embeddings(
        expression,
        selected_gene_ids=selected_gene_ids,
        aggregation=aggregation,
        batch_size=args.batch_size,
        model_kwargs=prediction_model_kwargs,
    )
    predictions = predict_tissue_labels(sample_embeddings, artifact_bundle["pipeline"])
    predictions_path, summary_path = write_prediction_outputs(
        predictions,
        expression.index,
        Path(args.output_dir),
        artifact_path=Path(artifact_path),
    )
    print(f"Wrote tissue predictions to {predictions_path}")
    print(f"Wrote tissue prediction summary to {summary_path}")
    return 0
