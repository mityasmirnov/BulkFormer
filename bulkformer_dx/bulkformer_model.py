"""BulkFormer asset discovery, model loading, and embedding utilities."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import torch

from model.config import get_model_params, normalize_model_variant

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = REPO_ROOT / "model"
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_GENE_INFO_PATH = DEFAULT_DATA_DIR / "bulkformer_gene_info.csv"
DEFAULT_GRAPH_PATH = DEFAULT_DATA_DIR / "G_tcga.pt"
DEFAULT_GRAPH_WEIGHTS_PATH = DEFAULT_DATA_DIR / "G_tcga_weight.pt"
DEFAULT_GENE_EMBEDDING_PATH = DEFAULT_DATA_DIR / "esm2_feature_concat.pt"
DEFAULT_AUTO_VARIANT_ORDER = ("37M", "50M", "93M", "127M", "147M")
AGGREGATION_TYPES = ("max", "mean", "median", "all")
STATE_DICT_KEYS = ("state_dict", "model_state_dict", "model")
PREFIXES_TO_STRIP = ("module.", "model.")


@dataclass(slots=True, frozen=True)
class BulkFormerAssets:
    """Filesystem paths required to instantiate a BulkFormer checkpoint."""

    variant: str
    checkpoint_path: Path
    graph_path: Path
    graph_weights_path: Path
    gene_embedding_path: Path
    gene_info_path: Path


@dataclass(slots=True)
class LoadedBulkFormer:
    """Loaded BulkFormer model plus its resolved assets and config."""

    model: torch.nn.Module
    assets: BulkFormerAssets
    config: dict[str, int]
    device: torch.device


def _checkpoint_candidates_for_variant(variant: str) -> tuple[str, ...]:
    return (
        f"BulkFormer_{variant}.pt",
        f"bulkformer_{variant}.pt",
        f"BulkFormer-{variant}.pt",
        f"bulkformer-{variant}.pt",
    )


def _missing_asset_message(path: Path, *, asset_kind: str) -> str:
    if asset_kind == "checkpoint":
        return (
            f"Missing BulkFormer checkpoint at {path}. Download a pretrained model "
            f"using the links documented in `model/README.md`."
        )
    return (
        f"Missing BulkFormer {asset_kind} asset at {path}. Download the required "
        f"data assets from the Zenodo record documented in `data/README.md`."
    )


def _require_existing_path(path: Path, *, asset_kind: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(_missing_asset_message(resolved, asset_kind=asset_kind))
    return resolved


def infer_model_variant_from_checkpoint(checkpoint_path: str | Path) -> str | None:
    """Infer the BulkFormer variant label from a checkpoint filename."""
    match = re.search(r"(\d+)m", Path(checkpoint_path).name.lower())
    if not match:
        return None
    variant = f"{match.group(1)}M"
    try:
        return normalize_model_variant(variant)
    except ValueError:
        return None


def discover_checkpoint_path(
    *,
    variant: str | None = None,
    checkpoint_path: str | Path | None = None,
    model_dir: Path = DEFAULT_MODEL_DIR,
) -> tuple[str, Path]:
    """Resolve a checkpoint path, preferring the local 37M asset for auto mode."""
    if checkpoint_path is not None:
        resolved_checkpoint = _require_existing_path(Path(checkpoint_path), asset_kind="checkpoint")
        resolved_variant = normalize_model_variant(
            variant or infer_model_variant_from_checkpoint(resolved_checkpoint) or "147M"
        )
        return resolved_variant, resolved_checkpoint

    resolved_model_dir = model_dir.expanduser().resolve()
    requested_variants = (
        (normalize_model_variant(variant),)
        if variant is not None
        else DEFAULT_AUTO_VARIANT_ORDER
    )
    for requested_variant in requested_variants:
        for candidate_name in _checkpoint_candidates_for_variant(requested_variant):
            candidate_path = resolved_model_dir / candidate_name
            if candidate_path.exists():
                return requested_variant, candidate_path.resolve()

    if variant is not None:
        expected = ", ".join(_checkpoint_candidates_for_variant(normalize_model_variant(variant)))
        raise FileNotFoundError(
            f"No checkpoint for BulkFormer {normalize_model_variant(variant)} was found in "
            f"{resolved_model_dir}. Looked for: {expected}. See `model/README.md` for "
            "download instructions."
        )

    raise FileNotFoundError(
        f"No BulkFormer checkpoint was found in {resolved_model_dir}. "
        "Place a checkpoint such as `BulkFormer_37M.pt` there or pass `checkpoint_path` "
        "explicitly. See `model/README.md` for download instructions."
    )


def resolve_bulkformer_assets(
    *,
    variant: str | None = None,
    checkpoint_path: str | Path | None = None,
    model_dir: Path = DEFAULT_MODEL_DIR,
    graph_path: str | Path = DEFAULT_GRAPH_PATH,
    graph_weights_path: str | Path = DEFAULT_GRAPH_WEIGHTS_PATH,
    gene_embedding_path: str | Path = DEFAULT_GENE_EMBEDDING_PATH,
    gene_info_path: str | Path = DEFAULT_GENE_INFO_PATH,
) -> BulkFormerAssets:
    """Resolve the checkpoint and supporting BulkFormer assets from disk."""
    resolved_variant, resolved_checkpoint = discover_checkpoint_path(
        variant=variant,
        checkpoint_path=checkpoint_path,
        model_dir=model_dir,
    )
    return BulkFormerAssets(
        variant=resolved_variant,
        checkpoint_path=resolved_checkpoint,
        graph_path=_require_existing_path(Path(graph_path), asset_kind="graph"),
        graph_weights_path=_require_existing_path(
            Path(graph_weights_path),
            asset_kind="graph weight",
        ),
        gene_embedding_path=_require_existing_path(
            Path(gene_embedding_path),
            asset_kind="gene embedding",
        ),
        gene_info_path=_require_existing_path(Path(gene_info_path), asset_kind="gene info"),
    )


def _unwrap_state_dict(checkpoint_object: Any) -> Mapping[str, Any]:
    if not isinstance(checkpoint_object, Mapping):
        raise TypeError(
            "Expected a checkpoint mapping or a mapping with a nested state dict."
        )

    for key in STATE_DICT_KEYS:
        candidate = checkpoint_object.get(key)
        if isinstance(candidate, Mapping):
            checkpoint_object = candidate
            break

    if not isinstance(checkpoint_object, Mapping):
        raise TypeError("BulkFormer checkpoint did not contain a readable state dict.")

    tensor_keys = {
        str(key): value for key, value in checkpoint_object.items() if torch.is_tensor(value)
    }
    if tensor_keys:
        return tensor_keys
    return {str(key): value for key, value in checkpoint_object.items()}


def cleanup_checkpoint_state_dict(state_dict: Mapping[str, Any]) -> OrderedDict[str, Any]:
    """Remove wrapper prefixes like `module.` from checkpoint keys."""
    cleaned_state_dict: OrderedDict[str, Any] = OrderedDict()
    for key, value in state_dict.items():
        new_key = str(key)
        prefix_removed = True
        while prefix_removed:
            prefix_removed = False
            for prefix in PREFIXES_TO_STRIP:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    prefix_removed = True
        cleaned_state_dict[new_key] = value
    return cleaned_state_dict


def load_checkpoint_state_dict(checkpoint_path: str | Path) -> OrderedDict[str, Any]:
    """Load a checkpoint and normalize it into a clean state dict."""
    checkpoint_object = torch.load(
        Path(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    return cleanup_checkpoint_state_dict(_unwrap_state_dict(checkpoint_object))


def _extract_graph_rows_and_cols(graph_object: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(graph_object, Mapping):
        if "edge_index" in graph_object:
            edge_index = graph_object["edge_index"]
            if torch.is_tensor(edge_index) and edge_index.ndim == 2 and edge_index.shape[0] == 2:
                return edge_index[1], edge_index[0]
        if "row" in graph_object and "col" in graph_object:
            row = graph_object["row"]
            col = graph_object["col"]
            if torch.is_tensor(row) and torch.is_tensor(col):
                return row, col

    if torch.is_tensor(graph_object) and graph_object.ndim == 2 and graph_object.shape[0] == 2:
        return graph_object[1], graph_object[0]

    if isinstance(graph_object, Sequence) and len(graph_object) >= 2:
        row_candidate = graph_object[1]
        col_candidate = graph_object[0]
        if torch.is_tensor(row_candidate) and torch.is_tensor(col_candidate):
            return row_candidate, col_candidate

    raise ValueError(
        "Unsupported BulkFormer graph format. Expected an edge-index tensor or a "
        "tuple/list containing source and destination tensors."
    )


def build_bulkformer_graph(
    graph_path: str | Path,
    graph_weights_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> Any:
    """Build the sparse graph object expected by `utils.BulkFormer.BulkFormer`.

    Prefers SparseTensor (PyG) when torch-sparse is available—recommended for GPU.
    Falls back to (edge_index, edge_weight) when SparseTensor import or construction
    fails (e.g. torch-sparse unavailable on CPU). BulkFormer_block accepts both.
    """
    graph_object = torch.load(Path(graph_path), map_location="cpu", weights_only=False)
    weights = torch.load(Path(graph_weights_path), map_location="cpu", weights_only=False)
    row, col = _extract_graph_rows_and_cols(graph_object)
    if not torch.is_tensor(weights):
        weights = torch.as_tensor(weights)
    resolved_device = torch.device(device)

    # Prefer SparseTensor for GPU (torch-sparse); fallback for CPU / broken torch-sparse
    try:
        from torch_geometric.typing import SparseTensor

        graph = SparseTensor(row=row, col=col, value=weights).t().to(resolved_device)
        return graph
    except Exception:
        # torch-sparse unavailable or SparseTensor construction failed (e.g. on CPU)
        edge_index = torch.stack([col, row], dim=0).long().to(resolved_device)
        edge_weight = weights.to(resolved_device)
        return (edge_index, edge_weight)


def _load_gene_embeddings(gene_embedding_path: str | Path) -> torch.Tensor:
    gene_embeddings = torch.load(
        Path(gene_embedding_path),
        map_location="cpu",
        weights_only=False,
    )
    if not torch.is_tensor(gene_embeddings):
        gene_embeddings = torch.as_tensor(gene_embeddings)
    return gene_embeddings


def load_bulkformer_model(
    *,
    variant: str | None = None,
    checkpoint_path: str | Path | None = None,
    model_dir: Path = DEFAULT_MODEL_DIR,
    graph_path: str | Path = DEFAULT_GRAPH_PATH,
    graph_weights_path: str | Path = DEFAULT_GRAPH_WEIGHTS_PATH,
    gene_embedding_path: str | Path = DEFAULT_GENE_EMBEDDING_PATH,
    gene_info_path: str | Path = DEFAULT_GENE_INFO_PATH,
    device: str | torch.device = "cpu",
    strict: bool = True,
    eval_mode: bool = True,
) -> LoadedBulkFormer:
    """Resolve assets, instantiate the model, and load the checkpoint weights."""
    from utils.BulkFormer import BulkFormer

    assets = resolve_bulkformer_assets(
        variant=variant,
        checkpoint_path=checkpoint_path,
        model_dir=model_dir,
        graph_path=graph_path,
        graph_weights_path=graph_weights_path,
        gene_embedding_path=gene_embedding_path,
        gene_info_path=gene_info_path,
    )
    resolved_device = torch.device(device)
    config = get_model_params(assets.variant)
    gene_embeddings = _load_gene_embeddings(assets.gene_embedding_path)
    if gene_embeddings.shape[0] != config["gene_length"]:
        raise ValueError(
            f"Gene embedding length {gene_embeddings.shape[0]} does not match the "
            f"{assets.variant} config gene_length {config['gene_length']}."
        )

    graph = build_bulkformer_graph(
        assets.graph_path,
        assets.graph_weights_path,
        device=resolved_device,
    )
    model = BulkFormer(
        graph=graph,
        gene_emb=gene_embeddings,
        **config,
    ).to(resolved_device)
    state_dict = load_checkpoint_state_dict(assets.checkpoint_path)
    model.load_state_dict(state_dict, strict=strict)
    if eval_mode:
        model.eval()

    return LoadedBulkFormer(
        model=model,
        assets=assets,
        config=config,
        device=resolved_device,
    )


def expression_to_tensor(
    expression: pd.DataFrame | np.ndarray | torch.Tensor,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert a 2D expression matrix into a float tensor on the target device."""
    if isinstance(expression, pd.DataFrame):
        tensor = torch.as_tensor(expression.to_numpy(dtype=np.float32, copy=True))
    elif torch.is_tensor(expression):
        tensor = expression.detach()
    else:
        tensor = torch.as_tensor(np.asarray(expression, dtype=np.float32))

    if tensor.ndim != 2:
        raise ValueError(
            f"Expected a 2D expression matrix of shape [samples, genes], got {tuple(tensor.shape)}."
        )
    return tensor.to(device=device, dtype=dtype)


def aggregate_gene_embeddings(
    gene_embeddings: torch.Tensor,
    *,
    aggregation: str = "mean",
) -> torch.Tensor:
    """Aggregate per-gene embeddings into a sample-level embedding."""
    if aggregation not in AGGREGATION_TYPES:
        raise ValueError(
            f"Unsupported aggregation {aggregation!r}. "
            f"Expected one of: {', '.join(AGGREGATION_TYPES)}."
        )

    if aggregation == "max":
        return gene_embeddings.max(dim=1).values
    if aggregation == "mean":
        return gene_embeddings.mean(dim=1)
    if aggregation == "median":
        return torch.quantile(gene_embeddings, 0.5, dim=1)

    return (
        gene_embeddings.max(dim=1).values
        + gene_embeddings.mean(dim=1)
        + torch.quantile(gene_embeddings, 0.5, dim=1)
    )


def _run_batches(
    model: torch.nn.Module,
    expression: pd.DataFrame | np.ndarray | torch.Tensor,
    *,
    batch_size: int = 16,
    mask_prob: float = 0.0,
    output_expr: bool = False,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")

    target_device = device or next(model.parameters()).device
    tensor = expression_to_tensor(expression, device=target_device)
    outputs: list[torch.Tensor] = []

    with torch.inference_mode():
        for start_idx in range(0, tensor.shape[0], batch_size):
            batch = tensor[start_idx : start_idx + batch_size]
            batch_output = model(batch, mask_prob=mask_prob, output_expr=output_expr)
            outputs.append(batch_output.detach().cpu())

    return torch.cat(outputs, dim=0)


def extract_gene_embeddings(
    model: torch.nn.Module,
    expression: pd.DataFrame | np.ndarray | torch.Tensor,
    *,
    batch_size: int = 16,
    mask_prob: float = 0.0,
    device: str | torch.device | None = None,
    gene_indices: Iterable[int] | None = None,
) -> np.ndarray:
    """Run BulkFormer and return per-gene embeddings as a NumPy array."""
    gene_embeddings = _run_batches(
        model,
        expression,
        batch_size=batch_size,
        mask_prob=mask_prob,
        output_expr=False,
        device=device,
    )
    if gene_indices is not None:
        gene_embeddings = gene_embeddings[:, list(gene_indices), :]
    return gene_embeddings.numpy()


def extract_sample_embeddings(
    model: torch.nn.Module,
    expression: pd.DataFrame | np.ndarray | torch.Tensor,
    *,
    batch_size: int = 16,
    mask_prob: float = 0.0,
    aggregation: str = "mean",
    device: str | torch.device | None = None,
    gene_indices: Iterable[int] | None = None,
) -> np.ndarray:
    """Run BulkFormer and aggregate per-gene embeddings into sample-level vectors."""
    gene_embeddings = _run_batches(
        model,
        expression,
        batch_size=batch_size,
        mask_prob=mask_prob,
        output_expr=False,
        device=device,
    )
    if gene_indices is not None:
        gene_embeddings = gene_embeddings[:, list(gene_indices), :]
    return aggregate_gene_embeddings(gene_embeddings, aggregation=aggregation).numpy()


def predict_expression(
    model: torch.nn.Module,
    expression: pd.DataFrame | np.ndarray | torch.Tensor,
    *,
    batch_size: int = 16,
    mask_prob: float = 0.0,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Run BulkFormer in expression-prediction mode."""
    predicted_expression = _run_batches(
        model,
        expression,
        batch_size=batch_size,
        mask_prob=mask_prob,
        output_expr=True,
        device=device,
    )
    return predicted_expression.numpy()
