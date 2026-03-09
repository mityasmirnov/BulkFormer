"""Tests for BulkFormer asset loading and embedding utilities."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from bulkformer_dx.bulkformer_model import (
    build_bulkformer_graph,
    extract_gene_embeddings,
    extract_sample_embeddings,
    load_checkpoint_state_dict,
    load_bulkformer_model,
    predict_expression,
    resolve_bulkformer_assets,
)
from model.config import get_model_params, normalize_model_variant
from utils.BulkFormer_block import BulkFormer_block


class DummyBulkFormer(torch.nn.Module):
    """Small stand-in model for embedding utility tests."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask_prob: float | None = None,
        output_expr: bool = False,
    ) -> torch.Tensor:
        x = x + self.anchor
        if output_expr:
            return x + float(mask_prob or 0.0)
        return torch.stack((x, x + 1.0, x + 2.0), dim=-1)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"placeholder")


def test_model_config_supports_variant_selection() -> None:
    assert normalize_model_variant("bulkformer_37m") == "37M"
    assert get_model_params("37m")["dim"] == 128
    assert get_model_params("147M")["p_repeat"] == 12


def test_resolve_bulkformer_assets_prefers_local_37m_checkpoint(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    data_dir = tmp_path / "data"
    for relative_path in (
        model_dir / "BulkFormer_37M.pt",
        model_dir / "BulkFormer_147M.pt",
        data_dir / "G_tcga.pt",
        data_dir / "G_tcga_weight.pt",
        data_dir / "esm2_feature_concat.pt",
        data_dir / "bulkformer_gene_info.csv",
    ):
        _touch(relative_path)

    assets = resolve_bulkformer_assets(
        model_dir=model_dir,
        graph_path=data_dir / "G_tcga.pt",
        graph_weights_path=data_dir / "G_tcga_weight.pt",
        gene_embedding_path=data_dir / "esm2_feature_concat.pt",
        gene_info_path=data_dir / "bulkformer_gene_info.csv",
    )

    assert assets.variant == "37M"
    assert assets.checkpoint_path.name == "BulkFormer_37M.pt"


def test_load_checkpoint_state_dict_cleans_wrapped_prefixes(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "state_dict": {
                "module.model.head.weight": torch.ones(2, 2),
                "module.model.head.bias": torch.zeros(2),
            }
        },
        checkpoint_path,
    )

    cleaned = load_checkpoint_state_dict(checkpoint_path)

    assert list(cleaned.keys()) == ["head.weight", "head.bias"]


def test_build_bulkformer_graph_falls_back_when_sparse_tensor_cannot_construct(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    graph_path = tmp_path / "graph.pt"
    weights_path = tmp_path / "weights.pt"
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    edge_weight = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float32)
    torch.save({"edge_index": edge_index}, graph_path)
    torch.save(edge_weight, weights_path)

    import torch_geometric.typing as pyg_typing

    class BrokenSparseTensor:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError("torch-sparse unavailable at runtime")

    monkeypatch.setattr(pyg_typing, "SparseTensor", BrokenSparseTensor)

    resolved_graph = build_bulkformer_graph(graph_path, weights_path, device="cpu")

    assert isinstance(resolved_graph, tuple)
    resolved_edge_index, resolved_edge_weight = resolved_graph
    assert torch.equal(resolved_edge_index, edge_index)
    assert torch.equal(resolved_edge_weight, edge_weight)


def test_bulkformer_block_accepts_edge_index_and_edge_weight_tuple() -> None:
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    edge_weight = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float32)
    block = BulkFormer_block(dim=16, gene_length=3, p_repeat=1)
    expression = torch.randn(2, 3, 16)

    resolved = block(expression, (edge_index, edge_weight))

    assert resolved.shape == expression.shape


def test_embedding_utilities_batch_and_aggregate_outputs() -> None:
    model = DummyBulkFormer()
    expression = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)

    gene_embeddings = extract_gene_embeddings(
        model,
        expression,
        batch_size=1,
        gene_indices=[1],
    )
    sample_embeddings = extract_sample_embeddings(
        model,
        expression,
        batch_size=1,
        aggregation="all",
        gene_indices=[0, 1],
    )
    predicted_expression = predict_expression(
        model,
        expression,
        batch_size=1,
        mask_prob=0.25,
    )

    assert gene_embeddings.shape == (2, 1, 3)
    assert np.allclose(gene_embeddings[0, 0], np.array([2.0, 3.0, 4.0]))
    assert np.allclose(
        sample_embeddings,
        np.array(
            [
                [5.0, 8.0, 11.0],
                [13.0, 16.0, 19.0],
            ]
        ),
    )
    assert np.allclose(predicted_expression, expression + 0.25)


@pytest.mark.skipif(
    os.environ.get("BULKFORMER_DX_RUN_MODEL_SMOKE") != "1",
    reason="Set BULKFORMER_DX_RUN_MODEL_SMOKE=1 to enable the local checkpoint smoke test.",
)
def test_load_bulkformer_model_smoke_when_local_assets_are_available() -> None:
    pytest.importorskip("torch_geometric")
    pytest.importorskip("performer_pytorch")

    variant = os.environ.get("BULKFORMER_DX_MODEL_VARIANT", "37M")
    try:
        loaded = load_bulkformer_model(variant=variant, device="cpu")
    except FileNotFoundError as exc:
        pytest.skip(str(exc))

    zeros = np.zeros((1, loaded.config["gene_length"]), dtype=np.float32)
    sample_embeddings = extract_sample_embeddings(
        loaded.model,
        zeros,
        batch_size=1,
        aggregation="mean",
        device=loaded.device,
    )

    assert sample_embeddings.shape[0] == 1
    assert sample_embeddings.shape[1] == loaded.config["dim"] + 3
