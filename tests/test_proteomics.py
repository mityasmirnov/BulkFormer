"""Tests for frozen-backbone proteomics workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from bulkformer_dx import proteomics


def test_transform_proteomics_targets_supports_log2_and_scaling() -> None:
    table = pd.DataFrame(
        {
            "P1": [4.0, 16.0],
            "P2": [8.0, np.nan],
        },
        index=["sample_a", "sample_b"],
    )

    transformed, stats = proteomics.transform_proteomics_targets(
        table,
        log2_transform=True,
        center_scale=True,
    )
    restored = proteomics.invert_transformed_targets(
        transformed.to_numpy(dtype=np.float32),
        transform_stats=stats,
    )

    assert np.allclose(restored[:, 0], [2.0, 4.0])
    assert np.isclose(restored[0, 1], 3.0)
    assert np.isnan(restored[1, 1])


def test_masked_mse_loss_ignores_missing_targets() -> None:
    predictions = torch.tensor([[1.0, 5.0]], dtype=torch.float32)
    targets = torch.tensor([[3.0, 0.0]], dtype=torch.float32)
    target_mask = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    loss = proteomics.masked_mse_loss(predictions, targets, target_mask)

    assert torch.isclose(loss, torch.tensor(4.0))


def test_run_train_and_predict_write_expected_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expression = pd.DataFrame(
        {
            "sample_id": ["sample_a", "sample_b", "sample_c", "sample_d"],
            "ENSG1": [0.0, 0.1, 2.0, 2.1],
            "ENSG2": [0.0, 0.2, 2.0, 2.2],
            "ENSG3": [-10.0, -10.0, -10.0, -10.0],
        }
    )
    expression_path = tmp_path / "aligned.tsv"
    expression.to_csv(expression_path, sep="\t", index=False)

    valid_gene_mask_path = tmp_path / "valid_gene_mask.tsv"
    pd.DataFrame(
        {
            "ensg_id": ["ENSG1", "ENSG2", "ENSG3"],
            "is_valid": [1, 1, 0],
        }
    ).to_csv(valid_gene_mask_path, sep="\t", index=False)

    proteomics_table = pd.DataFrame(
        {
            "sample_id": ["sample_a", "sample_b", "sample_c", "sample_d"],
            "P1": [0.0, 0.1, 2.0, 2.1],
            "P2": [0.0, np.nan, 2.0, 2.2],
        }
    )
    proteomics_path = tmp_path / "proteomics.tsv"
    proteomics_table.to_csv(proteomics_path, sep="\t", index=False)

    monkeypatch.setattr(
        proteomics,
        "load_bulkformer_model",
        lambda **kwargs: SimpleNamespace(
            model=object(),
            device="cpu",
            assets=SimpleNamespace(
                variant=kwargs.get("variant", "37M"),
                checkpoint_path=Path("/tmp/BulkFormer_37M.pt"),
                graph_path=Path("/tmp/G_tcga.pt"),
                graph_weights_path=Path("/tmp/G_tcga_weight.pt"),
                gene_embedding_path=Path("/tmp/esm2_feature_concat.pt"),
                gene_info_path=Path("/tmp/bulkformer_gene_info.csv"),
            ),
        ),
    )
    monkeypatch.setattr(
        proteomics,
        "extract_sample_embeddings",
        lambda _model, expression_matrix, **_kwargs: np.asarray(
            expression_matrix.iloc[:, :2].to_numpy(dtype=np.float32, copy=True)
            if isinstance(expression_matrix, pd.DataFrame)
            else np.asarray(expression_matrix, dtype=np.float32)[:, :2]
        ),
    )

    train_output_dir = tmp_path / "proteomics_train"
    train_exit_code = proteomics.run(
        argparse.Namespace(
            mode="train",
            input=str(expression_path),
            proteomics=str(proteomics_path),
            output_dir=str(train_output_dir),
            artifact_path=None,
            valid_gene_mask=str(valid_gene_mask_path),
            aggregation="mean",
            head_type="linear",
            hidden_dim=16,
            epochs=50,
            learning_rate=0.1,
            weight_decay=0.0,
            batch_size=2,
            val_fraction=0.25,
            patience=5,
            random_seed=0,
            log2_transform=False,
            already_log2=True,
            center_scale=False,
            alpha=0.05,
            variant="37M",
            checkpoint_path=None,
            graph_path=None,
            graph_weights_path=None,
            gene_embedding_path=None,
            gene_info_path=None,
            device="cpu",
        )
    )

    assert train_exit_code == 0
    artifact_path = train_output_dir / "protein_head.pt"
    predictions_path = train_output_dir / "predicted_proteomics.tsv"
    residuals_path = train_output_dir / "residuals.tsv"
    rankings_path = train_output_dir / "ranked_proteins" / "sample_a.tsv"
    summary_path = train_output_dir / "prediction_summary.json"
    assert artifact_path.exists()
    assert predictions_path.exists()
    assert residuals_path.exists()
    assert rankings_path.exists()
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["mode"] == "train"
    assert summary["proteins"] == 2
    assert summary["selected_genes"] == 2

    predict_expression = pd.DataFrame(
        {
            "sample_id": ["sample_x", "sample_y"],
            "ENSG1": [0.05, 2.05],
            "ENSG2": [0.05, 2.05],
            "ENSG3": [-10.0, -10.0],
        }
    )
    predict_expression_path = tmp_path / "aligned_predict.tsv"
    predict_expression.to_csv(predict_expression_path, sep="\t", index=False)

    predict_proteomics = pd.DataFrame(
        {
            "sample_id": ["sample_x", "sample_y"],
            "P1": [0.05, 2.05],
            "P2": [0.0, 2.0],
        }
    )
    predict_proteomics_path = tmp_path / "proteomics_predict.tsv"
    predict_proteomics.to_csv(predict_proteomics_path, sep="\t", index=False)

    predict_output_dir = tmp_path / "proteomics_predict"
    predict_exit_code = proteomics.run(
        argparse.Namespace(
            mode="predict",
            input=str(predict_expression_path),
            proteomics=str(predict_proteomics_path),
            output_dir=str(predict_output_dir),
            artifact_path=str(artifact_path),
            valid_gene_mask=str(valid_gene_mask_path),
            aggregation="mean",
            head_type="linear",
            hidden_dim=16,
            epochs=10,
            learning_rate=0.1,
            weight_decay=0.0,
            batch_size=2,
            val_fraction=0.25,
            patience=5,
            random_seed=0,
            log2_transform=False,
            already_log2=True,
            center_scale=False,
            alpha=0.05,
            variant="37M",
            checkpoint_path=None,
            graph_path=None,
            graph_weights_path=None,
            gene_embedding_path=None,
            gene_info_path=None,
            device="cpu",
        )
    )

    assert predict_exit_code == 0
    predictions = pd.read_csv(predict_output_dir / "predicted_proteomics.tsv", sep="\t")
    ranking = pd.read_csv(predict_output_dir / "ranked_proteins" / "sample_x.tsv", sep="\t")
    assert predictions["sample_id"].tolist() == ["sample_x", "sample_y"]
    assert {"P1", "P2"} <= set(predictions.columns)
    assert {"protein_id", "predicted_log2_intensity", "p_value", "padj", "call"} <= set(
        ranking.columns
    )


def test_predict_rejects_conflicting_checkpoint_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expression = pd.DataFrame(
        {
            "sample_id": ["sample_a", "sample_b", "sample_c"],
            "ENSG1": [0.0, 1.0, 2.0],
            "ENSG2": [0.0, 1.0, 2.0],
        }
    )
    expression_path = tmp_path / "aligned.tsv"
    expression.to_csv(expression_path, sep="\t", index=False)

    proteomics_table = pd.DataFrame(
        {
            "sample_id": ["sample_a", "sample_b", "sample_c"],
            "P1": [0.0, 1.0, 2.0],
        }
    )
    proteomics_path = tmp_path / "proteomics.tsv"
    proteomics_table.to_csv(proteomics_path, sep="\t", index=False)

    monkeypatch.setattr(
        proteomics,
        "load_bulkformer_model",
        lambda **kwargs: SimpleNamespace(
            model=object(),
            device="cpu",
            assets=SimpleNamespace(
                variant=kwargs.get("variant", "37M"),
                checkpoint_path=Path("/tmp/BulkFormer_37M.pt"),
                graph_path=Path("/tmp/G_tcga.pt"),
                graph_weights_path=Path("/tmp/G_tcga_weight.pt"),
                gene_embedding_path=Path("/tmp/esm2_feature_concat.pt"),
                gene_info_path=Path("/tmp/bulkformer_gene_info.csv"),
            ),
        ),
    )
    monkeypatch.setattr(
        proteomics,
        "extract_sample_embeddings",
        lambda _model, expression_matrix, **_kwargs: np.asarray(
            expression_matrix.to_numpy(dtype=np.float32, copy=True)[:, :2]
        ),
    )

    train_output_dir = tmp_path / "train"
    proteomics.run(
        argparse.Namespace(
            mode="train",
            input=str(expression_path),
            proteomics=str(proteomics_path),
            output_dir=str(train_output_dir),
            artifact_path=None,
            valid_gene_mask=None,
            aggregation="mean",
            head_type="linear",
            hidden_dim=16,
            epochs=10,
            learning_rate=0.1,
            weight_decay=0.0,
            batch_size=2,
            val_fraction=0.0,
            patience=2,
            random_seed=0,
            log2_transform=False,
            already_log2=True,
            center_scale=False,
            alpha=0.05,
            variant="37M",
            checkpoint_path=None,
            graph_path=None,
            graph_weights_path=None,
            gene_embedding_path=None,
            gene_info_path=None,
            device="cpu",
        )
    )

    with pytest.raises(ValueError, match="variant"):
        proteomics.run(
            argparse.Namespace(
                mode="predict",
                input=str(expression_path),
                proteomics=str(proteomics_path),
                output_dir=str(tmp_path / "predict"),
                artifact_path=str(train_output_dir / "protein_head.pt"),
                valid_gene_mask=None,
                aggregation="mean",
                head_type="linear",
                hidden_dim=16,
                epochs=10,
                learning_rate=0.1,
                weight_decay=0.0,
                batch_size=2,
                val_fraction=0.0,
                patience=2,
                random_seed=0,
                log2_transform=False,
                already_log2=True,
                center_scale=False,
                alpha=0.05,
                variant="147M",
                checkpoint_path=None,
                graph_path=None,
                graph_weights_path=None,
                gene_embedding_path=None,
                gene_info_path=None,
                device="cpu",
            )
        )
