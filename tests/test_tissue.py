"""Tests for BulkFormer tissue training and prediction workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bulkformer_dx import tissue


def test_train_tissue_classifier_supports_optional_pca() -> None:
    sample_embeddings = np.array(
        [
            [0.0, 0.0, 0.1],
            [0.1, 0.0, 0.2],
            [2.0, 2.0, 2.1],
            [2.1, 2.0, 2.2],
        ],
        dtype=np.float32,
    )
    labels = np.array(["brain", "brain", "liver", "liver"], dtype=object)

    result = tissue.train_tissue_classifier(
        sample_embeddings,
        labels,
        pca_components=2,
        n_estimators=64,
        random_seed=0,
    )
    predictions = tissue.predict_tissue_labels(sample_embeddings, result.pipeline)

    assert predictions["predicted_tissue_label"].tolist() == labels.tolist()
    assert "pca" in result.pipeline.named_steps
    assert result.metrics["samples"] == 4
    assert result.metrics["class_count"] == 2
    assert result.metrics["pca_components"] == 2


def test_load_tissue_labels_requires_unique_sample_ids(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.tsv"
    pd.DataFrame(
        {
            "sample_id": ["sample_a", "sample_a"],
            "tissue_label": ["brain", "liver"],
        }
    ).to_csv(labels_path, sep="\t", index=False)

    with pytest.raises(ValueError, match="duplicate sample IDs"):
        tissue.load_tissue_labels(labels_path)


def test_resolve_selected_gene_ids_rejects_zero_valid_genes(tmp_path: Path) -> None:
    expression = pd.DataFrame(
        [[1.0, 2.0]],
        index=["sample_a"],
        columns=["ENSG1", "ENSG2"],
    )
    valid_gene_mask_path = tmp_path / "valid_gene_mask.tsv"
    pd.DataFrame(
        {
            "ensg_id": ["ENSG1", "ENSG2"],
            "is_valid": [0, 0],
        }
    ).to_csv(valid_gene_mask_path, sep="\t", index=False)

    with pytest.raises(ValueError, match="At least one gene"):
        tissue.resolve_selected_gene_ids(
            expression,
            valid_gene_mask_path=str(valid_gene_mask_path),
        )


def test_run_train_and_predict_write_expected_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expression = pd.DataFrame(
        {
            "sample_id": ["sample_a", "sample_b", "sample_c", "sample_d"],
            "ENSG1": [0.0, 0.1, 2.0, 2.1],
            "ENSG2": [0.0, 0.0, 2.0, 2.0],
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

    labels_path = tmp_path / "labels.tsv"
    pd.DataFrame(
        {
            "sample_id": ["sample_a", "sample_b", "sample_c", "sample_d"],
            "tissue_label": ["brain", "brain", "liver", "liver"],
        }
    ).to_csv(labels_path, sep="\t", index=False)

    monkeypatch.setattr(
        tissue,
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
        tissue,
        "extract_sample_embeddings",
        lambda _model, expression_matrix, **_kwargs: np.asarray(
            expression_matrix.iloc[:, :2].to_numpy(dtype=np.float32, copy=True)
            if isinstance(expression_matrix, pd.DataFrame)
            else np.asarray(expression_matrix, dtype=np.float32)[:, :2]
        ),
    )

    train_output_dir = tmp_path / "tissue_train"
    train_exit_code = tissue.run(
        argparse.Namespace(
            mode="train",
            input=str(expression_path),
            labels=str(labels_path),
            output_dir=str(train_output_dir),
            artifact_path=None,
            valid_gene_mask=str(valid_gene_mask_path),
            variant="37M",
            checkpoint_path=None,
            graph_path=None,
            graph_weights_path=None,
            gene_embedding_path=None,
            gene_info_path=None,
            device="cpu",
            batch_size=8,
            aggregation="mean",
            pca_components=2,
            n_estimators=64,
            max_depth=None,
            random_seed=0,
        )
    )

    assert train_exit_code == 0
    artifact_path = train_output_dir / "tissue_model.joblib"
    summary_path = train_output_dir / "training_summary.json"
    assert artifact_path.exists()
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["samples"] == 4
    assert summary["class_count"] == 2
    assert summary["aggregation"] == "mean"

    predict_expression = pd.DataFrame(
        {
            "sample_id": ["sample_x", "sample_y"],
            "ENSG1": [0.05, 2.05],
            "ENSG2": [0.0, 2.0],
            "ENSG3": [-10.0, -10.0],
        }
    )
    predict_expression_path = tmp_path / "aligned_predict.tsv"
    predict_expression.to_csv(predict_expression_path, sep="\t", index=False)

    predict_output_dir = tmp_path / "tissue_predict"
    predict_exit_code = tissue.run(
        argparse.Namespace(
            mode="predict",
            input=str(predict_expression_path),
            labels=None,
            output_dir=str(predict_output_dir),
            artifact_path=str(artifact_path),
            valid_gene_mask=str(valid_gene_mask_path),
            variant="37M",
            checkpoint_path=None,
            graph_path=None,
            graph_weights_path=None,
            gene_embedding_path=None,
            gene_info_path=None,
            device="cpu",
            batch_size=8,
            aggregation="mean",
            pca_components=None,
            n_estimators=64,
            max_depth=None,
            random_seed=0,
        )
    )

    assert predict_exit_code == 0
    predictions_path = predict_output_dir / "tissue_predictions.tsv"
    run_metadata_path = predict_output_dir / "prediction_summary.json"
    assert predictions_path.exists()
    assert run_metadata_path.exists()

    predictions = pd.read_csv(predictions_path, sep="\t")
    assert predictions["sample_id"].tolist() == ["sample_x", "sample_y"]
    assert predictions["predicted_tissue_label"].tolist() == ["brain", "liver"]
    assert {"probability_brain", "probability_liver"} <= set(predictions.columns)


def test_predict_rejects_conflicting_model_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expression = pd.DataFrame(
        {
            "sample_id": ["sample_a", "sample_b"],
            "ENSG1": [0.0, 2.0],
            "ENSG2": [0.0, 2.0],
        }
    )
    expression_path = tmp_path / "aligned.tsv"
    expression.to_csv(expression_path, sep="\t", index=False)

    labels_path = tmp_path / "labels.tsv"
    pd.DataFrame(
        {
            "sample_id": ["sample_a", "sample_b"],
            "tissue_label": ["brain", "liver"],
        }
    ).to_csv(labels_path, sep="\t", index=False)

    monkeypatch.setattr(
        tissue,
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
        tissue,
        "extract_sample_embeddings",
        lambda _model, expression_matrix, **_kwargs: np.asarray(
            expression_matrix.to_numpy(dtype=np.float32, copy=True)[:, :2]
        ),
    )

    train_output_dir = tmp_path / "train"
    tissue.run(
        argparse.Namespace(
            mode="train",
            input=str(expression_path),
            labels=str(labels_path),
            output_dir=str(train_output_dir),
            artifact_path=None,
            valid_gene_mask=None,
            variant="37M",
            checkpoint_path=None,
            graph_path=None,
            graph_weights_path=None,
            gene_embedding_path=None,
            gene_info_path=None,
            device="cpu",
            batch_size=8,
            aggregation="mean",
            pca_components=None,
            n_estimators=32,
            max_depth=None,
            random_seed=0,
        )
    )

    with pytest.raises(ValueError, match="variant"):
        tissue.run(
            argparse.Namespace(
                mode="predict",
                input=str(expression_path),
                labels=None,
                output_dir=str(tmp_path / "predict"),
                artifact_path=str(train_output_dir / "tissue_model.joblib"),
                valid_gene_mask=None,
                variant="147M",
                checkpoint_path=None,
                graph_path=None,
                graph_weights_path=None,
                gene_embedding_path=None,
                gene_info_path=None,
                device="cpu",
                batch_size=8,
                aggregation="mean",
                pca_components=None,
                n_estimators=32,
                max_depth=None,
                random_seed=0,
            )
        )
