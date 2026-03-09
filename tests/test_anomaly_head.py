"""Tests for the anomaly uncertainty head workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from bulkformer_dx.anomaly import head


def test_gaussian_nll_loss_matches_closed_form() -> None:
    predicted_mean = torch.tensor([1.0, -0.5], dtype=torch.float32)
    predicted_log_sigma = torch.log(torch.tensor([2.0, 0.5], dtype=torch.float32))
    targets = torch.tensor([1.5, 0.0], dtype=torch.float32)

    loss = head.gaussian_nll_loss(predicted_mean, predicted_log_sigma, targets)

    sigma = torch.tensor([2.0, 0.5], dtype=torch.float32)
    expected = 0.5 * (((targets - predicted_mean) / sigma) ** 2 + 2.0 * torch.log(sigma))
    assert torch.allclose(loss, expected.mean())


def test_inject_synthetic_outliers_touches_valid_genes_only() -> None:
    expression = np.array(
        [
            [10.0, 1.0, 5.0],
            [11.0, 1.5, 6.0],
            [12.0, 2.0, 7.0],
        ],
        dtype=np.float32,
    )
    valid_gene_flags = np.array([True, False, True], dtype=bool)

    perturbed, labels = head.inject_synthetic_outliers(
        expression,
        valid_gene_flags,
        injection_rate=1.0,
        outlier_scale=2.5,
        random_seed=7,
    )

    assert perturbed.shape == expression.shape
    assert labels.shape == expression.shape
    assert labels.dtype == bool
    assert np.all(labels[:, 1] == 0)
    assert np.allclose(perturbed[:, 1], expression[:, 1])
    assert np.all(labels[:, [0, 2]])
    assert np.any(~np.isclose(perturbed[:, [0, 2]], expression[:, [0, 2]]))


def test_train_head_model_supports_injected_outlier_mode() -> None:
    features = np.array([[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float32)
    labels = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

    result = head.train_head_model(
        features,
        labels,
        mode="injected_outlier",
        hidden_dim=4,
        epochs=60,
        learning_rate=0.1,
        batch_size=2,
        random_seed=0,
        device="cpu",
    )

    assert result.mode == "injected_outlier"
    assert result.metrics["train_examples"] == 4
    assert result.metrics["train_accuracy"] >= 0.95


def test_run_trains_sigma_head_and_writes_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expression = pd.DataFrame(
        [[10.0, 20.0, -10.0], [14.0, 23.0, -10.0]],
        columns=["ENSG1", "ENSG2", "ENSG3"],
    )
    expression.insert(0, "sample_id", ["sample_a", "sample_b"])
    expression_path = tmp_path / "aligned.tsv"
    expression.to_csv(expression_path, sep="\t", index=False)

    valid_gene_mask = pd.DataFrame(
        {
            "ensg_id": ["ENSG1", "ENSG2", "ENSG3"],
            "is_valid": [1, 1, 0],
        }
    )
    valid_gene_mask_path = tmp_path / "valid_gene_mask.tsv"
    valid_gene_mask.to_csv(valid_gene_mask_path, sep="\t", index=False)

    monkeypatch.setattr(
        head,
        "load_bulkformer_model",
        lambda **_: SimpleNamespace(model=object(), device="cpu"),
    )
    monkeypatch.setattr(
        head,
        "extract_gene_embeddings",
        lambda *_args, **_kwargs: np.array(
            [
                [[1.0, 0.0], [0.0, 1.0], [9.0, 9.0]],
                [[2.0, 0.0], [0.0, 2.0], [9.0, 9.0]],
            ],
            dtype=np.float32,
        ),
    )
    monkeypatch.setattr(
        head,
        "predict_expression",
        lambda *_args, **_kwargs: np.array(
            [
                [9.0, 19.0, -10.0],
                [13.0, 21.0, -10.0],
            ],
            dtype=np.float32,
        ),
    )

    output_dir = tmp_path / "head_outputs"
    exit_code = head.run(
        argparse.Namespace(
            input=str(expression_path),
            valid_gene_mask=str(valid_gene_mask_path),
            output_dir=str(output_dir),
            mode="sigma_nll",
            variant="37M",
            checkpoint_path=None,
            graph_path=None,
            graph_weights_path=None,
            gene_embedding_path=None,
            gene_info_path=None,
            device="cpu",
            batch_size=8,
            hidden_dim=4,
            epochs=10,
            learning_rate=0.05,
            weight_decay=0.0,
            min_sigma=1e-3,
            injection_rate=0.1,
            outlier_scale=3.0,
            random_seed=0,
        )
    )

    assert exit_code == 0
    checkpoint_path = output_dir / "sigma_nll_head.pt"
    metrics_path = output_dir / "training_metrics.json"
    assert checkpoint_path.exists()
    assert metrics_path.exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["mode"] == "sigma_nll"
    assert metrics["train_examples"] == 4
