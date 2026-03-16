"""Tests for the standardized BulkFormer inference API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from bulkformer_dx.io.schemas import AlignedExpressionBundle, ModelPredictionBundle
from bulkformer_dx.model.bulkformer import (
    bundle_from_paths,
    bundle_from_preprocess_result,
    mc_predict,
    predict_mean,
)


@dataclass
class _LoadedModel:
    model: torch.nn.Module
    device: torch.device


class _DummyBulkFormer(torch.nn.Module):
    """Minimal model for API tests."""

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask_prob: float | None = None,
        output_expr: bool = False,
    ) -> torch.Tensor:
        if output_expr:
            return x + float(mask_prob or 0.0)
        return torch.stack((x, x + 1.0, x + 2.0), dim=-1)


def _make_bundle(n_samples: int = 4, n_genes: int = 10) -> AlignedExpressionBundle:
    rng = np.random.default_rng(42)
    Y = rng.uniform(0, 5, (n_samples, n_genes)).astype(np.float32)
    valid_mask = np.ones((n_samples, n_genes), dtype=bool)
    valid_mask[:, -2:] = False
    return AlignedExpressionBundle(
        expr_space="log1p_tpm",
        Y_obs=Y,
        valid_mask=valid_mask,
        gene_ids=[f"ENSG{i:05d}" for i in range(n_genes)],
        sample_ids=[f"S{i}" for i in range(n_samples)],
        counts=None,
        gene_length_kb=None,
        tpm_scaling_S=None,
        metadata=None,
    )


def test_predict_mean_returns_model_prediction_bundle() -> None:
    bundle = _make_bundle()
    model = _DummyBulkFormer()
    loaded = _LoadedModel(model=model, device=torch.device("cpu"))

    pred = predict_mean(bundle, loaded_model=loaded, batch_size=2)

    assert isinstance(pred, ModelPredictionBundle)
    assert pred.y_hat.shape == bundle.Y_obs.shape
    assert pred.embedding is not None
    assert pred.embedding.shape[0] == bundle.Y_obs.shape[0]
    assert pred.sigma_hat is None
    assert pred.mc_samples is None


def test_predict_mean_reconstruction_with_dummy_model() -> None:
    bundle = _make_bundle()
    model = _DummyBulkFormer()
    loaded = _LoadedModel(model=model, device=torch.device("cpu"))

    pred = predict_mean(bundle, loaded_model=loaded, batch_size=2)

    expected_y_hat = bundle.Y_obs + 0.0
    np.testing.assert_allclose(pred.y_hat, expected_y_hat, rtol=1e-5)


def test_mc_predict_returns_bundle_and_samples() -> None:
    bundle = _make_bundle(n_samples=2, n_genes=6)
    model = _DummyBulkFormer()
    loaded = _LoadedModel(model=model, device=torch.device("cpu"))

    pred_bundle, mc_samples = mc_predict(
        bundle,
        mc_passes=4,
        mask_prob=0.3,
        seed=0,
        batch_size=2,
        loaded_model=loaded,
    )

    assert isinstance(pred_bundle, ModelPredictionBundle)
    assert pred_bundle.mc_samples is not None
    assert mc_samples.shape == (4, 2, 6)
    assert pred_bundle.y_hat.shape == (2, 6)
    np.testing.assert_allclose(
        pred_bundle.y_hat,
        np.nanmean(mc_samples, axis=0),
        rtol=1e-5,
    )


def test_bundle_from_paths(tmp_path: Path) -> None:
    n_samples, n_genes = 3, 5
    expr = pd.DataFrame(
        np.random.rand(n_samples, n_genes).astype(np.float32),
        index=[f"S{i}" for i in range(n_samples)],
        columns=[f"ENSG{i:05d}" for i in range(n_genes)],
    )
    expr.index.name = "sample_id"
    expr.to_csv(tmp_path / "aligned_log1p_tpm.tsv", sep="\t")

    valid_mask = pd.DataFrame(
        {"ensg_id": expr.columns.tolist(), "is_valid": [1] * n_genes}
    )
    valid_mask.to_csv(tmp_path / "valid_gene_mask.tsv", sep="\t", index=False)

    bundle = bundle_from_paths(tmp_path)

    assert bundle.Y_obs.shape == (n_samples, n_genes)
    assert bundle.gene_ids == list(expr.columns)
    assert bundle.sample_ids == list(expr.index)


def test_mc_predict_with_loaded_model() -> None:
    """When loaded_model is provided, mc_predict uses it without loading."""
    bundle = _make_bundle(n_samples=2, n_genes=6)
    model = _DummyBulkFormer()
    loaded = _LoadedModel(model=model, device=torch.device("cpu"))

    pred_bundle, mc_samples = mc_predict(
        bundle,
        loaded_model=loaded,
        mc_passes=2,
        mask_prob=0.3,
        seed=0,
        batch_size=2,
    )

    assert pred_bundle.y_hat.shape == (2, 6)
    assert mc_samples.shape == (2, 2, 6)


def test_mc_predict_populates_sigma_hat_when_multiple_passes() -> None:
    """mc_predict sets sigma_hat from mc_variance when mc_passes > 1."""
    bundle = _make_bundle(n_samples=2, n_genes=6)
    model = _DummyBulkFormer()
    loaded = _LoadedModel(model=model, device=torch.device("cpu"))

    pred_bundle, mc_samples = mc_predict(
        bundle,
        loaded_model=loaded,
        mc_passes=4,
        mask_prob=0.3,
        seed=0,
        batch_size=2,
    )

    assert pred_bundle.sigma_hat is not None
    assert pred_bundle.sigma_hat.shape == (2, 6)
    np.testing.assert_array_less(0, pred_bundle.sigma_hat)


def test_mc_predict_calls_progress_callback() -> None:
    """progress_callback is invoked each pass with (pass_idx, total)."""
    bundle = _make_bundle(n_samples=2, n_genes=6)
    model = _DummyBulkFormer()
    loaded = _LoadedModel(model=model, device=torch.device("cpu"))
    calls: list[tuple[int, int]] = []

    pred_bundle, mc_samples = mc_predict(
        bundle,
        loaded_model=loaded,
        mc_passes=3,
        mask_prob=0.3,
        seed=0,
        batch_size=2,
        progress_callback=lambda p, t: calls.append((p, t)),
    )

    assert len(calls) == 3
    assert calls == [(1, 3), (2, 3), (3, 3)]


def test_predict_dispatches_by_method_config() -> None:
    """Unified predict() uses mc_predict when mc_passes > 0, else predict_mean."""
    from bulkformer_dx.io.schemas import MethodConfig
    from bulkformer_dx.model.bulkformer import predict

    bundle = _make_bundle(n_samples=2, n_genes=6)
    model = _DummyBulkFormer()
    loaded = _LoadedModel(model=model, device=torch.device("cpu"))

    config_mean = MethodConfig(method_id="mean", space="log1p_tpm", mc_passes=0)
    pred_mean = predict(bundle, config_mean, loaded_model=loaded, batch_size=2)
    assert pred_mean.sigma_hat is None
    assert pred_mean.mc_samples is None

    config_mc = MethodConfig(method_id="mc", space="log1p_tpm", mc_passes=2, seed=0)
    pred_mc = predict(bundle, config_mc, loaded_model=loaded, batch_size=2)
    assert pred_mc.sigma_hat is not None
    assert pred_mc.mc_samples is not None
