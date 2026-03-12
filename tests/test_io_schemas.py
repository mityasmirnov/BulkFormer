"""Tests for io/schemas (no torch required)."""

from __future__ import annotations

import numpy as np

from bulkformer_dx.io.schemas import (
    AlignedExpressionBundle,
    ModelPredictionBundle,
    MethodConfig,
    GeneOutlierRow,
    SampleOutlierRow,
)


def test_aligned_expression_bundle_shape() -> None:
    Y = np.random.rand(5, 10).astype(np.float32)
    valid = np.ones((5, 10), dtype=bool)
    bundle = AlignedExpressionBundle(
        expr_space="log1p_tpm",
        Y_obs=Y,
        valid_mask=valid,
        gene_ids=[f"ENSG{i}" for i in range(10)],
        sample_ids=[f"s{i}" for i in range(5)],
    )
    assert bundle.Y_obs.shape == (5, 10)
    assert len(bundle.gene_ids) == 10
    assert len(bundle.sample_ids) == 5


def test_model_prediction_bundle() -> None:
    y_hat = np.random.rand(3, 5).astype(np.float32)
    preds = ModelPredictionBundle(y_hat=y_hat)
    assert preds.y_hat.shape == (3, 5)
    assert preds.sigma_hat is None
    assert preds.mc_samples is None


def test_method_config_defaults() -> None:
    cfg = MethodConfig(method_id="test", space="log1p_tpm")
    assert cfg.cohort_mode == "global"
    assert cfg.multiple_testing == "BY"
    assert cfg.mc_passes == 16


def test_gene_outlier_row() -> None:
    row = GeneOutlierRow(
        sample_id="s1",
        gene_id="ENSG1",
        y_obs=3.0,
        y_hat=2.8,
        residual=0.2,
        score_gene=0.5,
        p_raw=0.1,
        p_adj=0.2,
        direction="over",
        method_id="residual",
        diagnostics_json={"masked_count": 4},
    )
    assert row.sample_id == "s1"
    assert row.gene_id == "ENSG1"
    assert row.score_gene == 0.5


def test_sample_outlier_row() -> None:
    row = SampleOutlierRow(
        sample_id="s1",
        score_sample=1.2,
        cohort_mode="global",
        method_id="residual",
    )
    assert row.sample_id == "s1"
    assert row.score_sample == 1.2
