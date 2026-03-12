"""Tests for benchmark harness (no torch required)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bulkformer_dx.benchmark.datasets import generate_synthetic_cohort
from bulkformer_dx.benchmark.inject import inject_outliers_log1p, InjectionResult
from bulkformer_dx.benchmark.metrics import (
    benchmark_metrics,
    compute_auroc,
    compute_auprc,
    compute_recall_at_fdr,
    compute_ks_uniform,
)
from bulkformer_dx.benchmark.runner import run_benchmark_smoke, grid_run, _load_config


def test_generate_synthetic_cohort_shape() -> None:
    aligned, valid_mask, sample_scaling = generate_synthetic_cohort(
        n_samples=20, n_genes=100, seed=0
    )
    assert aligned.shape == (20, 100)
    assert len(valid_mask) == 100
    assert len(sample_scaling) == 20


def test_inject_outliers_log1p_ground_truth() -> None:
    expr = np.random.rand(10, 20).astype(np.float32)
    valid = np.ones((10, 20), dtype=bool)
    result = inject_outliers_log1p(expr, valid, n_inject=5, seed=0)
    assert isinstance(result, InjectionResult)
    assert result.ground_truth_mask.sum() == 5
    assert result.expression_perturbed.shape == expr.shape
    assert len(result.injected_sample_idx) == 5


def test_compute_auroc_perfect_score() -> None:
    gt = np.array([True, False, True, False])
    score = np.array([1.0, 0.0, 1.0, 0.0])
    assert compute_auroc(gt, score) == 1.0


def test_compute_auroc_random() -> None:
    gt = np.array([True, False, True, False])
    score = np.array([0.5, 0.5, 0.5, 0.5])
    assert 0.3 <= compute_auroc(gt, score) <= 0.7


def test_compute_recall_at_fdr() -> None:
    gt = np.array([True, False, True, False, True])
    p_adj = np.array([0.01, 0.5, 0.02, 0.1, 0.03])
    r = compute_recall_at_fdr(gt, p_adj, fdr=0.05)
    assert 0 <= r <= 1


def test_benchmark_metrics_keys() -> None:
    gt = np.zeros(100, dtype=bool)
    gt[:10] = True
    score = np.random.rand(100)
    p_adj = np.random.rand(100)
    p_raw = np.random.rand(100)
    m = benchmark_metrics(gt, score, p_adj=p_adj, p_raw=p_raw)
    assert "auroc" in m
    assert "auprc" in m
    assert "recall_at_fdr_05" in m
    assert "ks_uniform" in m


def test_run_benchmark_smoke(tmp_path: Path) -> None:
    result = run_benchmark_smoke(tmp_path, n_samples=15, n_genes=50, n_inject=5, seed=0)
    assert (tmp_path / "benchmark_summary.json").exists()
    assert "metrics" in result
    assert "auroc" in result["metrics"]
    summary_path = tmp_path / "benchmark_summary.json"
    data = json.loads(summary_path.read_text())
    assert "n_samples" in data
    assert data["n_samples"] == 15


def test_grid_run(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "dataset": {"n_samples": 10, "n_genes": 30, "n_inject": 5},
        "methods": [{"method_id": "smoke_a", "seed": 0}, {"method_id": "smoke_b", "seed": 1}],
    }))
    out_dir = tmp_path / "grid_out"
    result = grid_run(config_path, out_dir, seed=42)
    assert (out_dir / "benchmark_summary.json").exists()
    summary = json.loads((out_dir / "benchmark_summary.json").read_text())
    assert "metrics_per_method" in summary
    assert len(summary["metrics_per_method"]) == 2


def test_load_config_json(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.json"
    cfg.write_text('{"dataset": {"n_samples": 5}, "methods": []}')
    data = _load_config(cfg)
    assert data["dataset"]["n_samples"] == 5
    assert data["methods"] == []


def test_benchmark_synthetic_e2e(tmp_path: Path) -> None:
    """Synthetic end-to-end: generate data, inject outliers, run metrics, verify artifacts."""
    result = run_benchmark_smoke(tmp_path, n_samples=20, n_genes=80, n_inject=10, seed=123)
    assert "metrics" in result
    assert "output_dir" in result
    assert (tmp_path / "benchmark_summary.json").exists()
    summary = json.loads((tmp_path / "benchmark_summary.json").read_text())
    assert summary["n_samples"] == 20
    assert summary["n_genes"] == 80
    assert summary["n_injected"] == 10
    assert 0 <= summary["auroc"] <= 1
    assert 0 <= summary["auprc"] <= 1
    results_file = tmp_path / "benchmark_results.parquet"
    if not results_file.exists():
        results_file = tmp_path / "benchmark_results.tsv"
    assert results_file.exists()
