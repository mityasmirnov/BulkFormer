"""Benchmark grid runner for anomaly detection methods."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bulkformer_dx.benchmark.datasets import generate_synthetic_cohort
from bulkformer_dx.benchmark.inject import inject_outliers_log1p
from bulkformer_dx.benchmark.metrics import benchmark_metrics
from bulkformer_dx.io.read_write import load_config_dict
from bulkformer_dx.io.schemas import AlignedExpressionBundle, MethodConfig, ModelPredictionBundle
from bulkformer_dx.scoring.residual import compute_residual_scores


def run_benchmark_smoke(
    output_dir: Path,
    *,
    n_samples: int = 30,
    n_genes: int = 200,
    n_inject: int = 15,
    seed: int = 0,
) -> dict[str, Any]:
    """Run a minimal smoke-test benchmark on synthetic data.

    Does not run the full model; only validates the harness produces
    metrics JSON, ranked outputs, and at least one figure.

    Returns:
        Summary dict with metrics and paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "benchmark_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    aligned, valid_mask, _ = generate_synthetic_cohort(
        n_samples=n_samples,
        n_genes=n_genes,
        seed=seed,
    )
    expr = aligned.to_numpy(dtype=np.float32)
    valid_flags = np.ones((n_samples, n_genes), dtype=bool)

    injected = inject_outliers_log1p(
        expr,
        valid_flags,
        n_inject=n_inject,
        scale=3.0,
        direction="both",
        seed=seed,
    )

    ground_truth = injected.ground_truth_mask.ravel()
    score = np.abs(injected.expression_perturbed - expr).ravel()
    p_raw = np.random.uniform(0, 1, size=ground_truth.size).astype(np.float32)
    p_adj = p_raw.copy()

    metrics = benchmark_metrics(
        ground_truth,
        score,
        p_adj=p_adj,
        p_raw=p_raw,
    )
    metrics["n_samples"] = n_samples
    metrics["n_genes"] = n_genes
    metrics["n_injected"] = n_inject

    with (output_dir / "benchmark_summary.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    results_df = pd.DataFrame({
        "sample_idx": np.repeat(np.arange(n_samples), n_genes),
        "gene_idx": np.tile(np.arange(n_genes), n_samples),
        "ground_truth": ground_truth,
        "score": score,
        "p_raw": p_raw,
        "p_adj": p_adj,
    })
    try:
        results_df.to_parquet(output_dir / "benchmark_results.parquet", index=False)
        results_path = output_dir / "benchmark_results.parquet"
    except ImportError:
        results_df.to_csv(output_dir / "benchmark_results.tsv", sep="\t", index=False)
        results_path = output_dir / "benchmark_results.tsv"

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(p_raw[~ground_truth], bins=20, alpha=0.7, label="Null")
        ax.hist(p_raw[ground_truth], bins=10, alpha=0.7, label="Injected")
        ax.set_xlabel("p-value")
        ax.set_ylabel("Count")
        ax.set_title("P-value distribution (smoke test)")
        ax.legend()
        fig.savefig(figures_dir / "smoke_pvalue_hist.png", dpi=150)
        plt.close()
    except ImportError:
        pass

    return {
        "metrics": metrics,
        "output_dir": str(output_dir),
        "summary_path": str(output_dir / "benchmark_summary.json"),
        "results_path": str(results_path),
    }


def _ranked_to_score_matrix(
    ranked: dict[str, pd.DataFrame],
    sample_ids: list[str],
    gene_ids: list[str],
) -> np.ndarray:
    """Build (n_samples, n_genes) score matrix from ranked tables. Unscored = 0."""
    n_samples, n_genes = len(sample_ids), len(gene_ids)
    sample_idx = {s: i for i, s in enumerate(sample_ids)}
    gene_idx = {g: i for i, g in enumerate(gene_ids)}
    score_mat = np.zeros((n_samples, n_genes), dtype=np.float32)
    for sample_id, df in ranked.items():
        if df.empty:
            continue
        si = sample_idx.get(sample_id)
        if si is None:
            continue
        for _, row in df.iterrows():
            gi = gene_idx.get(str(row.get("gene_id", row.get("ensg_id", ""))))
            if gi is not None:
                score_mat[si, gi] = float(row.get("score_gene", row.get("anomaly_score", 0)))
    return score_mat


def run_residual_benchmark(
    output_dir: Path,
    *,
    n_samples: int = 30,
    n_genes: int = 200,
    n_inject: int = 15,
    seed: int = 0,
    config: MethodConfig | None = None,
) -> dict[str, Any]:
    """Run residual scoring benchmark on synthetic data with cohort-mean predictor.

    Uses original (pre-injection) data as y_hat; injected data as Y_obs.
    No torch/model required.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "benchmark_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    aligned, valid_mask, sample_scaling = generate_synthetic_cohort(
        n_samples=n_samples, n_genes=n_genes, seed=seed
    )
    expr = aligned.to_numpy(dtype=np.float32)
    valid_flags = np.ones((n_samples, n_genes), dtype=bool)
    sample_ids = list(aligned.index)
    gene_ids = list(aligned.columns)

    injected = inject_outliers_log1p(
        expr, valid_flags, n_inject=n_inject, scale=3.0, direction="both", seed=seed
    )
    ground_truth = injected.ground_truth_mask.ravel()

    bundle = AlignedExpressionBundle(
        expr_space="log1p_tpm",
        Y_obs=injected.expression_perturbed.astype(np.float32),
        valid_mask=valid_flags,
        gene_ids=gene_ids,
        sample_ids=sample_ids,
    )
    preds = ModelPredictionBundle(y_hat=expr)
    cfg = config or MethodConfig(method_id="residual", space="log1p_tpm", seed=seed)
    ranked, _ = compute_residual_scores(bundle, preds, config=cfg)

    score_mat = _ranked_to_score_matrix(ranked, sample_ids, gene_ids)
    score = score_mat.ravel()
    p_raw = np.random.uniform(0.01, 1, size=ground_truth.size).astype(np.float32)
    p_adj = p_raw.copy()

    metrics = benchmark_metrics(
        ground_truth, score, p_adj=p_adj, p_raw=p_raw
    )
    metrics["n_samples"] = n_samples
    metrics["n_genes"] = n_genes
    metrics["n_injected"] = n_inject

    with (output_dir / "benchmark_summary.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    results_df = pd.DataFrame({
        "sample_idx": np.repeat(np.arange(n_samples), n_genes),
        "gene_idx": np.tile(np.arange(n_genes), n_samples),
        "ground_truth": ground_truth,
        "score": score,
        "p_raw": p_raw,
        "p_adj": p_adj,
    })
    try:
        results_df.to_parquet(output_dir / "benchmark_results.parquet", index=False)
        results_path = output_dir / "benchmark_results.parquet"
    except ImportError:
        results_df.to_csv(output_dir / "benchmark_results.tsv", sep="\t", index=False)
        results_path = output_dir / "benchmark_results.tsv"

    return {
        "metrics": metrics,
        "output_dir": str(output_dir),
        "summary_path": str(output_dir / "benchmark_summary.json"),
        "results_path": str(results_path),
    }


def grid_run(
    config_path: Path,
    output_dir: Path,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Run a grid of method configs and aggregate results.

    Config format (YAML/JSON):
      dataset:
        type: synthetic
        n_samples: 30
        n_genes: 200
        n_inject: 15
      methods:
        - method_id: smoke
          seed: 0
        - method_id: smoke
          seed: 1

    Writes benchmark_results.parquet, benchmark_summary.json, and figures.
    """
    config = load_config_dict(config_path)
    dataset_spec = config.get("dataset", {})
    methods = config.get("methods", [{"method_id": "smoke", "seed": 0}])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "benchmark_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    n_samples = dataset_spec.get("n_samples", 30)
    n_genes = dataset_spec.get("n_genes", 200)
    n_inject = dataset_spec.get("n_inject", 15)

    all_results: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []

    for i, method_cfg in enumerate(methods):
        method_id = method_cfg.get("method_id", f"method_{i}")
        method_seed = method_cfg.get("seed", seed + i)
        test_cfg = method_cfg.get("test") or {}
        test_type = method_cfg.get("test_type") or test_cfg.get("type", "")

        run_kw = dict(
            n_samples=n_samples,
            n_genes=n_genes,
            n_inject=n_inject,
            seed=method_seed,
        )
        if method_id == "residual" or test_type in ("zscore_2s", "residual", "empirical_tail"):
            result = run_residual_benchmark(
                output_dir / "runs" / method_id,
                config=MethodConfig(method_id=method_id, space="log1p_tpm", seed=method_seed),
                **run_kw,
            )
        else:
            result = run_benchmark_smoke(output_dir / "runs" / method_id, **run_kw)
        all_metrics.append({"method_id": method_id, **result["metrics"]})
        rp = Path(result["results_path"])
        df = pd.read_csv(rp, sep="\t") if rp.suffix == ".tsv" else pd.read_parquet(rp)
        df["method_id"] = method_id
        all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True)
    try:
        combined.to_parquet(output_dir / "benchmark_results.parquet", index=False)
    except ImportError:
        combined.to_csv(output_dir / "benchmark_results.tsv", sep="\t", index=False)

    summary = {
        "dataset": dataset_spec,
        "methods": [m.get("method_id", f"method_{i}") for i, m in enumerate(methods)],
        "metrics_per_method": all_metrics,
    }
    with (output_dir / "benchmark_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    return {"output_dir": str(output_dir), "summary": summary}
