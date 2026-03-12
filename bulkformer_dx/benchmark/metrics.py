"""Benchmark metrics for anomaly detection evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_auroc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """Compute AUROC. Higher score = more anomalous ranks higher."""
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true, dtype=bool).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    if y_true.sum() == 0 or (~y_true).sum() == 0:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def compute_auprc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """Compute AUPRC (area under precision-recall curve)."""
    from sklearn.metrics import average_precision_score
    y_true = np.asarray(y_true, dtype=bool).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    if y_true.sum() == 0:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def compute_recall_at_fdr(
    y_true: np.ndarray,
    p_adj: np.ndarray,
    fdr: float = 0.05,
) -> float:
    """Recall at given FDR: fraction of true positives among discoveries at FDR threshold."""
    y_true = np.asarray(y_true, dtype=bool).ravel()
    p_adj = np.asarray(p_adj, dtype=float).ravel()
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return 1.0
    discoveries = p_adj <= fdr
    if discoveries.sum() == 0:
        return 0.0
    tp = (y_true & discoveries).sum()
    return float(tp / n_pos)


def compute_precision_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int = 100,
) -> float:
    """Precision at top-k: fraction of true positives among top-k by score.

    Ranks by y_score descending (higher = more anomalous). Ties broken by order.

    Args:
        y_true: Binary ground truth.
        y_score: Anomaly score (higher = more anomalous).
        k: Number of top predictions to consider.

    Returns:
        Precision = TP / min(k, n_predicted).
    """
    y_true = np.asarray(y_true, dtype=bool).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    if y_true.size == 0 or k <= 0:
        return 0.0
    order = np.argsort(-y_score)
    top_k = order[: min(k, len(order))]
    tp = y_true[top_k].sum()
    return float(tp / len(top_k))


def compute_ks_uniform(p_values: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic against Uniform(0,1)."""
    from scipy.stats import kstest
    p_values = np.asarray(p_values, dtype=float).ravel()
    p_values = p_values[np.isfinite(p_values) & (p_values >= 0) & (p_values <= 1)]
    if p_values.size < 2:
        return 0.0
    stat, _ = kstest(p_values, "uniform")
    return float(stat)


def benchmark_metrics(
    ground_truth: np.ndarray,
    score: np.ndarray,
    p_adj: np.ndarray | None = None,
    p_raw: np.ndarray | None = None,
    k: int = 100,
) -> dict[str, float]:
    """Compute standard benchmark metrics."""
    gt = ground_truth.ravel()
    sc = score.ravel()
    result: dict[str, Any] = {
        "auroc": compute_auroc(gt, sc),
        "auprc": compute_auprc(gt, sc),
        "precision_at_k": compute_precision_at_k(gt, sc, k=k),
    }
    if p_adj is not None:
        result["recall_at_fdr_05"] = compute_recall_at_fdr(gt, p_adj.ravel(), fdr=0.05)
        result["recall_at_fdr_10"] = compute_recall_at_fdr(gt, p_adj.ravel(), fdr=0.1)
    if p_raw is not None and np.isfinite(p_raw).sum() > 0:
        result["ks_uniform"] = compute_ks_uniform(p_raw.ravel())
    return {k: float(v) for k, v in result.items()}
