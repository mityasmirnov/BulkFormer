"""Tests for heterogeneity metrics (kNN-local gate)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bulkformer_dx.stats.heterogeneity import (
    batch_entropy,
    suggest_knn_local,
    tissue_entropy,
)


def test_tissue_entropy_single_tissue() -> None:
    """Single tissue has near-zero entropy."""
    labels = pd.Series(["brain"] * 10, index=[f"s{i}" for i in range(10)])
    assert np.isclose(tissue_entropy(labels), 0.0)


def test_tissue_entropy_two_tissues() -> None:
    """Two equal tissues have positive entropy."""
    labels = pd.Series(
        ["brain"] * 5 + ["liver"] * 5,
        index=[f"s{i}" for i in range(10)],
    )
    ent = tissue_entropy(labels)
    assert ent > 0
    assert np.isclose(ent, np.log(2))


def test_batch_entropy_matches_tissue_entropy() -> None:
    """batch_entropy uses same formula as tissue_entropy."""
    labels = pd.Series(["a", "b", "c"] * 3, index=[f"s{i}" for i in range(9)])
    assert batch_entropy(labels) == tissue_entropy(labels)


def test_suggest_knn_local_no_metadata() -> None:
    """No metadata returns recommend=True with message."""
    recommend, reason = suggest_knn_local()
    assert recommend is True
    assert "No metadata" in reason


def test_suggest_knn_local_low_tissue_entropy() -> None:
    """Single-tissue cohort returns recommend=False."""
    labels = pd.Series(["brain"] * 20, index=[f"s{i}" for i in range(20)])
    recommend, reason = suggest_knn_local(tissue_labels=labels)
    assert recommend is False
    assert "low tissue entropy" in reason or "single-tissue" in reason


def test_suggest_knn_local_high_tissue_entropy() -> None:
    """Diverse tissue cohort returns recommend=True."""
    labels = pd.Series(
        ["brain", "liver", "heart", "lung"] * 5,
        index=[f"s{i}" for i in range(20)],
    )
    recommend, reason = suggest_knn_local(tissue_labels=labels)
    assert recommend is True
    assert "sufficient" in reason or "Heterogeneity" in reason
