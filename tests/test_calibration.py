"""Tests for bulkformer_dx.calibration (pvalues, multitest, cohort).

These tests do not require torch and can run without the full anomaly stack.
"""

from __future__ import annotations

import numpy as np
import pytest

from bulkformer_dx.calibration import (
    benjamini_hochberg,
    benjamini_yekutieli,
    empirical_tail_pvalue,
    zscore_two_sided_pvalue,
)
from bulkformer_dx.calibration.cohort import get_cohort_indices
from bulkformer_dx.calibration.multitest import apply_within_sample


def test_benjamini_yekutieli_matches_known_values() -> None:
    p_values = np.array([0.01, 0.02, 0.5], dtype=float)
    adjusted = benjamini_yekutieli(p_values)
    assert np.allclose(adjusted, [0.055, 0.055, 0.9166666667])


def test_benjamini_hochberg() -> None:
    p_values = np.array([0.01, 0.05, 0.1, 0.5])
    adjusted = benjamini_hochberg(p_values)
    assert adjusted.shape == p_values.shape
    assert np.all(adjusted >= 0) and np.all(adjusted <= 1)
    assert np.all(np.diff(adjusted[np.argsort(p_values)]) >= 0)


def test_empirical_tail_pvalue() -> None:
    dist = np.array([0.1, 0.2, 0.3, 0.5])
    pv = empirical_tail_pvalue(dist, 0.25, upper_tail=True)
    assert 0 <= pv <= 1
    # 0.25 is between 0.2 and 0.3; 2 values >= 0.25 -> (2+1)/(4+1) = 0.6
    assert np.isclose(pv, 0.6)


def test_zscore_two_sided_pvalue() -> None:
    z = np.array([0.0, 1.96, -2.0])
    pv = zscore_two_sided_pvalue(z)
    assert pv.shape == z.shape
    assert np.all((pv >= 0) & (pv <= 1))
    assert np.isclose(pv[0], 1.0)
    assert pv[1] < 0.1
    assert pv[2] < 0.1


def test_zscore_two_sided_pvalue_student_t() -> None:
    z = np.array([0.0, 1.0])
    pv = zscore_two_sided_pvalue(z, use_student_t=True, student_t_df=5.0)
    assert pv.shape == z.shape
    assert np.isclose(pv[0], 1.0)


def test_apply_within_sample() -> None:
    pmat = np.array([[0.01, 0.05, 0.1], [0.5, 0.1, 0.01]])
    adj = apply_within_sample(pmat, method="BY")
    assert adj.shape == pmat.shape
    assert np.all(adj >= 0) and np.all(adj <= 1)


def test_get_cohort_indices_global() -> None:
    sample_ids = ["a", "b", "c"]
    result = get_cohort_indices(sample_ids, cohort_mode="global")
    assert len(result) == 3
    assert set(result["a"]) == {1, 2}
    assert set(result["b"]) == {0, 2}
    assert set(result["c"]) == {0, 1}


def test_get_cohort_indices_knn_local() -> None:
    sample_ids = ["a", "b", "c", "d"]
    embedding = np.random.default_rng(42).standard_normal((4, 8))
    result = get_cohort_indices(
        sample_ids,
        cohort_mode="knn_local",
        embedding=embedding,
        knn_k=2,
    )
    assert len(result) == 4
    for sid, indices in result.items():
        assert len(indices) == 2
        assert all(0 <= i < 4 for i in indices)


def test_get_cohort_indices_knn_local_requires_embedding() -> None:
    with pytest.raises(ValueError, match="knn_local.*requires embeddings"):
        get_cohort_indices(["a", "b"], cohort_mode="knn_local", embedding=None)
