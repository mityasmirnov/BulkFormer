"""Tests for stats/gaussian (no torch required)."""

from __future__ import annotations

import numpy as np

from bulkformer_dx.stats.gaussian import gaussian_logpdf, student_t_logpdf


def test_gaussian_logpdf_shape() -> None:
    y = np.array([0.0, 1.0, 2.0])
    mu = np.array([0.0, 1.0, 2.0])
    sigma = np.array([1.0, 1.0, 1.0])
    logp = gaussian_logpdf(y, mu, sigma)
    assert logp.shape == (3,)
    assert np.all(np.isfinite(logp))


def test_gaussian_logpdf_at_mean() -> None:
    y = np.array([0.0])
    mu = np.array([0.0])
    sigma = np.array([1.0])
    logp = gaussian_logpdf(y, mu, sigma)
    expected = -0.5 * np.log(2 * np.pi)
    assert np.isclose(logp[0], expected)


def test_student_t_logpdf_shape() -> None:
    y = np.array([0.0, 1.0])
    mu = np.array([0.0, 1.0])
    sigma = np.array([1.0, 1.0])
    logp = student_t_logpdf(y, mu, sigma, df=5.0)
    assert logp.shape == (2,)
    assert np.all(np.isfinite(logp))


def test_student_t_heavier_tails() -> None:
    y = np.array([5.0])
    mu = np.array([0.0])
    sigma = np.array([1.0])
    logp_g = gaussian_logpdf(y, mu, sigma)
    logp_t = student_t_logpdf(y, mu, sigma, df=5.0)
    assert logp_t > logp_g
