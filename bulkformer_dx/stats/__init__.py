"""Statistical distributions and test utilities for anomaly scoring."""

from bulkformer_dx.stats.dispersion import (
    DispersionFitResult,
    fit_nb_dispersion_mle,
    fit_nb_dispersion_moments,
)
from bulkformer_dx.stats.gaussian import gaussian_logpdf, student_t_logpdf
from bulkformer_dx.stats.heterogeneity import (
    batch_entropy,
    suggest_knn_local,
    tissue_entropy,
)
from bulkformer_dx.stats.nb import nb_logpmf, outrider_two_sided_nb_pvalue

__all__ = [
    "gaussian_logpdf",
    "student_t_logpdf",
    "nb_logpmf",
    "outrider_two_sided_nb_pvalue",
    "DispersionFitResult",
    "fit_nb_dispersion_mle",
    "fit_nb_dispersion_moments",
    "tissue_entropy",
    "batch_entropy",
    "suggest_knn_local",
]
