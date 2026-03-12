"""Dataset loaders for benchmark harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_preprocess_output(preprocess_dir: Path) -> dict[str, Any]:
    """Load preprocess output directory into a dict of DataFrames/arrays."""
    preprocess_dir = Path(preprocess_dir)
    result: dict[str, Any] = {}
    for name, filename in [
        ("aligned_log1p_tpm", "aligned_log1p_tpm.tsv"),
        ("aligned_counts", "aligned_counts.tsv"),
        ("aligned_tpm", "aligned_tpm.tsv"),
        ("valid_gene_mask", "valid_gene_mask.tsv"),
        ("gene_lengths_aligned", "gene_lengths_aligned.tsv"),
        ("sample_scaling", "sample_scaling.tsv"),
    ]:
        p = preprocess_dir / filename
        if p.exists():
            sep = "\t" if p.suffix in {".tsv", ".txt"} else ","
            df = pd.read_csv(p, sep=sep)
            if name in ("aligned_log1p_tpm", "aligned_counts", "aligned_tpm"):
                if df.columns[0] != df.columns[0] or str(df.columns[0]).lower() in ("sample_id", "ensg_id"):
                    df = df.set_index(df.columns[0])
            result[name] = df
    return result


def generate_synthetic_cohort(
    n_samples: int = 50,
    n_genes: int = 500,
    *,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate a minimal synthetic cohort for smoke tests.

    Returns (aligned_log1p_tpm, valid_gene_mask, sample_scaling).
    """
    rng = np.random.default_rng(seed)
    log1p_tpm = rng.lognormal(mean=0, sigma=2, size=(n_samples, n_genes)).astype(np.float32)
    log1p_tpm = np.clip(log1p_tpm, 0, 15)
    sample_ids = [f"sample_{i}" for i in range(n_samples)]
    gene_ids = [f"ENSG{g:05d}" for g in range(n_genes)]
    aligned = pd.DataFrame(log1p_tpm, index=sample_ids, columns=gene_ids)
    valid_mask = pd.DataFrame({
        "ensg_id": gene_ids,
        "is_valid": np.ones(n_genes, dtype=int),
    })
    sample_scaling = pd.DataFrame({
        "sample_id": sample_ids,
        "S_j": np.full(n_samples, 1e6),
    }).set_index("sample_id")
    return aligned, valid_mask, sample_scaling
