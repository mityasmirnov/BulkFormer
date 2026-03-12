"""QC and benchmark plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from bulkformer_dx.preprocess import PreprocessResult


def generate_preprocess_qc_plots(
    result: PreprocessResult,
    figures_dir: Path,
) -> list[Path]:
    """Generate QC plots for preprocessing outputs. Returns paths to saved figures."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    figures_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # 1. Library sizes (total counts per sample)
    fig, ax = plt.subplots(figsize=(8, 4))
    library_sizes = result.counts.sum(axis=1)
    ax.hist(library_sizes, bins=min(50, max(10, len(library_sizes) // 2)), edgecolor="black", alpha=0.7)
    ax.set_xlabel("Library size (total counts per sample)")
    ax.set_ylabel("Count")
    ax.set_title("Library Size Distribution")
    fig.tight_layout()
    p = figures_dir / "step_two_library_sizes.png"
    fig.savefig(p, dpi=150)
    plt.close()
    saved.append(p)

    # 2. S_j distribution (sample scaling factor)
    fig, ax = plt.subplots(figsize=(8, 4))
    s_j = result.sample_scaling["S_j"]
    ax.hist(s_j, bins=min(50, max(10, len(s_j) // 2)), edgecolor="black", alpha=0.7)
    ax.set_xlabel(r"$S_j$ (sum of K_{jh}/L^{kb}_h per sample)")
    ax.set_ylabel("Count")
    ax.set_title(r"Sample Scaling Factor $S_j$ Distribution")
    fig.tight_layout()
    p = figures_dir / "step_two_sample_scaling.png"
    fig.savefig(p, dpi=150)
    plt.close()
    saved.append(p)

    # 3. Gene length histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    lengths = result.gene_lengths_aligned["length_kb"] * 1000  # back to bp for display
    lengths_valid = lengths[result.gene_lengths_aligned["has_length"] == 1]
    if len(lengths_valid) > 0:
        ax.hist(lengths_valid, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Gene length (bp)")
    ax.set_ylabel("Count")
    ax.set_title("Gene Length Distribution (aligned panel, has_length=1)")
    fig.tight_layout()
    p = figures_dir / "step_two_gene_length_hist.png"
    fig.savefig(p, dpi=150)
    plt.close()
    saved.append(p)

    # 4. Fraction of valid genes per sample
    valid_mask = result.valid_gene_mask.set_index("ensg_id")["is_valid"]
    valid_per_sample = valid_mask.sum() / len(valid_mask)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["Valid", "Missing"],
        [valid_per_sample, 1 - valid_per_sample],
        color=["green", "gray"],
        alpha=0.7,
    )
    ax.set_ylabel("Fraction")
    ax.set_title(f"BulkFormer Gene Coverage: {valid_per_sample:.2%} valid")
    fig.tight_layout()
    p = figures_dir / "step_two_valid_gene_fraction.png"
    fig.savefig(p, dpi=150)
    plt.close()
    saved.append(p)

    return saved


def build_preprocess_sanity_table(
    result: PreprocessResult,
    n_genes: int = 5,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a sanity table for n random valid genes: counts, TPM, log1p(TPM), mapping terms."""
    rng = np.random.default_rng(seed)
    valid_genes = result.valid_gene_mask[
        result.valid_gene_mask["is_valid"] == 1
    ]["ensg_id"].tolist()
    if len(valid_genes) == 0:
        return pd.DataFrame()
    n = min(n_genes, len(valid_genes))
    chosen = rng.choice(valid_genes, size=n, replace=False).tolist()

    sample_id = result.aligned_counts.index[0]
    lengths = result.gene_lengths_aligned.set_index("ensg_id")["length_kb"]
    s_j = result.sample_scaling.loc[sample_id, "S_j"]

    rows = []
    for g in chosen:
        counts = result.aligned_counts.loc[sample_id, g]
        tpm = result.aligned_tpm.loc[sample_id, g]
        log1p_tpm = result.aligned_log1p_tpm.loc[sample_id, g]
        l_kb = lengths.loc[g]
        mapping_term = tpm * s_j / 1e6 * l_kb if l_kb > 0 else np.nan
        rows.append({
            "ensg_id": g,
            "counts": counts,
            "TPM": tpm,
            "log1p_TPM": log1p_tpm,
            "length_kb": l_kb,
            "S_j": s_j,
            "expected_count_mapping": mapping_term,
        })
    return pd.DataFrame(rows)


def plot_pr_curve(
    ground_truth: np.ndarray,
    score: np.ndarray,
    output_path: Path,
) -> Path | None:
    """Plot precision-recall curve. Higher score = more anomalous.

    Returns output_path if successful, None if matplotlib unavailable.
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve
    except ImportError:
        return None
    ground_truth = np.asarray(ground_truth, dtype=bool).ravel()
    score = np.asarray(score, dtype=float).ravel()
    if ground_truth.sum() == 0:
        return None
    precision, recall, _ = precision_recall_curve(ground_truth, score)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_pvalue_histogram(
    p_values: np.ndarray,
    output_path: Path,
) -> Path | None:
    """Plot histogram of p-values (expect ~uniform under null)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    p_values = np.asarray(p_values, dtype=float).ravel()
    p_values = p_values[np.isfinite(p_values) & (p_values >= 0) & (p_values <= 1)]
    if p_values.size < 2:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(p_values, bins=20, edgecolor="black", alpha=0.7)
    ax.axhline(p_values.size / 20, color="red", linestyle="--", label="Expected (uniform)")
    ax.set_xlabel("p-value")
    ax.set_ylabel("Count")
    ax.set_title("P-value Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_pvalue_qq(
    p_values: np.ndarray,
    output_path: Path,
) -> Path | None:
    """QQ plot of p-values vs Uniform(0,1)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    p_values = np.asarray(p_values, dtype=float).ravel()
    p_values = np.sort(p_values[np.isfinite(p_values) & (p_values >= 0) & (p_values <= 1)])
    if p_values.size < 2:
        return None
    n = len(p_values)
    theoretical = np.linspace(0, 1, n + 2)[1:-1]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(theoretical, p_values, alpha=0.5, s=5)
    ax.plot([0, 1], [0, 1], "r--", label="Uniform(0,1)")
    ax.set_xlabel("Theoretical quantile")
    ax.set_ylabel("Sample quantile")
    ax.set_title("P-value QQ Plot")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()
    return output_path
