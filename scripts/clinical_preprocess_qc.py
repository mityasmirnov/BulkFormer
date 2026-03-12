#!/usr/bin/env python3
"""Generate QC figures for clinical preprocess outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", index_col=0)


def main() -> int:
    preprocess_dir = Path("runs/clinical_preprocess_37M")
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not preprocess_dir.exists():
        print(f"Preprocess dir not found: {preprocess_dir}", file=sys.stderr)
        return 1

    tpm = _read_tsv(preprocess_dir / "tpm.tsv")
    log1p_tpm = _read_tsv(preprocess_dir / "log1p_tpm.tsv")
    aligned = _read_tsv(preprocess_dir / "aligned_log1p_tpm.tsv")
    valid_mask = _read_tsv(preprocess_dir / "valid_gene_mask.tsv")

    with open(preprocess_dir / "preprocess_report.json") as f:
        report = json.load(f)

    # 1. Total TPM per sample
    fig, ax = plt.subplots(figsize=(8, 4))
    tpm_totals = tpm.sum(axis=1)
    ax.hist(tpm_totals, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(1e6, color="red", linestyle="--", label="Expected 1e6")
    ax.set_xlabel("Total TPM per sample")
    ax.set_ylabel("Count")
    ax.set_title("Clinical Preprocess: TPM Totals")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "clinical_preprocess_tpm_total_hist.png", dpi=150)
    plt.close()

    # 2. log1p(TPM) histogram (non-fill values only)
    fig, ax = plt.subplots(figsize=(8, 4))
    values = aligned.values.flatten()
    non_fill = values[values > -9.9]
    ax.hist(non_fill, bins=80, edgecolor="black", alpha=0.7)
    ax.set_xlabel("log1p(TPM)")
    ax.set_ylabel("Count")
    ax.set_title("Clinical Preprocess: log1p(TPM) Distribution (non-fill)")
    fig.tight_layout()
    fig.savefig(figures_dir / "clinical_preprocess_log1p_hist.png", dpi=150)
    plt.close()

    # 3. Valid gene fraction
    fig, ax = plt.subplots(figsize=(6, 4))
    valid_frac = report.get("bulkformer_valid_gene_fraction", 0)
    ax.bar(["Valid", "Missing"], [valid_frac, 1 - valid_frac], color=["green", "gray"], alpha=0.7)
    ax.set_ylabel("Fraction")
    ax.set_title(f"BulkFormer Gene Coverage: {valid_frac:.2%} valid")
    fig.tight_layout()
    fig.savefig(figures_dir / "clinical_preprocess_valid_gene_fraction.png", dpi=150)
    plt.close()

    print(f"Saved QC figures to {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
