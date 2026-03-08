"""Tests for BulkFormer preprocessing utilities."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bulkformer_dx.preprocess import (
    align_to_bulkformer_genes,
    counts_to_tpm,
    normalize_ensembl_id,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the diagnostics CLI from the repository root."""
    return subprocess.run(
        [sys.executable, "-m", "bulkformer_dx.cli", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_normalize_ensembl_id_strips_versions_and_whitespace() -> None:
    assert normalize_ensembl_id(" ENSG00000123456.12 ") == "ENSG00000123456"
    assert normalize_ensembl_id("ENSG00000123456") == "ENSG00000123456"
    assert normalize_ensembl_id(None) is None


def test_counts_to_tpm_matches_expected_formula() -> None:
    counts = pd.DataFrame(
        [[100.0, 50.0], [25.0, 75.0]],
        index=["sample_a", "sample_b"],
        columns=["ENSG1", "ENSG2"],
    )
    gene_lengths = {"ENSG1": 1000.0, "ENSG2": 2000.0}

    tpm, missing = counts_to_tpm(counts, gene_lengths)

    expected = pd.DataFrame(
        [
            [800000.0, 200000.0],
            [400000.0, 600000.0],
        ],
        index=counts.index,
        columns=counts.columns,
    )

    assert missing == []
    assert np.allclose(tpm.to_numpy(), expected.to_numpy())


def test_align_to_bulkformer_genes_fills_missing_entries_and_emits_mask() -> None:
    expression = pd.DataFrame(
        [[1.0, 2.0]],
        index=["sample_a"],
        columns=["ENSG1", "ENSG3"],
    )

    aligned, mask = align_to_bulkformer_genes(
        expression,
        ["ENSG1", "ENSG2", "ENSG3"],
        fill_value=-10.0,
    )

    assert list(aligned.columns) == ["ENSG1", "ENSG2", "ENSG3"]
    assert aligned.loc["sample_a", "ENSG2"] == -10.0
    assert mask.to_dict(orient="records") == [
        {"ensg_id": "ENSG1", "is_valid": 1, "is_missing_fill": 0},
        {"ensg_id": "ENSG2", "is_valid": 0, "is_missing_fill": 1},
        {"ensg_id": "ENSG3", "is_valid": 1, "is_missing_fill": 0},
    ]


def test_preprocess_cli_writes_outputs_and_normalizes_gene_ids(tmp_path: Path) -> None:
    counts_path = tmp_path / "counts.tsv"
    annotation_path = tmp_path / "annotation.tsv"
    gene_info_path = tmp_path / "bulkformer_gene_info.tsv"
    output_dir = tmp_path / "outputs"

    counts_path.write_text(
        "gene_id\tsample_a\tsample_b\n"
        "ENSG1.1\t100\t25\n"
        "ENSG2\t50\t75\n"
        "ENSG1.2\t5\t10\n",
        encoding="utf-8",
    )
    annotation_path.write_text(
        "ensg_id\tgene_length\n"
        "ENSG1\t1000\n"
        "ENSG2\t2000\n",
        encoding="utf-8",
    )
    gene_info_path.write_text(
        "ensg_id\n"
        "ENSG1\n"
        "ENSG2\n"
        "ENSG3\n",
        encoding="utf-8",
    )

    result = run_cli(
        "preprocess",
        "--counts",
        str(counts_path),
        "--annotation",
        str(annotation_path),
        "--bulkformer-gene-info",
        str(gene_info_path),
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    assert "Wrote preprocessing outputs" in result.stdout

    aligned = pd.read_csv(output_dir / "aligned_log1p_tpm.tsv", sep="\t", index_col=0)
    mask = pd.read_csv(output_dir / "valid_gene_mask.tsv", sep="\t")
    report = json.loads((output_dir / "preprocess_report.json").read_text(encoding="utf-8"))

    assert list(aligned.columns) == ["ENSG1", "ENSG2", "ENSG3"]
    assert aligned.loc["sample_a", "ENSG3"] == -10.0
    assert mask["is_valid"].tolist() == [1, 1, 0]
    assert report["collapsed_input_gene_columns"] == 1
    assert report["bulkformer_valid_gene_count"] == 2
