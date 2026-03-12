"""Tests for BulkFormer preprocessing utilities."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bulkformer_dx.preprocess import (
    align_counts_to_bulkformer,
    align_to_bulkformer_genes,
    build_gene_lengths_aligned,
    compute_sample_scaling,
    counts_to_tpm,
    load_counts_matrix,
    load_gene_lengths,
    normalize_ensembl_id,
    preprocess_counts,
    write_preprocess_outputs,
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


def test_load_gene_lengths_can_fall_back_to_genomic_span(tmp_path: Path) -> None:
    annotation_path = tmp_path / "annotation.tsv"
    annotation_path.write_text(
        "gene_id\tstart\tend\n"
        "ENSG1.1\t100\t199\n"
        "ENSG2\t500\t749\n",
        encoding="utf-8",
    )

    gene_lengths, metadata = load_gene_lengths(annotation_path)

    assert gene_lengths == {"ENSG1": 100.0, "ENSG2": 250.0}
    assert metadata["length_strategy"] == "genomic_span_from_start_end"


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

    # Step-two artifacts for NB tests
    aligned_counts = pd.read_csv(output_dir / "aligned_counts.tsv", sep="\t", index_col=0)
    sample_scaling = pd.read_csv(output_dir / "sample_scaling.tsv", sep="\t", index_col=0)
    gene_lengths = pd.read_csv(output_dir / "gene_lengths_aligned.tsv", sep="\t")
    assert list(aligned_counts.columns) == ["ENSG1", "ENSG2", "ENSG3"]
    assert aligned_counts.loc["sample_a", "ENSG3"] == 0.0
    assert "S_j" in sample_scaling.columns
    assert list(gene_lengths.columns) == ["ensg_id", "length_kb", "has_length"]


def test_preprocess_counts_emits_aligned_counts_tpm_scaling_and_gene_lengths(
    tmp_path: Path,
) -> None:
    """Verify preprocess_counts produces aligned_counts, sample_scaling, gene_lengths_aligned."""
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

    result = preprocess_counts(
        counts_path=counts_path,
        annotation_path=annotation_path,
        bulkformer_gene_info_path=gene_info_path,
        counts_orientation="genes-by-samples",
    )

    assert list(result.aligned_counts.columns) == ["ENSG1", "ENSG2", "ENSG3"]
    assert result.aligned_counts.loc["sample_a", "ENSG3"] == 0.0
    assert result.aligned_counts.loc["sample_a", "ENSG1"] == 105.0  # collapsed ENSG1.1 + ENSG1.2
    assert result.aligned_tpm.shape == result.aligned_counts.shape
    assert list(result.sample_scaling.columns) == ["S_j"]
    assert result.sample_scaling.index.tolist() == ["sample_a", "sample_b"]
    assert list(result.gene_lengths_aligned.columns) == ["ensg_id", "length_kb", "has_length"]
    assert result.gene_lengths_aligned["ensg_id"].tolist() == ["ENSG1", "ENSG2", "ENSG3"]
    assert result.gene_lengths_aligned.loc[result.gene_lengths_aligned["ensg_id"] == "ENSG3", "has_length"].iloc[0] == 0

    output_dir.mkdir(parents=True)
    write_preprocess_outputs(result, output_dir)
    assert (output_dir / "aligned_counts.tsv").exists()
    assert (output_dir / "aligned_tpm.tsv").exists()
    assert (output_dir / "gene_lengths_aligned.tsv").exists()
    assert (output_dir / "sample_scaling.tsv").exists()


def test_align_counts_to_bulkformer_uses_zero_for_missing_genes() -> None:
    """Counts must use 0 for missing genes, not -10 (NB test requirement)."""
    counts = pd.DataFrame(
        [[10.0, 20.0]],
        index=["s1"],
        columns=["ENSG1", "ENSG2"],
    )
    aligned = align_counts_to_bulkformer(counts, ["ENSG1", "ENSG2", "ENSG3"])
    assert aligned.loc["s1", "ENSG3"] == 0.0
    assert aligned.loc["s1", "ENSG1"] == 10.0


def test_compute_sample_scaling_matches_tpm_formula() -> None:
    """S_j = sum_h K_{jh}/L^{kb}_h should match TPM denominator."""
    counts = pd.DataFrame(
        [[100.0, 50.0], [25.0, 75.0]],
        index=["s1", "s2"],
        columns=["ENSG1", "ENSG2"],
    )
    gene_lengths_kb = {"ENSG1": 1.0, "ENSG2": 2.0}
    scaling = compute_sample_scaling(counts, gene_lengths_kb)
    assert scaling.loc["s1", "S_j"] == 100.0 / 1.0 + 50.0 / 2.0  # 100 + 25 = 125
    assert scaling.loc["s2", "S_j"] == 25.0 / 1.0 + 75.0 / 2.0  # 25 + 37.5 = 62.5


def test_load_counts_matrix_autogenerates_sample_ids_for_sample_by_gene_demo_tables(
    tmp_path: Path,
) -> None:
    counts_path = tmp_path / "demo_counts.csv"
    counts_path.write_text(
        "ENSG1,ENSG2,ENSG3\n"
        "10,0,5\n"
        "3,8,1\n",
        encoding="utf-8",
    )

    counts, metadata = load_counts_matrix(
        counts_path,
        orientation="samples-by-genes",
    )

    assert list(counts.index) == ["sample_1", "sample_2"]
    assert list(counts.columns) == ["ENSG1", "ENSG2", "ENSG3"]
    assert counts.index.name == "sample_id"
    assert counts.loc["sample_1", "ENSG1"] == 10.0
    assert metadata["orientation"] == "samples-by-genes"
