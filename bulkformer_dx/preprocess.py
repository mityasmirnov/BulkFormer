"""Counts-to-TPM preprocessing and BulkFormer gene alignment."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_FILL_VALUE = -10.0
DEFAULT_MISSING_GENE_LENGTH_BP = 1000.0
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BULKFORMER_GENE_INFO = REPO_ROOT / "data" / "bulkformer_gene_info.csv"
COUNT_ORIENTATIONS = ("genes-by-samples", "samples-by-genes")
GENE_ID_COLUMN_CANDIDATES = (
    "ensg_id",
    "gene_id",
    "ensembl_gene_id",
    "gene",
    "gene stable id",
)
GENE_LENGTH_COLUMN_CANDIDATES = (
    "gene_length",
    "length",
    "gene_length_bp",
    "gene_length_base_pairs",
)
GENOMIC_START_COLUMN_CANDIDATES = ("start", "gene_start", "tx_start")
GENOMIC_END_COLUMN_CANDIDATES = ("end", "gene_end", "tx_end")


@dataclass(slots=True)
class PreprocessResult:
    """In-memory outputs from the preprocessing pipeline."""

    counts: pd.DataFrame
    tpm: pd.DataFrame
    log1p_tpm: pd.DataFrame
    aligned_log1p_tpm: pd.DataFrame
    valid_gene_mask: pd.DataFrame
    report: dict[str, Any]


def normalize_ensembl_id(gene_id: object) -> str | None:
    """Normalize an Ensembl gene identifier by trimming whitespace and versions."""
    if gene_id is None or pd.isna(gene_id):
        return None

    normalized = str(gene_id).strip()
    if not normalized or normalized.lower() == "nan":
        return None

    return normalized.split(".", 1)[0]


def _read_table(path: Path) -> pd.DataFrame:
    """Read a delimited table using the filename extension as the separator hint."""
    suffix = path.suffix.lower()
    separator = "\t" if suffix in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=separator)


def _resolve_column(columns: pd.Index, candidates: tuple[str, ...], label: str) -> str:
    """Resolve a column name from a list of case-insensitive candidates."""
    normalized = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError(
        f"Could not infer the {label} column. "
        f"Available columns: {', '.join(map(str, columns))}"
    )


def _collapse_duplicate_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Sum duplicate gene columns created after Ensembl version normalization."""
    duplicated = int(frame.columns.duplicated().sum())
    if duplicated == 0:
        return frame, 0

    collapsed = frame.T.groupby(level=0, sort=False).sum().T
    return collapsed, duplicated


def load_counts_matrix(
    counts_path: Path,
    *,
    orientation: str = "genes-by-samples",
    gene_column: str | None = None,
    sample_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    """Load raw counts into a sample-by-gene matrix."""
    counts_table = _read_table(counts_path)
    if counts_table.empty:
        raise ValueError("The counts table is empty.")

    if orientation not in COUNT_ORIENTATIONS:
        raise ValueError(
            f"Unsupported counts orientation {orientation!r}. "
            f"Expected one of {COUNT_ORIENTATIONS}."
        )

    if orientation == "genes-by-samples":
        resolved_gene_column = gene_column or str(counts_table.columns[0])
        if resolved_gene_column not in counts_table.columns:
            raise ValueError(
                f"Gene column {resolved_gene_column!r} was not found in the counts table."
            )

        gene_ids = [normalize_ensembl_id(value) for value in counts_table[resolved_gene_column]]
        numeric_counts = counts_table.drop(columns=[resolved_gene_column]).apply(
            pd.to_numeric,
            errors="coerce",
        )
        numeric_counts = numeric_counts.fillna(0.0)
        sample_by_gene = numeric_counts.T
        sample_by_gene.columns = gene_ids
        sample_by_gene.index.name = "sample_id"
    else:
        working_table = counts_table.copy()
        resolved_sample_column = sample_column
        normalized_columns = [normalize_ensembl_id(column) for column in working_table.columns]
        columns_are_all_ensembl = all(column is not None for column in normalized_columns)
        if resolved_sample_column is None and not columns_are_all_ensembl:
            if str(working_table.columns[0]).lower() not in {
                "ensg_id",
                "gene_id",
                "ensembl_gene_id",
                "gene",
            }:
                resolved_sample_column = str(working_table.columns[0])

        if resolved_sample_column is not None:
            if resolved_sample_column not in working_table.columns:
                raise ValueError(
                    f"Sample column {resolved_sample_column!r} was not found in the counts table."
                )
            working_table = working_table.set_index(resolved_sample_column)
        sample_by_gene = working_table.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        sample_by_gene.columns = [normalize_ensembl_id(column) for column in sample_by_gene.columns]
        if resolved_sample_column is None and columns_are_all_ensembl:
            sample_by_gene.index = [f"sample_{index + 1}" for index in range(len(sample_by_gene))]
        sample_by_gene.index.name = "sample_id"

    sample_by_gene = sample_by_gene.loc[:, [column is not None for column in sample_by_gene.columns]]
    sample_by_gene, collapsed_gene_columns = _collapse_duplicate_columns(sample_by_gene)
    sample_by_gene = sample_by_gene.astype(float)

    metadata = {
        "samples": int(sample_by_gene.shape[0]),
        "genes": int(sample_by_gene.shape[1]),
        "collapsed_gene_columns": collapsed_gene_columns,
        "orientation": orientation,
    }
    return sample_by_gene, metadata


def load_gene_lengths(
    annotation_path: Path,
    *,
    gene_id_column: str | None = None,
    length_column: str | None = None,
) -> tuple[dict[str, float], dict[str, int | str]]:
    """Load gene lengths keyed by normalized Ensembl gene identifiers."""
    annotation_table = _read_table(annotation_path)
    if annotation_table.empty:
        raise ValueError("The annotation table is empty.")

    resolved_gene_id_column = gene_id_column or _resolve_column(
        annotation_table.columns,
        GENE_ID_COLUMN_CANDIDATES,
        "gene ID",
    )
    gene_ids = annotation_table[resolved_gene_id_column].map(normalize_ensembl_id)
    length_strategy = "explicit_length_column"
    resolved_length_column = length_column
    if resolved_length_column is None:
        try:
            resolved_length_column = _resolve_column(
                annotation_table.columns,
                GENE_LENGTH_COLUMN_CANDIDATES,
                "gene length",
            )
        except ValueError:
            resolved_length_column = None

    if resolved_length_column is not None:
        lengths = pd.to_numeric(annotation_table[resolved_length_column], errors="coerce")
    else:
        resolved_start_column = _resolve_column(
            annotation_table.columns,
            GENOMIC_START_COLUMN_CANDIDATES,
            "gene start",
        )
        resolved_end_column = _resolve_column(
            annotation_table.columns,
            GENOMIC_END_COLUMN_CANDIDATES,
            "gene end",
        )
        starts = pd.to_numeric(annotation_table[resolved_start_column], errors="coerce")
        ends = pd.to_numeric(annotation_table[resolved_end_column], errors="coerce")
        lengths = ends - starts + 1.0
        length_strategy = "genomic_span_from_start_end"
    cleaned = pd.DataFrame({"gene_id": gene_ids, "gene_length": lengths}).dropna()
    cleaned = cleaned.loc[cleaned["gene_length"] > 0]
    if cleaned.empty:
        raise ValueError("No positive gene lengths were found in the annotation table.")

    duplicate_gene_ids = int(cleaned["gene_id"].duplicated().sum())
    gene_lengths = (
        cleaned.groupby("gene_id", sort=False)["gene_length"].max().astype(float).to_dict()
    )
    metadata = {
        "annotation_rows": int(len(annotation_table)),
        "usable_gene_lengths": int(len(gene_lengths)),
        "duplicate_gene_ids": duplicate_gene_ids,
        "length_strategy": length_strategy,
    }
    return gene_lengths, metadata


def counts_to_tpm(
    counts: pd.DataFrame,
    gene_lengths_bp: dict[str, float],
    *,
    missing_gene_length_bp: float = DEFAULT_MISSING_GENE_LENGTH_BP,
) -> tuple[pd.DataFrame, list[str]]:
    """Convert raw counts to TPM using gene-length normalization."""
    if missing_gene_length_bp <= 0:
        raise ValueError("Missing gene length fallback must be positive.")

    missing_lengths = [
        gene_id for gene_id in counts.columns if gene_id not in gene_lengths_bp
    ]
    gene_lengths_kb = np.array(
        [
            gene_lengths_bp.get(gene_id, missing_gene_length_bp) / 1000.0
            for gene_id in counts.columns
        ],
        dtype=float,
    )
    rate = counts.to_numpy(dtype=float) / gene_lengths_kb
    rate_sum = rate.sum(axis=1, keepdims=True)
    rate_sum[rate_sum == 0] = 1e-6
    tpm = rate / rate_sum * 1e6
    tpm_frame = pd.DataFrame(tpm, index=counts.index, columns=counts.columns)
    return tpm_frame, missing_lengths


def load_bulkformer_gene_panel(gene_info_path: Path = DEFAULT_BULKFORMER_GENE_INFO) -> list[str]:
    """Load the BulkFormer gene vocabulary in model order."""
    gene_info = _read_table(gene_info_path)
    if "ensg_id" not in gene_info.columns:
        raise ValueError("BulkFormer gene info must contain an 'ensg_id' column.")

    normalized_ids = [
        gene_id
        for gene_id in (normalize_ensembl_id(value) for value in gene_info["ensg_id"])
        if gene_id is not None
    ]
    return list(dict.fromkeys(normalized_ids))


def align_to_bulkformer_genes(
    expression: pd.DataFrame,
    gene_panel: list[str],
    *,
    fill_value: float = DEFAULT_FILL_VALUE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align an expression matrix to the BulkFormer vocabulary and fill missing genes."""
    aligned = expression.reindex(columns=gene_panel, fill_value=fill_value)
    present_genes = set(expression.columns)
    mask = pd.DataFrame(
        {
            "ensg_id": gene_panel,
            "is_valid": [int(gene_id in present_genes) for gene_id in gene_panel],
        }
    )
    mask["is_missing_fill"] = 1 - mask["is_valid"]
    return aligned, mask


def preprocess_counts(
    *,
    counts_path: Path,
    annotation_path: Path,
    bulkformer_gene_info_path: Path = DEFAULT_BULKFORMER_GENE_INFO,
    counts_orientation: str = "genes-by-samples",
    gene_column: str | None = None,
    sample_column: str | None = None,
    annotation_gene_column: str | None = None,
    annotation_length_column: str | None = None,
    fill_value: float = DEFAULT_FILL_VALUE,
    missing_gene_length_bp: float = DEFAULT_MISSING_GENE_LENGTH_BP,
) -> PreprocessResult:
    """Run counts loading, TPM normalization, and BulkFormer alignment."""
    counts, counts_metadata = load_counts_matrix(
        counts_path,
        orientation=counts_orientation,
        gene_column=gene_column,
        sample_column=sample_column,
    )
    gene_lengths, annotation_metadata = load_gene_lengths(
        annotation_path,
        gene_id_column=annotation_gene_column,
        length_column=annotation_length_column,
    )
    tpm, genes_missing_lengths = counts_to_tpm(
        counts,
        gene_lengths,
        missing_gene_length_bp=missing_gene_length_bp,
    )
    log1p_tpm = np.log1p(tpm)
    gene_panel = load_bulkformer_gene_panel(bulkformer_gene_info_path)
    aligned_log1p_tpm, valid_gene_mask = align_to_bulkformer_genes(
        log1p_tpm,
        gene_panel,
        fill_value=fill_value,
    )

    valid_gene_count = int(valid_gene_mask["is_valid"].sum())
    report = {
        "counts_path": str(counts_path),
        "annotation_path": str(annotation_path),
        "bulkformer_gene_info_path": str(bulkformer_gene_info_path),
        "samples": counts_metadata["samples"],
        "input_genes": counts_metadata["genes"],
        "collapsed_input_gene_columns": counts_metadata["collapsed_gene_columns"],
        "counts_orientation": counts_metadata["orientation"],
        "annotation_rows": annotation_metadata["annotation_rows"],
        "usable_annotation_gene_lengths": annotation_metadata["usable_gene_lengths"],
        "duplicate_annotation_gene_ids": annotation_metadata["duplicate_gene_ids"],
        "annotation_length_strategy": annotation_metadata["length_strategy"],
        "genes_missing_annotation_length": len(genes_missing_lengths),
        "genes_missing_annotation_length_examples": genes_missing_lengths[:10],
        "bulkformer_gene_count": len(gene_panel),
        "bulkformer_valid_gene_count": valid_gene_count,
        "bulkformer_missing_gene_count": int(len(gene_panel) - valid_gene_count),
        "bulkformer_valid_gene_fraction": valid_gene_count / len(gene_panel),
        "fill_value": fill_value,
        "missing_gene_length_bp": missing_gene_length_bp,
    }

    return PreprocessResult(
        counts=counts,
        tpm=tpm,
        log1p_tpm=log1p_tpm,
        aligned_log1p_tpm=aligned_log1p_tpm,
        valid_gene_mask=valid_gene_mask,
        report=report,
    )


def write_preprocess_outputs(result: PreprocessResult, output_dir: Path) -> None:
    """Persist preprocessing outputs to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    result.tpm.to_csv(output_dir / "tpm.tsv", sep="\t")
    result.log1p_tpm.to_csv(output_dir / "log1p_tpm.tsv", sep="\t")
    result.aligned_log1p_tpm.to_csv(output_dir / "aligned_log1p_tpm.tsv", sep="\t")
    result.valid_gene_mask.to_csv(output_dir / "valid_gene_mask.tsv", sep="\t", index=False)

    with (output_dir / "preprocess_report.json").open("w", encoding="utf-8") as handle:
        json.dump(result.report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def register_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the preprocessing command group."""
    parser = subparsers.add_parser(
        "preprocess",
        help="Convert raw counts into BulkFormer-aligned log1p(TPM).",
        description=(
            "Load raw counts, normalize them to TPM and log1p(TPM), align them "
            "to the BulkFormer gene vocabulary, and export a validity mask plus "
            "a preprocessing report."
        ),
    )
    parser.add_argument("--counts", required=True, help="Path to the raw counts table.")
    parser.add_argument("--annotation", required=True, help="Path to the gene annotation table.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where preprocessed outputs should be written.",
    )
    parser.add_argument(
        "--bulkformer-gene-info",
        default=str(DEFAULT_BULKFORMER_GENE_INFO),
        help="Path to the BulkFormer gene info table. Defaults to data/bulkformer_gene_info.csv.",
    )
    parser.add_argument(
        "--counts-orientation",
        choices=COUNT_ORIENTATIONS,
        default="genes-by-samples",
        help="Whether the counts table is organized as genes-by-samples or samples-by-genes.",
    )
    parser.add_argument(
        "--gene-column",
        help="Gene ID column for genes-by-samples input. Defaults to the first column.",
    )
    parser.add_argument(
        "--sample-column",
        help="Sample ID column for samples-by-genes input. Defaults to the first column.",
    )
    parser.add_argument(
        "--annotation-gene-column",
        help="Gene ID column in the annotation table. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--annotation-length-column",
        help="Gene length column in the annotation table. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--fill-value",
        type=float,
        default=DEFAULT_FILL_VALUE,
        help="Value used for genes missing from the input matrix after BulkFormer alignment.",
    )
    parser.add_argument(
        "--missing-gene-length-bp",
        type=float,
        default=DEFAULT_MISSING_GENE_LENGTH_BP,
        help="Fallback gene length in base pairs for genes absent from the annotation.",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    """Execute the preprocessing workflow."""
    result = preprocess_counts(
        counts_path=Path(args.counts),
        annotation_path=Path(args.annotation),
        bulkformer_gene_info_path=Path(args.bulkformer_gene_info),
        counts_orientation=args.counts_orientation,
        gene_column=args.gene_column,
        sample_column=args.sample_column,
        annotation_gene_column=args.annotation_gene_column,
        annotation_length_column=args.annotation_length_column,
        fill_value=args.fill_value,
        missing_gene_length_bp=args.missing_gene_length_bp,
    )
    output_dir = Path(args.output_dir)
    write_preprocess_outputs(result, output_dir)

    print(f"Wrote preprocessing outputs to {output_dir}")
    print(
        "Samples: {samples} | input genes: {input_genes} | BulkFormer-valid genes: "
        "{valid}/{panel}".format(
            samples=result.report["samples"],
            input_genes=result.report["input_genes"],
            valid=result.report["bulkformer_valid_gene_count"],
            panel=result.report["bulkformer_gene_count"],
        )
    )
    return 0
