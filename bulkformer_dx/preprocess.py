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
DEFAULT_FPKM_CUTOFF = 1.0
DEFAULT_FPKM_PERCENTILE = 0.95
DEFAULT_MIN_COUNTS_FRACTION = 0.01  # 1 per 100 samples
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
    "basepairs",
    "exon_basepairs",
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
    aligned_counts: pd.DataFrame
    aligned_tpm: pd.DataFrame
    gene_lengths_aligned: pd.DataFrame
    sample_scaling: pd.DataFrame
    valid_gene_mask: pd.DataFrame
    report: dict[str, Any]
    expression_filter_table: pd.DataFrame | None = None


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
    """Load raw counts into a sample-by-gene matrix.

    Supported input layouts:
    - `genes-by-samples` (default): first column is gene id, remaining columns
      are samples.
    - `samples-by-genes`: rows are samples and columns are genes; optionally
      with a dedicated sample id column.

    Returns a normalized matrix in canonical orientation `(n_samples, n_genes)`
    plus metadata describing inferred shape and any duplicated genes collapsed
    after Ensembl version stripping.
    """
    counts_table = _read_table(counts_path)
    if counts_table.empty:
        raise ValueError("The counts table is empty.")

    if orientation not in COUNT_ORIENTATIONS:
        raise ValueError(
            f"Unsupported counts orientation {orientation!r}. "
            f"Expected one of {COUNT_ORIENTATIONS}."
        )

    if orientation == "genes-by-samples":
        # Most external count matrices provide Ensembl IDs in the first column.
        # We normalize them and transpose so all downstream code can assume
        # a sample-by-gene contract.
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
        # For samples-by-genes input, detect whether the first column looks like
        # a sample identifier or a gene id column from accidental orientation
        # confusion and handle both safely.
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

    # Remove columns that failed Ensembl normalization and collapse duplicate
    # columns created after dropping version suffixes (e.g. ENSG...1/.2).
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
    """Load gene lengths keyed by normalized Ensembl gene identifiers.

    Lengths are resolved from an explicit length column when available; if not,
    genomic start/end coordinates are used to derive a positive span.
    """
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
    """Convert raw counts to TPM using gene-length normalization.

    Any gene without an annotation length falls back to
    `missing_gene_length_bp`, and its id is included in the returned
    `missing_lengths` list for reporting/QC.
    """
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


def compute_fpkm(
    counts: pd.DataFrame,
    gene_lengths_bp: dict[str, float],
    *,
    missing_gene_length_bp: float = DEFAULT_MISSING_GENE_LENGTH_BP,
) -> pd.DataFrame:
    """Compute FPKM = (counts / length_kb) / (library_size / 1e6)."""
    library_size_m = counts.sum(axis=1).to_numpy() / 1e6
    library_size_m[library_size_m == 0] = 1e-6

    gene_lengths_kb = np.array(
        [
            gene_lengths_bp.get(gene_id, missing_gene_length_bp) / 1000.0
            for gene_id in counts.columns
        ],
        dtype=float,
    )
    # FPKM = (counts / length_kb) / (library_size / 1e6)
    # counts: [samples x genes]
    # library_size_m: [samples]
    # gene_lengths_kb: [genes]
    fpkm = counts.to_numpy(dtype=float) / gene_lengths_kb / library_size_m[:, np.newaxis]
    return pd.DataFrame(fpkm, index=counts.index, columns=counts.columns)


def outrider_like_passed_filter(
    fpkm: pd.DataFrame,
    fpkm_cutoff: float = DEFAULT_FPKM_CUTOFF,
    percentile: float = DEFAULT_FPKM_PERCENTILE,
) -> pd.Series:
    """For each gene: passed = quantile(fpkm_gene, percentile) > fpkm_cutoff."""
    quantiles = fpkm.quantile(percentile, axis=0)
    return quantiles > fpkm_cutoff


def min_count_requirements(
    counts: pd.DataFrame,
    fraction: float = DEFAULT_MIN_COUNTS_FRACTION,
) -> pd.Series:
    """Enforce minimum non-zero counts across the cohort.

    Passed if:
    1. At least one sample has non-zero count.
    2. Non-zero count in at least ceil(n_samples * fraction) samples.
    """
    n_samples = len(counts)
    nonzero_mask = (counts > 0).sum(axis=0)
    min_nonzero = int(np.ceil(n_samples * fraction))
    # Standard OUTRIDER-like behavior has 1/100 default.
    # We use fraction instead of counts per sample to be robust.
    return (nonzero_mask >= 1) & (nonzero_mask >= min_nonzero)


def compute_exon_union_lengths(gtf_path: Path) -> dict[str, float]:
    """Parse GTF/GFF to compute exon-union length per gene."""
    import re
    exon_intervals: dict[str, list[tuple[int, int]]] = {}

    # Parse file for exon features
    _logger_fn = lambda msg: print(f"GTF Parsing: {msg}")
    _logger_fn(f"Reading GTF from {gtf_path}")

    # Simple GTF parser for exon intervals
    with gtf_path.open("r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9 or parts[2] != "exon":
                continue

            start, end = int(parts[3]), int(parts[4])
            attributes = parts[8]

            # Extract gene_id
            match = re.search(r'gene_id "([^"]+)"', attributes)
            if not match:
                match = re.search(r'gene_id=([^;]+)', attributes)
            
            if match:
                gene_id = normalize_ensembl_id(match.group(1))
                if gene_id:
                    if gene_id not in exon_intervals:
                        exon_intervals[gene_id] = []
                    exon_intervals[gene_id].append((start, end))

    # Union overlapping intervals per gene and sum
    gene_lengths: dict[str, float] = {}
    for gene_id, intervals in exon_intervals.items():
        intervals.sort()
        if not intervals:
            continue

        union_len = 0
        curr_start, curr_end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start <= curr_end:
                curr_end = max(curr_end, next_end)
            else:
                union_len += curr_end - curr_start + 1
                curr_start, curr_end = next_start, next_end
        union_len += curr_end - curr_start + 1
        gene_lengths[gene_id] = float(union_len)

    _logger_fn(f"Computed lengths for {len(gene_lengths)} genes from GTF.")
    return gene_lengths


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
    passed_filter_genes: set[str] | None = None,
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
    if passed_filter_genes is not None:
        mask["passed_expression_filter"] = [
            int(gene_id in passed_filter_genes) for gene_id in gene_panel
        ]
    else:
        mask["passed_expression_filter"] = 1  # Default to everything if no filter run

    mask["is_scored_gene"] = mask["is_valid"] & mask["passed_expression_filter"]
    mask["is_missing_fill"] = 1 - mask["is_valid"]
    return aligned, mask


def align_counts_to_bulkformer(
    counts: pd.DataFrame,
    gene_panel: list[str],
) -> pd.DataFrame:
    """Align counts to the BulkFormer gene panel. Missing genes get 0 (not -10)."""
    return counts.reindex(columns=gene_panel, fill_value=0.0)


def compute_sample_scaling(
    counts: pd.DataFrame,
    gene_lengths_kb: dict[str, float],
    *,
    valid_genes: set[str] | None = None,
    missing_gene_length_kb: float = DEFAULT_MISSING_GENE_LENGTH_BP / 1000.0,
) -> pd.DataFrame:
    """Compute S_j = sum_h K_{jh}/L^{kb}_h per sample for TPM↔counts mapping."""
    if valid_genes is None:
        valid_genes = set(counts.columns)
    lengths = np.array(
        [
            gene_lengths_kb.get(g, missing_gene_length_kb)
            for g in counts.columns
        ],
        dtype=float,
    )
    lengths[lengths <= 0] = missing_gene_length_kb
    rate = counts.to_numpy(dtype=float) / lengths
    s_j = rate.sum(axis=1)
    return pd.DataFrame(
        {"sample_id": counts.index, "S_j": s_j},
    ).set_index("sample_id")


def build_gene_lengths_aligned(
    gene_panel: list[str],
    annotation_lengths: dict[str, float],
    bulkformer_gene_info_path: Path = DEFAULT_BULKFORMER_GENE_INFO,
    *,
    missing_gene_length_bp: float = DEFAULT_MISSING_GENE_LENGTH_BP,
    length_source: str = "annotation",
) -> pd.DataFrame:
    """Build gene_lengths_aligned table for the BulkFormer panel."""
    gene_info = _read_table(bulkformer_gene_info_path)
    bulkformer_lengths = {}
    if "gene_length" in gene_info.columns and "ensg_id" in gene_info.columns:
        for _, row in gene_info.iterrows():
            gid = normalize_ensembl_id(row["ensg_id"])
            if gid and pd.notna(row.get("gene_length")):
                bulkformer_lengths[gid] = float(row["gene_length"])

    rows = []
    for gid in gene_panel:
        length_bp = annotation_lengths.get(gid) or bulkformer_lengths.get(gid)
        source = length_source if gid in annotation_lengths else "bulkformer_info"
        if length_bp is not None and length_bp > 0:
            length_kb = length_bp / 1000.0
            has_length = 1
        else:
            length_kb = missing_gene_length_bp / 1000.0
            has_length = 0
            source = "fallback_default"
        rows.append({
            "ensg_id": gid,
            "length_kb": length_kb,
            "has_length": has_length,
            "length_source": source,
        })
    return pd.DataFrame(rows)


def _apply_low_expression_filter(
    counts: pd.DataFrame,
    tpm: pd.DataFrame,
    *,
    min_count: float | None = None,
    min_tpm: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Filter out low-expression genes. Returns filtered counts, tpm, and genes_removed."""
    if min_count is None and min_tpm is None:
        return counts, tpm, 0
    mask = np.ones(counts.shape[1], dtype=bool)
    if min_count is not None:
        total_counts = counts.sum(axis=0)
        mask &= (total_counts >= min_count).to_numpy()
    if min_tpm is not None:
        median_tpm = tpm.median(axis=0)
        mask &= (median_tpm >= min_tpm).to_numpy()
    genes_removed = int((~mask).sum())
    if genes_removed > 0:
        counts = counts.loc[:, mask]
        tpm = tpm.loc[:, mask]
    return counts, tpm, genes_removed


def preprocess_counts(
    *,
    counts_path: Path,
    annotation_path: Path | None = None,
    bulkformer_gene_info_path: Path = DEFAULT_BULKFORMER_GENE_INFO,
    counts_orientation: str = "genes-by-samples",
    gene_column: str | None = None,
    sample_column: str | None = None,
    annotation_gene_column: str | None = None,
    annotation_length_column: str | None = None,
    fill_value: float = DEFAULT_FILL_VALUE,
    missing_gene_length_bp: float = DEFAULT_MISSING_GENE_LENGTH_BP,
    min_count: float | None = None,
    min_tpm: float | None = None,
    expression_filter: str = "none",
    fpkm_cutoff: float = DEFAULT_FPKM_CUTOFF,
    fpkm_percentile: float = DEFAULT_FPKM_PERCENTILE,
    min_counts_fraction: float = DEFAULT_MIN_COUNTS_FRACTION,
    min_counts_only: bool = False,
    gtf_path: Path | None = None,
    exon_lengths_tsv: Path | None = None,
) -> PreprocessResult:
    """Run counts loading, TPM normalization, and BulkFormer alignment."""
    counts, counts_metadata = load_counts_matrix(
        counts_path,
        orientation=counts_orientation,
        gene_column=gene_column,
        sample_column=sample_column,
    )

    # Determine gene lengths
    gene_lengths: dict[str, float] = {}
    length_source = "fallback_default"
    annotation_metadata = {
        "annotation_rows": 0,
        "usable_gene_lengths": 0,
        "duplicate_gene_ids": 0,
        "length_strategy": "none",
    }

    if exon_lengths_tsv:
        table = _read_table(exon_lengths_tsv)
        id_col = _resolve_column(table.columns, GENE_ID_COLUMN_CANDIDATES, "gene ID")
        len_col = _resolve_column(table.columns, GENE_LENGTH_COLUMN_CANDIDATES, "gene length")
        gene_lengths = {
            normalize_ensembl_id(row[id_col]): float(row[len_col])
            for _, row in table.iterrows()
            if normalize_ensembl_id(row[id_col])
        }
        length_source = "txdb_tsv"
        annotation_metadata["length_strategy"] = "exon_lengths_tsv"
        annotation_metadata["usable_gene_lengths"] = len(gene_lengths)
    elif gtf_path:
        gene_lengths = compute_exon_union_lengths(gtf_path)
        length_source = "gtf_exon_union"
        annotation_metadata["length_strategy"] = "gtf_parsing"
        annotation_metadata["usable_gene_lengths"] = len(gene_lengths)
    elif annotation_path:
        gene_lengths, annotation_metadata = load_gene_lengths(
            annotation_path,
            gene_id_column=annotation_gene_column,
            length_column=annotation_length_column,
        )
        length_source = "annotation_table"

    # Expression Filtering
    fpkm: pd.DataFrame | None = None
    passed_min_counts = min_count_requirements(counts, fraction=min_counts_fraction)
    passed_fpkm_filter = pd.Series(True, index=counts.columns)
    filter_info = {}

    if expression_filter == "outrider_like":
        fpkm = compute_fpkm(counts, gene_lengths, missing_gene_length_bp=missing_gene_length_bp)
        if not min_counts_only:
            passed_fpkm_filter = outrider_like_passed_filter(
                fpkm, fpkm_cutoff=fpkm_cutoff, percentile=fpkm_percentile
            )
        
        passed_final = passed_min_counts & passed_fpkm_filter
        passed_filter_genes = set(passed_final.index[passed_final])
        
        # Diagnostics (all rows are input genes, so is_present_in_input=1)
        diag_df = pd.DataFrame({
            "ensg_id": counts.columns,
            "is_present_in_input": 1,
            "passed_min_counts": passed_min_counts.astype(int),
            "passed_fpkm_filter": passed_fpkm_filter.astype(int),
            "passed_expression_filter": passed_final.astype(int),
        })
        if fpkm is not None:
            diag_df[f"fpkm_p{int(fpkm_percentile*100)}"] = fpkm.quantile(fpkm_percentile, axis=0).values
            diag_df["median_fpkm"] = fpkm.median(axis=0).values
        
        diag_df["length_bp_used"] = [
            gene_lengths.get(g, missing_gene_length_bp) for g in counts.columns
        ]
        diag_df["length_source"] = [
            length_source if g in gene_lengths else "fallback_default"
            for g in counts.columns
        ]
        filter_info["expression_filter_table"] = diag_df
    else:
        passed_filter_genes = None

    # Proceed with normal flow
    tpm, genes_missing_lengths = counts_to_tpm(
        counts,
        gene_lengths,
        missing_gene_length_bp=missing_gene_length_bp,
    )
    
    # Legacy low-expression filter
    counts, tpm, genes_filtered_legacy = _apply_low_expression_filter(
        counts, tpm, min_count=min_count, min_tpm=min_tpm
    )

    gene_lengths_kb = {
        g: gene_lengths.get(g, missing_gene_length_bp) / 1000.0
        for g in counts.columns
    }
    sample_scaling = compute_sample_scaling(
        counts,
        gene_lengths_kb,
        missing_gene_length_kb=missing_gene_length_bp / 1000.0,
    )
    log1p_tpm = np.log1p(tpm)
    gene_panel = load_bulkformer_gene_panel(bulkformer_gene_info_path)
    aligned_counts = align_counts_to_bulkformer(counts, gene_panel)
    aligned_tpm = tpm.reindex(columns=gene_panel, fill_value=0.0)
    
    aligned_log1p_tpm, valid_gene_mask = align_to_bulkformer_genes(
        log1p_tpm,
        gene_panel,
        fill_value=fill_value,
        passed_filter_genes=passed_filter_genes,
    )
    gene_lengths_aligned = build_gene_lengths_aligned(
        gene_panel,
        gene_lengths,
        bulkformer_gene_info_path,
        missing_gene_length_bp=missing_gene_length_bp,
        length_source=length_source,
    )

    valid_gene_count = int(valid_gene_mask["is_valid"].sum())
    scored_gene_count = int(valid_gene_mask["is_scored_gene"].sum())
    
    report = {
        "counts_path": str(counts_path),
        "bulkformer_gene_info_path": str(bulkformer_gene_info_path),
        "samples": counts_metadata["samples"],
        "input_genes": counts_metadata["genes"],
        "genes_after_low_expression_filter": int(counts.shape[1]),
        "genes_filtered_low_expression": genes_filtered_legacy,
        "genes_passed_expression_filter": scored_gene_count if expression_filter != "none" else valid_gene_count,
        "genes_filtered_by_min_counts": int((~passed_min_counts).sum()) if expression_filter == "outrider_like" else 0,
        "genes_filtered_by_fpkm": int((~passed_fpkm_filter).sum()) if expression_filter == "outrider_like" else 0,
        "min_nonzero_samples_threshold": int(np.ceil(counts_metadata["samples"] * min_counts_fraction)),
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
        "bulkformer_scored_gene_count": scored_gene_count,
        "fill_value": fill_value,
        "missing_gene_length_bp": missing_gene_length_bp,
        "expression_filter_mode": expression_filter,
    }
    if annotation_path:
        report["annotation_path"] = str(annotation_path)
    if gtf_path:
        report["gtf_path"] = str(gtf_path)
    if exon_lengths_tsv:
        report["exon_lengths_tsv"] = str(exon_lengths_tsv)
    return PreprocessResult(
        counts=counts,
        tpm=tpm,
        log1p_tpm=log1p_tpm,
        aligned_log1p_tpm=aligned_log1p_tpm,
        aligned_counts=aligned_counts,
        aligned_tpm=aligned_tpm,
        gene_lengths_aligned=gene_lengths_aligned,
        sample_scaling=sample_scaling,
        valid_gene_mask=valid_gene_mask,
        report=report,
        expression_filter_table=filter_info.get("expression_filter_table"),
    )


def write_preprocess_outputs(result: PreprocessResult, output_dir: Path) -> None:
    """Persist preprocessing outputs to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    result.tpm.to_csv(output_dir / "tpm.tsv", sep="\t")
    result.log1p_tpm.to_csv(output_dir / "log1p_tpm.tsv", sep="\t")
    result.aligned_log1p_tpm.to_csv(output_dir / "aligned_log1p_tpm.tsv", sep="\t")
    result.aligned_counts.to_csv(output_dir / "aligned_counts.tsv", sep="\t")
    result.aligned_tpm.to_csv(output_dir / "aligned_tpm.tsv", sep="\t")
    result.gene_lengths_aligned.to_csv(
        output_dir / "gene_lengths_aligned.tsv", sep="\t", index=False
    )
    result.sample_scaling.to_csv(output_dir / "sample_scaling.tsv", sep="\t")
    result.valid_gene_mask.to_csv(output_dir / "valid_gene_mask.tsv", sep="\t", index=False)

    if result.expression_filter_table is not None:
        result.expression_filter_table.to_csv(
            output_dir / "expression_filter.tsv", sep="\t", index=False
        )

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
    parser.add_argument("--annotation", help="Path to the gene annotation table.")
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
    parser.add_argument(
        "--min-count",
        type=float,
        default=None,
        metavar="N",
        help="Filter out genes with total count across samples < N (optional low-expression filter).",
    )
    parser.add_argument(
        "--min-tpm",
        type=float,
        default=None,
        metavar="T",
        help="Filter out genes with median TPM across samples < T (optional low-expression filter).",
    )
    # New arguments for OUTRIDER-like filtering
    parser.add_argument(
        "--expression-filter",
        choices=["none", "outrider_like"],
        default="outrider_like",
        help="Gene expression filter logic. Default 'outrider_like' for clinical use.",
    )
    parser.add_argument(
        "--fpkm-cutoff",
        type=float,
        default=DEFAULT_FPKM_CUTOFF,
        help="FPKM cutoff for the percentile-based filter (default 1.0).",
    )
    parser.add_argument(
        "--fpkm-percentile",
        type=float,
        default=DEFAULT_FPKM_PERCENTILE,
        help="Percentile for FPKM-based filtering (default 0.95).",
    )
    parser.add_argument(
        "--min-counts-fraction",
        type=float,
        default=DEFAULT_MIN_COUNTS_FRACTION,
        help="Min sample fraction with non-zero counts (default 0.01 = 1 per 100 samples).",
    )
    parser.add_argument(
        "--min-counts-only",
        action="store_true",
        help="If set, skip FPKM percentile filter; only drop genes violating min-counts requirements.",
    )
    parser.add_argument(
        "--gtf",
        help="Path to a GTF/GFF file for exon-union length calculation.",
    )
    parser.add_argument(
        "--exon-lengths-tsv",
        help="Path to a TSV with precomputed exon-union lengths (columns: gene_id, basepairs).",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    """Execute the preprocessing workflow."""
    result = preprocess_counts(
        counts_path=Path(args.counts),
        annotation_path=Path(args.annotation) if args.annotation else None,
        bulkformer_gene_info_path=Path(args.bulkformer_gene_info),
        counts_orientation=args.counts_orientation,
        gene_column=args.gene_column,
        sample_column=args.sample_column,
        annotation_gene_column=args.annotation_gene_column,
        annotation_length_column=args.annotation_length_column,
        fill_value=args.fill_value,
        missing_gene_length_bp=args.missing_gene_length_bp,
        min_count=getattr(args, "min_count", None),
        min_tpm=getattr(args, "min_tpm", None),
        expression_filter=args.expression_filter,
        fpkm_cutoff=args.fpkm_cutoff,
        fpkm_percentile=args.fpkm_percentile,
        min_counts_fraction=args.min_counts_fraction,
        min_counts_only=args.min_counts_only,
        gtf_path=Path(args.gtf) if args.gtf else None,
        exon_lengths_tsv=Path(args.exon_lengths_tsv) if args.exon_lengths_tsv else None,
    )
    output_dir = Path(args.output_dir)
    write_preprocess_outputs(result, output_dir)

    print(f"Wrote preprocessing outputs to {output_dir}")
    valid = result.report["bulkformer_valid_gene_count"]
    panel = result.report["bulkformer_gene_count"]
    print(f"BulkFormer-valid genes: {valid}/{panel}")
    if result.report.get("expression_filter_mode") != "none":
        scored = result.report.get("genes_passed_expression_filter", valid)
        print(f"Scored genes (expression filter applied): {scored}/{valid}")
    return 0
