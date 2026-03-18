# Preprocess Workflow

`bulkformer_dx preprocess` converts raw counts into BulkFormer-aligned `log1p(TPM)` matrices. In clinical pipelines, it also performs OUTRIDER-like gene expression filtering to ensure anomaly detection only runs on reliably quantified genes.

## Inputs

- `--counts`: raw counts table in CSV or TSV format.
- `--annotation`: gene annotation table (optional if `--gtf` or `--exon-lengths-tsv` provided).
- `--gtf`: Path to a GTF/GFF file for exon-union length calculation (preferred for accuracy).
- `--exon-lengths-tsv`: Path to a TSV with precomputed exon-union lengths (columns: `gene_id`, `basepairs`).
- `--output-dir`: directory for the generated matrices and report.

### Expression Filtering (Clinical)

- `--expression-filter {none, outrider_like}`: Default is `outrider_like`.
- `--fpkm-cutoff`: FPKM threshold (default 1.0).
- `--fpkm-percentile`: Percentile for across-cohort FPKM filter (default 0.95).
- `--min-counts-fraction`: Minimum fraction of samples with non-zero counts (default 0.01 = 1 per 100).
- `--minCounts-only`: Skip FPKM filter, only apply min-counts check.

## What It Does

1. Normalizes Ensembl IDs by stripping version suffixes such as `ENSG00000123456.7 -> ENSG00000123456`.
2. Loads gene lengths from GTF (exon-union), TSV, or annotation table.
3. Applies the notebook TPM formula:
   `rate = counts / (gene_length_bp / 1000)`, `TPM = rate / sum(rate) * 1e6`, `log1p(TPM) = ln(TPM + 1)`.
4. Performs OUTRIDER-like filtering:
   - A gene passes if its `pth` percentile FPKM > `fpkmCutoff`.
   - A gene must also have non-zero counts in at least `ceil(n_samples * fraction)` samples.
5. Aligns the matrix to `data/bulkformer_gene_info.csv` order.
6. Exports a validity mask, expression filter diagnostics, and a JSON report.

## Outputs

- `tpm.tsv`: sample-by-gene TPM matrix.
- `log1p_tpm.tsv`: sample-by-gene `log1p(TPM)` matrix before alignment.
- `aligned_log1p_tpm.tsv`: BulkFormer-aligned `log1p(TPM)` with `-10` fill.
- `expression_filter.tsv`: Per-gene filtering diagnostics (FPKM quantiles, pass/fail flags).
- `valid_gene_mask.tsv`: Includes `is_valid` (present) and `passed_expression_filter` flags. Anomaly scoring only uses genes where both are true (`is_scored_gene`).
- `preprocess_report.json`: Detailed counts of genes passing/failing each filter (`genes_passed_expression_filter`, `genes_filtered_by_min_counts`, `genes_filtered_by_fpkm`). When expression filtering is used, `anomaly_run.json` `valid_gene_count` equals `genes_passed_expression_filter`.

## Gene Lengths from TxDb (Clinical)

When you have a TxDb database (e.g. `data/clinical_rnaseq/txdb.db`), export exon-union lengths to a TSV first:

```bash
Rscript scripts/export_txdb_exon_lengths.R data/clinical_rnaseq/txdb.db data/clinical_rnaseq/exon_lengths.tsv
```

Requires R with `GenomicFeatures`: `BiocManager::install("GenomicFeatures")`. Then run preprocess with:

```bash
python -m bulkformer_dx preprocess \
  --counts data/clinical_rnaseq/raw_counts.tsv \
  --exon-lengths-tsv data/clinical_rnaseq/exon_lengths.tsv \
  --expression-filter outrider_like \
  --output-dir preprocess_out
```

## Example

```bash
python -m bulkformer_dx preprocess \
  --counts counts.tsv \
  --gtf genes.gtf \
  --expression-filter outrider_like \
  --output-dir preprocess_out
```
