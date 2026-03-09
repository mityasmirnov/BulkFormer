# Preprocess Workflow

`bulkformer_dx preprocess` converts raw counts into BulkFormer-aligned `log1p(TPM)` matrices.

## Inputs

- `--counts`: raw counts table in CSV or TSV format.
- `--annotation`: gene annotation table with Ensembl IDs and either a gene-length column or `start`/`end` coordinates.
- `--output-dir`: directory for the generated matrices and report.

The default input orientation is `genes-by-samples`, where the first column contains gene IDs and the remaining columns are sample counts. For already transposed matrices, use `--counts-orientation samples-by-genes`.

## What It Does

1. Normalizes Ensembl IDs by stripping version suffixes such as `ENSG00000123456.7 -> ENSG00000123456`.
2. Loads gene lengths from the annotation table and applies the notebook TPM formula:
   `rate = counts / (gene_length_bp / 1000)`, `TPM = rate / sum(rate) * 1e6`, `log1p(TPM) = ln(TPM + 1)`.
   If a direct gene-length column is absent, the workflow falls back to a genomic-span estimate
   derived from `end - start + 1`.
3. Aligns the matrix to `data/bulkformer_gene_info.csv` order and fills genes missing from the input with `-10`.
4. Exports a validity mask and a JSON preprocessing report.

## Outputs

- `tpm.tsv`: sample-by-gene TPM matrix.
- `log1p_tpm.tsv`: sample-by-gene `log1p(TPM)` matrix before BulkFormer alignment.
- `aligned_log1p_tpm.tsv`: BulkFormer-ordered `log1p(TPM)` matrix with `-10` fill for missing genes.
- `valid_gene_mask.tsv`: per-gene validity mask in BulkFormer order.
- `preprocess_report.json`: sample counts, matched gene counts, fallback gene-length usage, and alignment coverage.

## Example

```bash
python -m bulkformer_dx.cli preprocess \
  --counts counts.tsv \
  --annotation gene_annotation_v29.tsv \
  --output-dir preprocess_out
```
