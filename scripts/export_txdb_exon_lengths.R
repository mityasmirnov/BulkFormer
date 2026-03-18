#!/usr/bin/env Rscript
# Export exon-union lengths from a TxDb database to a TSV for bulkformer_dx preprocess.
#
# Usage:
#   Rscript scripts/export_txdb_exon_lengths.R [txdb_path] [output_path]
#
# Defaults:
#   txdb_path:  data/clinical_rnaseq/txdb.db
#   output_path: data/clinical_rnaseq/exon_lengths.tsv
#
# Requires: Bioconductor GenomicFeatures
#   if (!require("BiocManager", quietly = TRUE)) install.packages("BiocManager")
#   BiocManager::install("GenomicFeatures")
#
# Output columns: gene_id, basepairs
# Gene IDs are normalized (Ensembl without version) for compatibility with bulkformer_dx.

args <- commandArgs(trailingOnly = TRUE)
txdb_path <- if (length(args) >= 1) args[1] else "data/clinical_rnaseq/txdb.db"
output_path <- if (length(args) >= 2) args[2] else "data/clinical_rnaseq/exon_lengths.tsv"

if (!file.exists(txdb_path)) {
  stop("TxDb file not found: ", txdb_path, "\n  Place txdb.db in data/clinical_rnaseq/ or pass path as first argument.")
}

suppressPackageStartupMessages({
  if (!requireNamespace("GenomicFeatures", quietly = TRUE)) {
    stop("GenomicFeatures required. Install with: BiocManager::install(\"GenomicFeatures\")")
  }
  library(GenomicFeatures)
})

txdb <- loadDb(txdb_path)
exons_by_gene <- exonsBy(txdb, by = "gene")
gene_lengths <- sum(width(reduce(exons_by_gene)))

out_dir <- dirname(output_path)
if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

df <- data.frame(
  gene_id = names(gene_lengths),
  basepairs = as.integer(gene_lengths),
  stringsAsFactors = FALSE
)

# Strip Ensembl version suffix if present (e.g. ENSG00000123456.12 -> ENSG00000123456)
df$gene_id <- sub("\\.[0-9]+$", "", df$gene_id)

write.table(df, output_path, sep = "\t", row.names = FALSE, quote = FALSE)
message("Wrote ", nrow(df), " gene lengths to ", output_path)
