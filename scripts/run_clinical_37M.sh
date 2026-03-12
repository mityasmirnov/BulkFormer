#!/usr/bin/env bash
# Run clinical pipeline for 37M: preprocess, anomaly score, calibrate, embeddings, report.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Clinical 37M pipeline ==="

# 1. Preprocess
python -m bulkformer_dx.cli preprocess \
  --counts data/clinical_rnaseq/raw_counts.tsv \
  --annotation data/clinical_rnaseq/gene_annotation_v29.tsv \
  --output-dir runs/clinical_preprocess_37M \
  --counts-orientation genes-by-samples

# 2. Anomaly score
python -m bulkformer_dx.cli anomaly score \
  --input runs/clinical_preprocess_37M/aligned_log1p_tpm.tsv \
  --valid-gene-mask runs/clinical_preprocess_37M/valid_gene_mask.tsv \
  --output-dir runs/clinical_anomaly_score_37M \
  --variant 37M --device cuda --mc-passes 16 --mask-prob 0.15

# 3. Calibrate
python -m bulkformer_dx.cli anomaly calibrate \
  --scores runs/clinical_anomaly_score_37M \
  --output-dir runs/clinical_anomaly_calibrated_37M \
  --alpha 0.05

# 4. Embeddings
python -m bulkformer_dx.cli embeddings extract \
  --input runs/clinical_preprocess_37M/aligned_log1p_tpm.tsv \
  --valid-gene-mask runs/clinical_preprocess_37M/valid_gene_mask.tsv \
  --output-dir runs/clinical_embeddings_37M \
  --variant 37M --device cuda

# 5. Merge annotation
python scripts/merge_clinical_annotation.py --variant 37M

# 6. Generate report
PYTHONPATH=. python scripts/generate_clinical_report.py --variant 37M

echo "=== Done. Report: reports/bulkformer_dx_clinical_report.md ==="
