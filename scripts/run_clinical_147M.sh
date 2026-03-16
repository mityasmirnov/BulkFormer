#!/usr/bin/env bash
# Run full clinical RNA-seq pipeline with BulkFormer 147M.
# Preprocess is shared; anomaly/calibrate/embeddings use 147M.
# Use --alpha 0.01 for stricter outlier calls (fewer false positives).

set -euo pipefail

VARIANT="147M"
ALPHA="${ALPHA:-0.01}"  # Stricter default to reduce outlier inflation
PREPROCESS_DIR="runs/clinical_preprocess_37M"  # Shared preprocess output

echo "=== Clinical 147M pipeline (alpha=${ALPHA}) ==="

# 1. Anomaly score (preprocess must exist)
python -m bulkformer_dx.cli anomaly score \
  --input "${PREPROCESS_DIR}/aligned_log1p_tpm.tsv" \
  --valid-gene-mask "${PREPROCESS_DIR}/valid_gene_mask.tsv" \
  --output-dir "runs/clinical_anomaly_score_${VARIANT}" \
  --variant "${VARIANT}" \
  --device cuda \
  --batch-size 4 \
  --mask-schedule deterministic --K-target 5 --mask-prob 0.10

# 2. Calibrate (stricter alpha)
python -m bulkformer_dx.cli anomaly calibrate \
  --scores "runs/clinical_anomaly_score_${VARIANT}" \
  --output-dir "runs/clinical_anomaly_calibrated_${VARIANT}" \
  --alpha "${ALPHA}"

# 3. Embeddings
python -m bulkformer_dx.cli embeddings extract \
  --input "${PREPROCESS_DIR}/aligned_log1p_tpm.tsv" \
  --valid-gene-mask "${PREPROCESS_DIR}/valid_gene_mask.tsv" \
  --output-dir "runs/clinical_embeddings_${VARIANT}" \
  --variant "${VARIANT}" \
  --device cuda \
  --batch-size 4

# 4. Merge annotation
python scripts/merge_clinical_annotation.py --variant "${VARIANT}"

# 5. Generate report
PYTHONPATH=. python scripts/generate_clinical_report.py --variant 147M

echo "=== Done. Outputs in runs/clinical_*_${VARIANT}/ Report: reports/bulkformer_dx_clinical_report_147M.md ==="
