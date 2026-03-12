#!/usr/bin/env bash
# Run full demo pipeline: preprocess, spike inject, anomaly score (base + spiked), calibrate, benchmark metrics, report.
# Uses conda env: bulkformer-cuda (Linux) or bulkformer-mps (macOS). Override with CONDA_ENV=myenv.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
CONDA_ENV="${CONDA_ENV:-bulkformer-cuda}"
PY="conda run -n $CONDA_ENV python"

echo "=== Demo 37M pipeline (conda env: $CONDA_ENV) ==="

# 1. Preprocess
$PY -m bulkformer_dx.cli preprocess \
  --counts data/demo_count_data.csv \
  --annotation data/gene_length_df.csv \
  --output-dir runs/demo_preprocess_37M \
  --counts-orientation samples-by-genes

# 2. Spike inject
PYTHONPATH=. $PY scripts/demo_spike_inject.py

# 3. Anomaly score (base)
$PY -m bulkformer_dx.cli anomaly score \
  --input runs/demo_preprocess_37M/aligned_log1p_tpm.tsv \
  --valid-gene-mask runs/demo_preprocess_37M/valid_gene_mask.tsv \
  --output-dir runs/demo_anomaly_score_37M \
  --variant 37M --device cuda --mc-passes 16 --mask-prob 0.15

# 4. Calibrate (base)
$PY -m bulkformer_dx.cli anomaly calibrate \
  --scores runs/demo_anomaly_score_37M \
  --output-dir runs/demo_anomaly_calibrated_37M \
  --alpha 0.05

# 5. Anomaly score (spiked)
$PY -m bulkformer_dx.cli anomaly score \
  --input runs/demo_spike_37M/aligned_log1p_tpm_spiked.tsv \
  --valid-gene-mask runs/demo_preprocess_37M/valid_gene_mask.tsv \
  --output-dir runs/demo_spike_anomaly_score_37M \
  --variant 37M --device cuda --mc-passes 16 --mask-prob 0.15

# 6. Calibrate (spiked)
$PY -m bulkformer_dx.cli anomaly calibrate \
  --scores runs/demo_spike_anomaly_score_37M \
  --output-dir runs/demo_spike_anomaly_calibrated_37M \
  --alpha 0.05

# 7. Spike recovery metrics
PYTHONPATH=. $PY scripts/spike_recovery_metrics.py

# 8. Generate report
PYTHONPATH=. $PY scripts/generate_demo_report.py

echo "=== Done. Report: reports/bulkformer_dx_demo_report.md ==="
