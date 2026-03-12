# BulkFormer DX Reports

Reports document preprocessing, anomaly scoring, calibration, and benchmark metrics for demo and clinical pipelines.

## Report Files

| Report | Description |
| --- | --- |
| `bulkformer_dx_demo_report.md` | Demo pipeline (37M): preprocess, spike inject, anomaly score, calibrate, benchmark metrics |
| `bulkformer_dx_clinical_report.md` | Clinical pipeline (37M): preprocess, anomaly score, calibrate, embeddings |
| `bulkformer_dx_clinical_report_147M.md` | Clinical pipeline (147M): anomaly score, calibrate (alpha=0.01), embeddings |
| `HARMONIZED_REPORT_TEMPLATE.md` | Canonical section order and tables for demo/clinical/method-comparison reporting |

## Benchmarking Fields

When spike recovery is run (`scripts/spike_recovery_metrics.py`), reports include:

- **AUROC**: Area under ROC curve for detecting injected outliers
- **AUPRC**: Area under precision-recall curve
- **Recall at FDR 0.05 / 0.10**: Fraction of injected genes recovered at given FDR
- **Precision at top 100**: Fraction of top-100 ranked genes that are injected
- **Spike recovery table**: `spike_recovery.tsv` with base vs spiked ranks, scores, significance

## Generating Reports

```bash
# Demo (full pipeline)
bash scripts/run_demo_37M.sh

# Clinical 37M
bash scripts/run_clinical_37M.sh

# Clinical 147M (requires preprocess from 37M)
ALPHA=0.01 bash scripts/run_clinical_147M.sh

# Report-only (after pipeline has run)
PYTHONPATH=. python scripts/generate_demo_report.py
PYTHONPATH=. python scripts/generate_clinical_report.py --variant both
```

## Figures

- `figures/preprocess_*.png` – Preprocess QC
- `figures/calibration_*.png` – Calibration diagnostics
- `figures/spike_*.png` – Spike recovery (PR curve, p-value QQ, histograms)
- `figures/calibration_outliers_per_sample_*.tsv` – Outliers per sample tables

## Notebooks

- `notebooks/bulkformer_dx_demo_37M.ipynb` – Reproduces demo pipeline
- `notebooks/bulkformer_dx_clinical_37M_147M.ipynb` – Reproduces clinical pipeline

Run notebooks from repo root with `PYTHONPATH=.` so `bulkformer_dx` is importable.

## Harmonized reporting format

Use `reports/HARMONIZED_REPORT_TEMPLATE.md` as the default structure for new reports so metrics, methods, and caveats are comparable across runs.
