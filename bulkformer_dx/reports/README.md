# BulkFormer DX Reports

Reports document preprocessing, benchmarking, and anomaly detection steps.

## Report Naming

| Report | Step | Description |
|--------|------|-------------|
| `step0_schema.md` | 0 | Core schemas, artifact tree, IO helpers |
| `docs/plans/2026-03-12-bulkformer-dx-anomaly-design.md` | 0 | High-level design summary |
| `step2_model_api.md` | 2 | BulkFormer inference API: predict_mean, embeddings, MC samples |
| `step_two_preprocess_counts.md` | 2 | Preprocess QC: aligned counts, TPM, gene lengths, sample scaling, sanity table |
| `step_three_nb_outrider_test.md` | 3 | OUTRIDER-style NB test: expected-count mapping, dispersion, p-value formula, calibration wiring |
| `step_four_tabpfn_like_likelihood.md` | 4 | Pseudo-likelihood vs TabPFN, NLL vs residual, sigma source comparisons |
| `step_five_local_cohort.md` | 5 | Global vs kNN local cohort: embedding viz, significant-gene distributions |
| `step_one_benchmark_scaffold.md` | 1,6 | Benchmark scaffold, method configs, smoke test |
| `final_benchmark_report.md` | 6 | Leaderboard template, calibration table, reproduction |

## Generating Reports

```bash
# Step two: requires preprocess outputs in runs/demo_preprocess_37M
PYTHONPATH=. python scripts/generate_step_two_report.py
```

## Expected Artifacts (Step Two)

- `aligned_counts.tsv` – samples × BulkFormer panel, 0 for missing genes
- `aligned_tpm.tsv` – TPM aligned to panel
- `gene_lengths_aligned.tsv` – length_kb, has_length per gene
- `sample_scaling.tsv` – S_j per sample (for TPM↔counts mapping)
- `valid_gene_mask.tsv` – is_valid (NB tests use is_valid==1 only)

## Benchmark Artifacts

- `benchmark_results.parquet` (or `.tsv`) – per-(sample,gene,method) scores and p-values
- `benchmark_summary.json` – metrics per method, dataset spec
- `benchmark_figures/` – QC plots, PR curves, p-value histograms
