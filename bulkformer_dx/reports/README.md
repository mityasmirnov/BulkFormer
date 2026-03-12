# BulkFormer DX Reports

Reports document preprocessing, benchmarking, and anomaly detection steps.

## Report Naming

| Report | Step | Description |
|--------|------|-------------|
| `step_two_preprocess_counts.md` | 2 | Preprocess QC: aligned counts, TPM, gene lengths, sample scaling, sanity table |

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
