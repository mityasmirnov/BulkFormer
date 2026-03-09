# Examples

These lightweight configs document the expected input shapes and CLI arguments for the diagnostics
toolkit without committing large datasets.

- `rna_anomaly.yaml`: counts -> preprocess -> anomaly scoring/calibration.
- `proteomics_train.yaml`: frozen-backbone proteomics training.
- `proteomics_predict.yaml`: proteomics inference from a saved head artifact.
- `tissue_train.yaml`: tissue classifier training from labeled reference RNA.

Conventions:

- RNA matrices are sample-by-gene after preprocessing.
- Use Ensembl gene IDs without version suffixes where possible; the preprocessing workflow strips
  versions automatically.
- Protein tables are sample-by-protein with `sample_id` as the first column.
