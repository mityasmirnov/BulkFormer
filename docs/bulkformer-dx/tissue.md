# Tissue Workflow

The tissue workflow trains a `RandomForestClassifier` on frozen BulkFormer sample
embeddings and then reuses the serialized sklearn bundle for inference.

## Inputs

- A BulkFormer-aligned sample-by-gene matrix such as `aligned_log1p_tpm.tsv`
- A tissue label table for training with columns `sample_id` and `tissue_label`
- Optionally, `valid_gene_mask.tsv` to restrict embeddings to genes observed in
  the cohort before aggregation

## Train

```bash
python -m bulkformer_dx.cli tissue train \
  --input path/to/aligned_log1p_tpm.tsv \
  --labels path/to/tissue_labels.tsv \
  --valid-gene-mask path/to/valid_gene_mask.tsv \
  --output-dir tissue_train \
  --variant 37M \
  --aggregation mean \
  --pca-components 128
```

Training writes:

- `tissue_model.joblib`: sklearn pipeline bundle for inference
- `training_summary.json`: sample counts, classes, PCA setting, and train accuracy

## Predict

```bash
python -m bulkformer_dx.cli tissue predict \
  --input path/to/aligned_log1p_tpm.tsv \
  --artifact-path tissue_train/tissue_model.joblib \
  --valid-gene-mask path/to/valid_gene_mask.tsv \
  --output-dir tissue_predict
```

Prediction writes:

- `tissue_predictions.tsv`: predicted tissue label plus per-class probabilities
- `prediction_summary.json`: compact metadata for the prediction run

## Notes

- The BulkFormer backbone stays frozen; only the sklearn classifier is fit.
- `--pca-components` is optional. Omit it to train directly on the aggregated
  sample embeddings.
- The artifact bundle stores the training aggregation mode, selected gene set,
  and resolved BulkFormer asset contract. Prediction reuses that contract and
  rejects conflicting `--variant` or asset-path overrides.
- If you pass `--valid-gene-mask` during prediction, it must resolve to the same
  ordered gene set that was used for training.
