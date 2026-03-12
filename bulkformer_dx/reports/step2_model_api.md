# Step 2: BulkFormer Inference API

## Overview

The standardized BulkFormer inference API produces `ModelPredictionBundle` with:

- **y_hat**: Predicted mean expression (n_samples × n_genes) in the same space as input (log1p_tpm by default)
- **embedding**: Sample-level embeddings (n_samples × d) for kNN cohort selection
- **sigma_hat**: Optional; from MC variance when `mc_predict` runs with `mc_passes > 1`, or from sigma head (future)
- **mc_samples**: Optional; from `mc_predict()` when MC masking is used

## API Functions

### `predict_mean(bundle) -> ModelPredictionBundle`

Runs BulkFormer with `mask_prob=0` (no masking). Produces point predictions and sample embeddings.

```python
from bulkformer_dx.model.bulkformer import predict_mean, bundle_from_paths

bundle = bundle_from_paths(Path("preprocess_output/"))
pred = predict_mean(bundle, variant="37M", device="cpu")
# pred.y_hat, pred.embedding
```

### `predict(bundle, method_config) -> ModelPredictionBundle`

Unified entrypoint driven by `MethodConfig`. Dispatches to `predict_mean` when `mc_passes=0`, or `mc_predict` when `mc_passes > 0`. Uses `method_config.seed` for deterministic MC masking.

```python
from bulkformer_dx.io.schemas import MethodConfig
from bulkformer_dx.model.bulkformer import predict, bundle_from_paths

bundle = bundle_from_paths(Path("preprocess_output/"))
config = MethodConfig(method_id="mc", space="log1p_tpm", mc_passes=16, seed=0)
pred = predict(bundle, config, device="cpu")
# pred.y_hat, pred.embedding, pred.sigma_hat, pred.mc_samples
```

### `mc_predict(bundle, mc_passes=16, mask_prob=0.15, seed=0) -> (ModelPredictionBundle, mc_samples)`

Runs multiple MC masking passes with deterministic seeding. Returns mean over passes as `y_hat`, per-sample-per-gene `sigma_hat` from MC variance when `mc_passes > 1`, and full `mc_samples` array (n_mc × n_samples × n_genes).

### `predict_sigma_head(bundle) -> np.ndarray | None`

Returns `None` for current checkpoints; reserved for future sigma-head support.

## CLI

```bash
python -m bulkformer_dx predict --input-dir preprocess_output/ --output-dir predict_output/
python -m bulkformer_dx predict --input-dir preprocess_output/ --output-dir predict_output/ --mc-passes 16
```

With `--mc-passes > 1`, the CLI writes `sigma_hat.tsv` (per-sample-per-gene std from MC variance).

## Sanity Checks

- **Reconstruction**: On non-masked genes, `y_hat` should be close to `Y_obs` (MSE on order of training objective).
- **Embeddings**: Non-degenerate; PCA should show meaningful structure for cohort selection.
- **MC variance**: When `mc_passes > 1`, `sigma_hat` is populated from std across MC passes; usable with `uncertainty_source="mc_variance"` in scoring.

## Dependencies

- `AlignedExpressionBundle`, `MethodConfig` from `bulkformer_dx.io.schemas`
- `bundle_from_paths()` or `bundle_from_preprocess_result()` to build input from preprocess outputs
