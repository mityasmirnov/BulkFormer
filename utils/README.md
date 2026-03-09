# BulkFormer Backbone Utilities

This directory contains the original pretrained BulkFormer implementation that the diagnostics
toolkit reuses directly rather than reimplementing.

## Files

- `BulkFormer.py`
  - defines the main `BulkFormer` model
  - accepts the graph structure plus gene embeddings at construction time
  - supports two main forward modes:
    - embedding mode (`output_expr=False`)
    - expression reconstruction mode (`output_expr=True`)

- `BulkFormer_block.py`
  - defines the hybrid graph + Performer block used inside the backbone
  - combines `GCNConv` with Performer attention layers

- `Rope.py`
  - defines the expression embedding logic and mask-token handling
  - preserves the `-10` mask token convention used throughout the repo

## Diagnostics Reuse

The `bulkformer_dx` package uses these modules through `bulkformer_dx/bulkformer_model.py`.
That wrapper is responsible for:

- locating checkpoints and supporting assets
- building the sparse graph from the repo data files
- loading and cleaning checkpoint state dicts
- exposing batch helpers for expression prediction, gene embeddings, and sample embeddings

If you are extending diagnostics workflows, prefer changing `bulkformer_dx/` first and treat the
files in `utils/` as the shared pretrained backbone implementation.
