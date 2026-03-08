"""BulkFormer model variant configuration helpers."""

from __future__ import annotations

from copy import deepcopy

DEFAULT_MODEL_VARIANT = "147M"

MODEL_VARIANT_CONFIGS = {
    "37M": {
        "dim": 128,
        "bins": 0,
        "gb_repeat": 1,
        "p_repeat": 1,
        "bin_head": 12,
        "full_head": 8,
        "gene_length": 20010,
    },
    "50M": {
        "dim": 256,
        "bins": 0,
        "gb_repeat": 1,
        "p_repeat": 2,
        "bin_head": 12,
        "full_head": 8,
        "gene_length": 20010,
    },
    "93M": {
        "dim": 512,
        "bins": 0,
        "gb_repeat": 1,
        "p_repeat": 6,
        "bin_head": 12,
        "full_head": 8,
        "gene_length": 20010,
    },
    "127M": {
        "dim": 640,
        "bins": 0,
        "gb_repeat": 1,
        "p_repeat": 8,
        "bin_head": 12,
        "full_head": 8,
        "gene_length": 20010,
    },
    "147M": {
        "dim": 640,
        "bins": 0,
        "gb_repeat": 1,
        "p_repeat": 12,
        "bin_head": 12,
        "full_head": 8,
        "gene_length": 20010,
    },
}


def normalize_model_variant(variant: str) -> str:
    """Normalize common BulkFormer variant strings like 37m or BulkFormer-37M."""
    normalized = variant.strip().upper().replace("BULKFORMER", "")
    normalized = normalized.replace("_", "").replace("-", "")
    if normalized.endswith("M") and normalized[:-1].isdigit():
        return normalized
    raise ValueError(
        f"Unsupported BulkFormer variant {variant!r}. "
        f"Expected one of: {', '.join(sorted(MODEL_VARIANT_CONFIGS))}."
    )


def get_model_params(variant: str = DEFAULT_MODEL_VARIANT) -> dict[str, int]:
    """Return a copy of the parameter dictionary for a named BulkFormer variant."""
    normalized_variant = normalize_model_variant(variant)
    try:
        return deepcopy(MODEL_VARIANT_CONFIGS[normalized_variant])
    except KeyError as exc:
        raise ValueError(
            f"Unsupported BulkFormer variant {variant!r}. "
            f"Expected one of: {', '.join(sorted(MODEL_VARIANT_CONFIGS))}."
        ) from exc


# Preserve the original import surface for legacy code.
model_params = get_model_params(DEFAULT_MODEL_VARIANT)

