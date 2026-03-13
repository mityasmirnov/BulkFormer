"""IO helpers for TSV/parquet load/write and MethodConfig materialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from bulkformer_dx.io.schemas import MethodConfig


def load_tsv(path: Path, *, set_index: str | None = None) -> pd.DataFrame:
    """Load a TSV file into a DataFrame.

    Args:
        path: Path to TSV file.
        set_index: Optional column name to use as index.

    Returns:
        Loaded DataFrame.
    """
    path = Path(path)
    df = pd.read_csv(path, sep="\t")
    if set_index and set_index in df.columns:
        df = df.set_index(set_index)
    return df


def write_tsv(df: pd.DataFrame, path: Path) -> Path:
    """Write DataFrame to TSV.

    Args:
        df: DataFrame to write.
        path: Output path.

    Returns:
        Path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=True if df.index.name else False)
    return path


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a parquet file into a DataFrame.

    Args:
        path: Path to parquet file.

    Returns:
        Loaded DataFrame.

    Raises:
        ImportError: If pyarrow/fastparquet not available.
    """
    path = Path(path)
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Write DataFrame to parquet.

    Args:
        df: DataFrame to write.
        path: Output path.

    Returns:
        Path written.

    Raises:
        ImportError: If pyarrow/fastparquet not available.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_table(path: Path, *, set_index: str | None = None) -> pd.DataFrame:
    """Load TSV or parquet based on file extension.

    Args:
        path: Path to file.
        set_index: Optional column name to use as index (TSV only).

    Returns:
        Loaded DataFrame.
    """
    path = Path(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        return load_parquet(path)
    return load_tsv(path, set_index=set_index)


def write_table(df: pd.DataFrame, path: Path, *, prefer_parquet: bool = True) -> Path:
    """Write DataFrame to TSV or parquet based on path/availability.

    Uses parquet if path ends in .parquet/.pq or prefer_parquet and pyarrow available.
    Falls back to TSV otherwise.

    Args:
        df: DataFrame to write.
        path: Output path.
        prefer_parquet: If True, try parquet when path has no extension.

    Returns:
        Path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    use_parquet = path.suffix.lower() in (".parquet", ".pq") or (
        prefer_parquet and not path.suffix
    )
    if use_parquet:
        try:
            out = path if path.suffix else path.with_suffix(".parquet")
            return write_parquet(df, out)
        except ImportError:
            pass
    out = path.with_suffix(".tsv") if not path.suffix else path
    return write_tsv(df, out)


def load_method_config(path: Path) -> MethodConfig:
    """Load and validate MethodConfig from YAML or JSON.

    Args:
        path: Path to YAML or JSON file.

    Returns:
        Validated MethodConfig.

    Raises:
        ValueError: If required fields missing or invalid.
    """
    data = load_config_dict(path)
    return method_config_from_dict(data)


def method_config_from_dict(data: dict[str, Any]) -> MethodConfig:
    """Materialize MethodConfig from dict with validation.

    Args:
        data: Dict with method_id, space, and optional overrides.

    Returns:
        Validated MethodConfig.

    Raises:
        ValueError: If method_id or space missing or invalid.
    """
    if not isinstance(data, dict):
        raise ValueError("MethodConfig must be a dict")
    method_id = data.get("method_id")
    if not method_id or not isinstance(method_id, str):
        raise ValueError("method_id is required and must be a non-empty string")
    space = data.get("space", "log1p_tpm")
    if space not in ("log1p_tpm", "counts"):
        raise ValueError(f"space must be 'log1p_tpm' or 'counts', got {space!r}")

    cohort = data.get("cohort") or {}
    uncertainty = data.get("uncertainty") or {}
    distribution = data.get("distribution") or {}
    test = data.get("test") or {}
    mult = data.get("multiple_testing")
    mult_dict = mult if isinstance(mult, dict) else {}
    runtime = data.get("runtime") or {}

    return MethodConfig(
        method_id=str(method_id),
        space=str(space),
        cohort_mode=str(data.get("cohort_mode", cohort.get("mode", "global"))),
        knn_k=int(data.get("knn_k", cohort.get("knn_k", 50))),
        uncertainty_source=str(data.get("uncertainty_source", uncertainty.get("source", "cohort_sigma"))),
        distribution_family=str(data.get("distribution_family", distribution.get("family", "gaussian"))),
        test_type=str(data.get("test_type", test.get("type", "zscore_2s"))),
        multiple_testing=str(data.get("multiple_testing", mult_dict.get("correction", "BY"))),
        alpha=float(data.get("alpha", mult_dict.get("alpha", 0.05))),
        mc_passes=int(data.get("mc_passes", runtime.get("mc_passes", 16))),
        mask_rate=float(data.get("mask_rate", runtime.get("mask_rate", 0.15))),
        mask_schedule=str(data.get("mask_schedule", runtime.get("mask_schedule", "stochastic"))),
        K_target=int(data.get("K_target", runtime.get("K_target", 5))),
        seed=int(data.get("seed", runtime.get("seed", 0))),
        student_t_df=float(data.get("student_t_df", 5.0)),
    )


def load_config_dict(path: Path) -> dict[str, Any]:
    """Load YAML or JSON config into dict.

    Args:
        path: Path to YAML or JSON file.

    Returns:
        Parsed config dict.
    """
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
            out = yaml.safe_load(text)
            return dict(out) if out is not None else {}
        except ImportError as e:
            raise ImportError("PyYAML required for YAML config. pip install pyyaml") from e
    return json.loads(text)
