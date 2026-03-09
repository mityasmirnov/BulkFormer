"""Smoke tests for the BulkFormer diagnostics CLI scaffold."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the CLI entrypoint from the repository root."""
    return subprocess.run(
        [sys.executable, "-m", "bulkformer_dx.cli", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_top_level_help_lists_scaffolded_commands() -> None:
    result = run_cli("--help")

    assert result.returncode == 0
    assert "preprocess" in result.stdout
    assert "anomaly" in result.stdout
    assert "tissue" in result.stdout
    assert "proteomics" in result.stdout


def test_anomaly_help_lists_scaffolded_subcommands() -> None:
    result = run_cli("anomaly", "--help")

    assert result.returncode == 0
    assert "score" in result.stdout
    assert "head" in result.stdout
    assert "calibrate" in result.stdout


def test_tissue_help_lists_train_predict_and_artifact_options() -> None:
    result = run_cli("tissue", "--help")

    assert result.returncode == 0
    assert "train" in result.stdout
    assert "predict" in result.stdout
    assert "--labels" in result.stdout
    assert "--artifact-path" in result.stdout
