#!/usr/bin/env python3
"""Merge sample annotation with cohort_scores, calibration_summary, and embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge sample annotation with pipeline outputs.")
    parser.add_argument(
        "--variant",
        default="37M",
        help="Model variant (37M or 147M). Defaults to 37M.",
    )
    args = parser.parse_args()
    variant = args.variant

    sample_ann_path = Path("data/clinical_rnaseq/sample_annotation.tsv")
    cohort_scores_path = Path(f"runs/clinical_anomaly_score_{variant}/cohort_scores.tsv")
    calibration_path = Path(f"runs/clinical_anomaly_calibrated_{variant}/calibration_summary.tsv")
    embeddings_path = Path(f"runs/clinical_embeddings_{variant}/sample_embeddings.tsv")
    output_dir = Path(f"runs/clinical_annotated_{variant}" if variant != "37M" else "runs/clinical_annotated")

    if not sample_ann_path.exists():
        print(f"Sample annotation not found: {sample_ann_path}", file=sys.stderr)
        return 1

    sample_ann = _read_tsv(sample_ann_path)
    sample_ann = sample_ann.rename(columns={"SAMPLE_ID": "sample_id"})

    output_dir.mkdir(parents=True, exist_ok=True)

    if cohort_scores_path.exists():
        cohort = _read_tsv(cohort_scores_path)
        merged = cohort.merge(sample_ann, on="sample_id", how="left")
        out_path = output_dir / "cohort_scores_annotated.tsv"
        merged.to_csv(out_path, sep="\t", index=False)
        print(f"Wrote {out_path}")

    if calibration_path.exists():
        cal = _read_tsv(calibration_path)
        merged = cal.merge(sample_ann, on="sample_id", how="left")
        out_path = output_dir / "calibration_summary_annotated.tsv"
        merged.to_csv(out_path, sep="\t", index=False)
        print(f"Wrote {out_path}")

    if embeddings_path.exists():
        emb = _read_tsv(embeddings_path)
        merged = emb.merge(sample_ann, on="sample_id", how="left")
        out_path = output_dir / "sample_embeddings_annotated.tsv"
        merged.to_csv(out_path, sep="\t", index=False)
        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
