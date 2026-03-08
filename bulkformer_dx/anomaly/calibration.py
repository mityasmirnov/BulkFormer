"""Placeholder anomaly calibration workflow."""

from __future__ import annotations

import argparse


def run(args: argparse.Namespace) -> int:
    """Placeholder anomaly calibration entrypoint."""
    print(
        "The anomaly calibration scaffold is in place. "
        "Empirical residual calibration will be added in a follow-up step."
    )
    if args.output_dir:
        print(f"Requested output directory: {args.output_dir}")
    return 0
