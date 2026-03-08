"""Placeholder anomaly scoring workflow."""

from __future__ import annotations

import argparse


def run(args: argparse.Namespace) -> int:
    """Placeholder anomaly scoring entrypoint."""
    print(
        "The anomaly scoring scaffold is in place. "
        "Monte Carlo masking and residual ranking will be added in a follow-up step."
    )
    if args.output_dir:
        print(f"Requested output directory: {args.output_dir}")
    return 0
