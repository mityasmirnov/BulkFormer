"""Placeholder anomaly head workflow."""

from __future__ import annotations

import argparse


def run(args: argparse.Namespace) -> int:
    """Placeholder anomaly head entrypoint."""
    print(
        "The anomaly head scaffold is in place. "
        "Sigma/NLL head training will be added in a follow-up step."
    )
    if args.output_dir:
        print(f"Requested output directory: {args.output_dir}")
    return 0
