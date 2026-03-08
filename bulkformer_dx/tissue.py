"""CLI scaffold for tissue validation workflows."""

from __future__ import annotations

import argparse


def register_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the tissue validation command group."""
    parser = subparsers.add_parser(
        "tissue",
        help="Train or run tissue validation models.",
        description=(
            "Train tissue classifiers from BulkFormer embeddings or predict "
            "tissue labels with serialized sklearn artifacts."
        ),
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("train", "predict"),
        help="Planned workflow mode.",
    )
    parser.add_argument("--input", help="Path to the embedding or sample input table.")
    parser.add_argument("--output-dir", help="Directory for model artifacts or predictions.")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    """Placeholder tissue validation entrypoint."""
    mode = args.mode or "train"
    print(
        f"The tissue `{mode}` workflow scaffold is in place. "
        "Implementation will be added in a follow-up step."
    )
    return 0
