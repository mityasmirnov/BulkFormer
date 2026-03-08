"""CLI scaffold for proteomics workflows."""

from __future__ import annotations

import argparse


def register_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the proteomics command group."""
    parser = subparsers.add_parser(
        "proteomics",
        help="Train or run proteomics heads on BulkFormer embeddings.",
        description=(
            "Use frozen BulkFormer transcriptome embeddings for proteomics "
            "training, inference, and residual ranking workflows."
        ),
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("train", "predict"),
        help="Planned workflow mode.",
    )
    parser.add_argument("--input", help="Path to the transcriptome feature table.")
    parser.add_argument("--targets", help="Path to proteomics targets or labels.")
    parser.add_argument("--output-dir", help="Directory for outputs.")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    """Placeholder proteomics entrypoint."""
    mode = args.mode or "train"
    print(
        f"The proteomics `{mode}` workflow scaffold is in place. "
        "Implementation will be added in a follow-up step."
    )
    return 0
