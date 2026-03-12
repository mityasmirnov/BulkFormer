"""Command-line entrypoint for BulkFormer diagnostics workflows."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from . import anomaly, benchmark, embeddings, preprocess, predict_cli, proteomics, tissue


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser for the diagnostics toolkit.

    Notes:
    - This function is intentionally the single place where top-level
      subcommands are registered so `--help` output remains discoverable.
    - Each submodule owns its own argument contract via `register_parser`.
    """
    parser = argparse.ArgumentParser(
        prog="bulkformer-dx",
        description=(
            "BulkFormer diagnostics toolkit for RNA preprocessing, anomaly "
            "ranking/calibration, tissue validation, and frozen-backbone "
            "proteomics prediction."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    preprocess.register_parser(subparsers)
    predict_cli.register_parser(subparsers)
    anomaly.register_parser(subparsers)
    benchmark.register_parser(subparsers)
    embeddings.register_parser(subparsers)
    tissue.register_parser(subparsers)
    proteomics.register_parser(subparsers)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and dispatch to the selected subcommand.

    The dispatch contract is simple: subparsers set `args.func` to a callable
    that accepts the parsed namespace and returns an optional integer exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Subcommands register a handler callback on the namespace. If no
    # subcommand is provided, print root help and exit cleanly.
    handler = getattr(args, "func", None)
    if handler is None:
        parser.print_help()
        return 0

    return int(handler(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
