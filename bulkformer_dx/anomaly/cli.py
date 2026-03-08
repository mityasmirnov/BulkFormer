"""CLI scaffold for anomaly analysis workflows."""

from __future__ import annotations

import argparse

from . import calibration, head, scoring


def register_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the anomaly command group and placeholder subcommands."""
    parser = subparsers.add_parser(
        "anomaly",
        help="Run anomaly scoring and calibration workflows.",
        description=(
            "Anomaly workflow entrypoint for ranking, uncertainty head "
            "training, and cohort calibration."
        ),
    )
    anomaly_subparsers = parser.add_subparsers(dest="anomaly_command", metavar="ANOMALY_COMMAND")

    score_parser = anomaly_subparsers.add_parser(
        "score",
        help="Planned residual ranking workflow.",
    )
    score_parser.add_argument("--input", help="Path to aligned sample inputs.")
    score_parser.add_argument("--output-dir", help="Directory for anomaly reports.")
    score_parser.set_defaults(func=scoring.run)

    head_parser = anomaly_subparsers.add_parser(
        "head",
        help="Planned anomaly head training workflow.",
    )
    head_parser.add_argument("--train-table", help="Path to a training manifest or table.")
    head_parser.add_argument("--output-dir", help="Directory for head checkpoints.")
    head_parser.set_defaults(func=head.run)

    calibrate_parser = anomaly_subparsers.add_parser(
        "calibrate",
        help="Planned cohort calibration workflow.",
    )
    calibrate_parser.add_argument("--scores", help="Path to anomaly scores.")
    calibrate_parser.add_argument("--output-dir", help="Directory for calibration outputs.")
    calibrate_parser.set_defaults(func=calibration.run)

    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    """Fallback handler when no anomaly subcommand is selected."""
    print(
        "The anomaly command group is scaffolded. "
        "Choose one of: score, head, calibrate."
    )
    return 0
