"""CLI entrypoints for anomaly analysis workflows."""

from __future__ import annotations

import argparse

from . import calibration, head, scoring


def register_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the anomaly command group and its subcommands."""
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
        help="Run Monte Carlo masking residual ranking.",
    )
    score_parser.add_argument(
        "--input",
        required=True,
        help="Path to the BulkFormer-aligned sample-by-gene input table.",
    )
    score_parser.add_argument(
        "--valid-gene-mask",
        required=True,
        help="Path to the valid_gene_mask.tsv file emitted by preprocessing.",
    )
    score_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for cohort scores, ranked genes, and QC summaries.",
    )
    score_parser.add_argument(
        "--variant",
        help="BulkFormer model variant to load, such as 37M or 147M.",
    )
    score_parser.add_argument(
        "--checkpoint-path",
        help="Optional path to a specific BulkFormer checkpoint.",
    )
    score_parser.add_argument(
        "--graph-path",
        default=None,
        help="Optional path to the BulkFormer graph asset.",
    )
    score_parser.add_argument(
        "--graph-weights-path",
        default=None,
        help="Optional path to the BulkFormer graph weights asset.",
    )
    score_parser.add_argument(
        "--gene-embedding-path",
        default=None,
        help="Optional path to the BulkFormer ESM2 gene embedding asset.",
    )
    score_parser.add_argument(
        "--gene-info-path",
        default=None,
        help="Optional path to the BulkFormer gene info table.",
    )
    score_parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for inference, for example cpu, cuda, or mps.",
    )
    score_parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of masked samples per prediction batch.",
    )
    score_parser.add_argument(
        "--mask-prob",
        type=float,
        default=scoring.DEFAULT_MASK_PROB,
        help="Fraction of valid genes to mask on each Monte Carlo pass.",
    )
    score_parser.add_argument(
        "--mc-passes",
        type=int,
        default=scoring.DEFAULT_MC_PASSES,
        help="Number of Monte Carlo masking passes per sample.",
    )
    score_parser.add_argument(
        "--fill-value",
        type=float,
        default=scoring.MASK_TOKEN_VALUE,
        help="Mask token value used for BulkFormer-compatible masked genes.",
    )
    score_parser.add_argument(
        "--random-seed",
        type=int,
        default=scoring.DEFAULT_RANDOM_SEED,
        help="Random seed used for Monte Carlo mask generation.",
    )
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
