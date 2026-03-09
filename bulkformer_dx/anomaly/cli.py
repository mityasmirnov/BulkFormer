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
        aliases=["train-head"],
        help="Train a small anomaly head on frozen BulkFormer embeddings.",
    )
    head_parser.add_argument(
        "--input",
        "--train-table",
        dest="input",
        required=True,
        help="Path to the BulkFormer-aligned sample-by-gene input table.",
    )
    head_parser.add_argument(
        "--valid-gene-mask",
        required=True,
        help="Path to the valid_gene_mask.tsv file emitted by preprocessing.",
    )
    head_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for trained head checkpoints and metrics.",
    )
    head_parser.add_argument(
        "--mode",
        choices=head.SUPPORTED_HEAD_MODES,
        default=head.DEFAULT_HEAD_MODE,
        help="Head objective to train. Sigma/NLL is the recommended default baseline.",
    )
    head_parser.add_argument(
        "--variant",
        help="BulkFormer model variant to load, such as 37M or 147M.",
    )
    head_parser.add_argument(
        "--checkpoint-path",
        help="Optional path to a specific BulkFormer checkpoint.",
    )
    head_parser.add_argument(
        "--graph-path",
        default=None,
        help="Optional path to the BulkFormer graph asset.",
    )
    head_parser.add_argument(
        "--graph-weights-path",
        default=None,
        help="Optional path to the BulkFormer graph weights asset.",
    )
    head_parser.add_argument(
        "--gene-embedding-path",
        default=None,
        help="Optional path to the BulkFormer ESM2 gene embedding asset.",
    )
    head_parser.add_argument(
        "--gene-info-path",
        default=None,
        help="Optional path to the BulkFormer gene info table.",
    )
    head_parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for feature extraction and head training.",
    )
    head_parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size used for frozen BulkFormer feature extraction and head training.",
    )
    head_parser.add_argument(
        "--hidden-dim",
        type=int,
        default=head.DEFAULT_HIDDEN_DIM,
        help="Hidden dimension for the small anomaly head MLP.",
    )
    head_parser.add_argument(
        "--epochs",
        type=int,
        default=head.DEFAULT_EPOCHS,
        help="Number of optimization epochs for head training.",
    )
    head_parser.add_argument(
        "--learning-rate",
        type=float,
        default=head.DEFAULT_LEARNING_RATE,
        help="Learning rate for AdamW during head training.",
    )
    head_parser.add_argument(
        "--weight-decay",
        type=float,
        default=head.DEFAULT_WEIGHT_DECAY,
        help="Weight decay applied during head training.",
    )
    head_parser.add_argument(
        "--min-sigma",
        type=float,
        default=head.DEFAULT_MIN_SIGMA,
        help="Minimum sigma clamp for the sigma/NLL head.",
    )
    head_parser.add_argument(
        "--injection-rate",
        type=float,
        default=head.DEFAULT_INJECTION_RATE,
        help="Fraction of valid sample-gene positions to perturb in injected-outlier mode.",
    )
    head_parser.add_argument(
        "--outlier-scale",
        type=float,
        default=head.DEFAULT_OUTLIER_SCALE,
        help="Per-gene standard-deviation multiplier used for synthetic outliers.",
    )
    head_parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed for synthetic labels and head training.",
    )
    head_parser.set_defaults(func=head.run)

    calibrate_parser = anomaly_subparsers.add_parser(
        "calibrate",
        help="Calibrate ranked anomaly scores across the cohort.",
    )
    calibrate_parser.add_argument(
        "--scores",
        required=True,
        help="Path to an anomaly score output directory or ranked_genes directory.",
    )
    calibrate_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for calibrated ranked tables and cohort summaries.",
    )
    calibrate_parser.add_argument(
        "--count-space-method",
        choices=calibration.SUPPORTED_COUNT_SPACE_METHODS,
        default=calibration.DEFAULT_COUNT_SPACE_METHOD,
        help=(
            "Optional count-space support path. 'nb_approx' adds a TPM-derived negative-binomial "
            "approximation and is clearly labeled as approximate."
        ),
    )
    calibrate_parser.set_defaults(func=calibration.run)

    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    """Fallback handler when no anomaly subcommand is selected."""
    print(
        "The anomaly command group is scaffolded. "
        "Choose one of: score, head, calibrate."
    )
    return 0
