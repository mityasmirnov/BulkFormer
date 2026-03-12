"""CLI for benchmark harness."""

from __future__ import annotations

import argparse
from pathlib import Path

from bulkformer_dx.benchmark.runner import run_benchmark_smoke


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register benchmark command group."""
    parser = subparsers.add_parser(
        "benchmark",
        help="Run anomaly detection benchmarks.",
        description="Benchmark harness for comparing anomaly scoring methods.",
    )
    bench_sub = parser.add_subparsers(dest="benchmark_command", metavar="BENCHMARK_COMMAND")

    run_parser = bench_sub.add_parser("run", help="Run a single benchmark (smoke test).")
    run_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for benchmark outputs.",
    )
    run_parser.add_argument(
        "--n-samples",
        type=int,
        default=30,
        help="Number of synthetic samples. Default 30.",
    )
    run_parser.add_argument(
        "--n-genes",
        type=int,
        default=200,
        help="Number of synthetic genes. Default 200.",
    )
    run_parser.add_argument(
        "--n-inject",
        type=int,
        default=15,
        help="Number of injected outliers. Default 15.",
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed. Default 0.",
    )
    run_parser.set_defaults(func=run_benchmark)

    grid_parser = bench_sub.add_parser("grid-run", help="Run a grid of methods from config.")
    grid_parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML/JSON config with dataset and methods.",
    )
    grid_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for aggregated benchmark outputs.",
    )
    grid_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed. Default 0.",
    )
    grid_parser.set_defaults(func=run_grid)


def run_benchmark(args: argparse.Namespace) -> int:
    """Execute benchmark run."""
    result = run_benchmark_smoke(
        Path(args.output_dir),
        n_samples=args.n_samples,
        n_genes=args.n_genes,
        n_inject=args.n_inject,
        seed=args.seed,
    )
    print(f"Benchmark outputs written to {result['output_dir']}")
    print(f"Metrics: {result['metrics']}")
    return 0


def run_grid(args: argparse.Namespace) -> int:
    """Execute benchmark grid run."""
    from bulkformer_dx.benchmark.runner import grid_run
    result = grid_run(
        Path(args.config),
        Path(args.output_dir),
        seed=args.seed,
    )
    print(f"Grid outputs written to {result['output_dir']}")
    return 0
