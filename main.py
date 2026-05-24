from __future__ import annotations

import argparse
import sys
from pathlib import Path

from easd.runner import RunConfig, run_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MEASE on a parquet survival dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("filepath", type=Path, help="Path to a parquet dataset.")
    parser.add_argument("-time", "--time_col", required=True, help="Survival time column name.")
    parser.add_argument("-event", "--event_col", required=True, help="Event/censoring status column name.")
    parser.add_argument("--output_dir", type=Path, default=Path("results"), help="Directory for generated outputs.")
    parser.add_argument("--dataset_name", default=None, help="Optional name used in output files.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducibility.")
    parser.add_argument("-th", "--threshold", type=float, default=0.9, help="Jaccard similarity threshold.")
    parser.add_argument("-g", "--generations", type=int, default=500, help="Maximum number of generations.")
    parser.add_argument("-p", "--population", type=int, default=500, help="Population size.")
    parser.add_argument("--restart_gen", type=int, default=3, help="Generation limit without improvement.")
    parser.add_argument("--restart_pop", type=int, default=3, help="Population restart limit.")
    parser.add_argument("--restart_pct", type=int, default=10, help="Population percentage restarted each time.")
    parser.add_argument(
        "-comp",
        "--comparacao",
        choices=["complement", "population"],
        default="complement",
        help="Baseline group for the log-rank test.",
    )
    parser.add_argument("-a", "--alpha", type=float, default=0.5, help="Fitness alpha weight.")
    parser.add_argument("-exe", "--executions", type=int, default=1, help="Independent algorithm executions.")
    parser.add_argument("-k", "--ksize", type=int, default=10, help="Top-K rule rank size.")
    parser.add_argument(
        "-plt",
        "--plt_rank",
        type=int,
        default=0,
        help="Save plots for the top-N rules. Only enabled for a single execution.",
    )
    parser.add_argument(
        "-d",
        "--debug_performance",
        choices=["on", "off"],
        default="off",
        help="Collect CPU/RAM measurements.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        filepath=args.filepath,
        time_col=args.time_col,
        event_col=args.event_col,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        seed=args.seed,
        executions=args.executions,
        generations=args.generations,
        population=args.population,
        restart_gen=args.restart_gen,
        restart_pop=args.restart_pop,
        restart_pct=args.restart_pct,
        comparacao=args.comparacao,
        alpha=args.alpha,
        ksize=args.ksize,
        plot_rank=args.plt_rank,
        threshold=args.threshold,
        debug_performance=args.debug_performance == "on",
    )


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = run_dataset(config_from_args(args))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Results saved to {summary.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
