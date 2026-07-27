from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from easd.evaluation import SCORE_METRICS
from easd.runner import RunConfig, run_dataset

DEFAULT_OUTPUT_DIR = Path("experiments/artificial_datsets/score_metric_benchmark")
DEFAULT_DATASET_DIR = DEFAULT_OUTPUT_DIR / "datasets"
DEFAULT_SUBJECTS = [10_000, 50_000]
DEFAULT_METRICS = ["legacy_logrank", "fast_logrank", "km_cvm", "km_abc"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark MEASE score metrics on planted-rule synthetic survival datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--subjects", nargs="+", type=int, default=DEFAULT_SUBJECTS)
    parser.add_argument("--metrics", nargs="+", choices=SCORE_METRICS, default=DEFAULT_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--p", type=int, default=10)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--subgroup-ratio", type=float, default=0.10)
    parser.add_argument("--censoring-ratio", type=float, default=0.10)
    parser.add_argument("--population-scale", type=float, default=5.0)
    parser.add_argument("--subgroup-scale", type=float, default=1.0)
    parser.add_argument("--population-shape", type=float, default=1.5)
    parser.add_argument("--subgroup-shape", type=float, default=1.5)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--population", type=int, default=80)
    parser.add_argument("--ksize", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--baseline", choices=["complement", "population"], default="complement")
    parser.add_argument("--km-time-bins", type=int, default=512)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.dataset_dir.mkdir(parents=True, exist_ok=True)

    generator = _load_generator_module()
    rows: list[dict] = []
    result_path = args.output_dir / "score_metric_benchmark.csv"

    for subject_count in args.subjects:
        for repeat in range(args.repeats):
            dataset_seed = args.seed + repeat + (subject_count * 100)
            dataset_name = f"scorebench_n{subject_count}_rep{repeat:02d}"
            parquet_path, metadata_path = _generate_dataset(
                generator=generator,
                dataset_name=dataset_name,
                output_dir=args.dataset_dir,
                seed=dataset_seed,
                subject_count=subject_count,
                p=args.p,
                k=args.k,
                subgroup_ratio=args.subgroup_ratio,
                censoring_ratio=args.censoring_ratio,
                population_scale=args.population_scale,
                subgroup_scale=args.subgroup_scale,
                population_shape=args.population_shape,
                subgroup_shape=args.subgroup_shape,
            )

            print(f"Generated {parquet_path} and {metadata_path}")
            for score_metric in args.metrics:
                run_name = f"{dataset_name}_{score_metric}"
                print(f"Running {run_name}")
                started = time.perf_counter()
                summary = run_dataset(
                    RunConfig(
                        filepath=parquet_path,
                        time_col="time",
                        event_col="event",
                        label_col="subgroup",
                        output_dir=args.output_dir / "runs",
                        dataset_name=run_name,
                        executions=1,
                        generations=args.generations,
                        population=args.population,
                        restart_gen=args.generations + 1,
                        restart_pop=args.generations + 1,
                        restart_pct=10,
                        comparacao=args.baseline,
                        alpha=args.alpha,
                        ksize=args.ksize,
                        plot_rank=0,
                        threshold=0.9,
                        debug_performance=False,
                        rate_policy="fixed",
                        score_metric=score_metric,
                        km_time_bins=None if args.km_time_bins <= 0 else args.km_time_bins,
                    )
                )
                wall_time = time.perf_counter() - started
                info = _read_info(summary.output_dir, run_name, args.baseline)
                metrics = summary.run_metrics[0] if summary.run_metrics else {}

                row = {
                    "subjects": subject_count,
                    "repeat": repeat,
                    "dataset_seed": dataset_seed,
                    "score_metric": score_metric,
                    "baseline": args.baseline,
                    "generations": args.generations,
                    "population": args.population,
                    "ksize": args.ksize,
                    "km_time_bins": args.km_time_bins,
                    "algorithm_time": info.get("total_time", np.nan),
                    "wall_time": wall_time,
                    "rules_count": info.get("rules_count", np.nan),
                    "best_fitness": info.get("best_fitness", np.nan),
                    "mean_rule_score": metrics.get("mean_rule_score", np.nan),
                    "max_f1_score": metrics.get("max_f1_score", np.nan),
                    "exceptionality": metrics.get("exceptionality", np.nan),
                    "sgCov": metrics.get("sgCov", np.nan),
                    "setCov": metrics.get("setCov", np.nan),
                }
                rows.append(row)
                pd.DataFrame(rows).to_csv(result_path, index=False)

    print(f"Saved benchmark summary to {result_path}")
    return 0


def _load_generator_module():
    module_path = Path(__file__).with_name("main.py")
    spec = importlib.util.spec_from_file_location("synthetic_survival_generator", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load synthetic generator from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _generate_dataset(
    *,
    generator,
    dataset_name: str,
    output_dir: Path,
    seed: int,
    subject_count: int,
    p: int,
    k: int,
    subgroup_ratio: float,
    censoring_ratio: float,
    population_scale: float,
    subgroup_scale: float,
    population_shape: float,
    subgroup_shape: float,
) -> tuple[Path, Path]:
    config = generator.SyntheticSurvivalConfig(
        n=subject_count,
        p=p,
        k=k,
        subgroup_ratio=subgroup_ratio,
        censoring_ratio=censoring_ratio,
        population_scale=population_scale,
        subgroup_scale=subgroup_scale,
        population_shape=population_shape,
        subgroup_shape=subgroup_shape,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    df, metadata = generator.make_survival_data(config, rng)
    metadata["dataset_name"] = dataset_name

    parquet_path = output_dir / f"{dataset_name}.parquet"
    metadata_path = output_dir / f"{dataset_name}_metadata.json"
    df.to_parquet(parquet_path, engine="pyarrow", index=False)
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return parquet_path, metadata_path


def _read_info(output_dir: Path, dataset_name: str, baseline: str) -> dict:
    info_path = output_dir / f"{dataset_name}_28_{baseline}_Info.csv"
    if not info_path.exists():
        return {}

    info = pd.read_csv(info_path)
    if info.empty:
        return {}

    return info.iloc[0].to_dict()


if __name__ == "__main__":
    raise SystemExit(main())
