from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

DEFAULT_OUTPUT_DIR = Path("experiments/artificial_datsets/generated")
DEFAULT_DATASET_NAME = "synthetic_survival"

VARY_NONE = "none"
VARY_FEATURES = "features"
VARY_CENSORING = "censoring"
VARY_SUBGROUP_SIZE = "subgroup_size"
VARY_SUBJECTS = "subjects"
VARY_HAZARD_RATIO = "hazard_ratio"
VARY_CHOICES = (
    VARY_NONE,
    VARY_FEATURES,
    VARY_CENSORING,
    VARY_SUBGROUP_SIZE,
    VARY_SUBJECTS,
    VARY_HAZARD_RATIO,
)

DEFAULT_VARIATION_VALUES: dict[str, list[int | float]] = {
    VARY_FEATURES: [10, 100, 1000],
    VARY_CENSORING: [0.0, 0.3, 0.6, 0.9],
    VARY_SUBGROUP_SIZE: [0.1, 0.2, 0.3, 0.4],
    VARY_SUBJECTS: [1024, 4096, 16384],
    VARY_HAZARD_RATIO: [0.2, 1.0, 1.8],
}


@dataclass(frozen=True)
class SyntheticSurvivalConfig:
    n: int = 1000
    p: int = 10
    k: int = 2
    subgroup_ratio: float = 0.10
    censoring_ratio: float = 0.10
    population_scale: float = 5.0
    subgroup_scale: float = 1.0
    population_shape: float = 1.5
    subgroup_shape: float = 1.5
    seed: int | None = None
    vary: str = VARY_NONE
    vary_value: int | float | None = None
    repeat: int = 0
    hazard_ratio: float | None = None


def make_survival_data(
    config: SyntheticSurvivalConfig,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate a synthetic survival dataset with one planted hyper-box subgroup."""
    _validate_config(config)

    feature_columns = [f"feature_{i}" for i in range(config.p)]
    features = rng.uniform(0.0, 1.0, size=(config.n, config.p))

    subg_feature_indices = np.sort(rng.choice(config.p, size=config.k, replace=False))
    epsilon = config.subgroup_ratio ** (1.0 / config.k)
    lower_bounds = rng.uniform(0.0, 1.0 - epsilon, size=config.k)
    upper_bounds = lower_bounds + epsilon

    subgroup_mask = rng.random(config.n) < config.subgroup_ratio
    for rule_pos, feature_idx in enumerate(subg_feature_indices):
        lower = float(lower_bounds[rule_pos])
        upper = float(upper_bounds[rule_pos])

        features[subgroup_mask, feature_idx] = rng.uniform(lower, upper, size=int(subgroup_mask.sum()))
        features[~subgroup_mask, feature_idx] = _sample_uniform_outside_interval(
            rng,
            lower=lower,
            upper=upper,
            size=int((~subgroup_mask).sum()),
        )

    true_time = _sample_true_times(config, subgroup_mask, rng)
    event = rng.binomial(1, 1.0 - config.censoring_ratio, size=config.n).astype(int)
    observed_time = true_time.copy()
    censored_mask = event == 0
    observed_time[censored_mask] = rng.uniform(0.0, true_time[censored_mask])
    observed_time = np.maximum(observed_time, np.finfo(float).tiny)

    df = pd.DataFrame(features, columns=feature_columns)
    df["time"] = observed_time
    df["event"] = event
    df["subgroup"] = subgroup_mask.astype(int)

    metadata = _build_metadata(
        config=config,
        true_feature_indices=subg_feature_indices,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        epsilon=epsilon,
        df=df,
    )

    return df, metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic survival datasets with a planted hyper-box subgroup.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated files.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME, help="Base name for generated datasets.")
    parser.add_argument("--seed", type=int, default=None, help="Base seed for reproducible generation.")
    parser.add_argument("--n", type=int, default=1000, help="Number of subjects.")
    parser.add_argument("--p", type=int, default=10, help="Number of covariates.")
    parser.add_argument("--k", type=int, default=2, help="Number of true conditions/features in the planted subgroup.")
    parser.add_argument(
        "--subgroup-ratio", type=float, default=0.10, help="Target percentage of subjects in subgroup."
    )
    parser.add_argument("--censoring-ratio", type=float, default=0.10, help="Target percentage of censored subjects.")
    parser.add_argument("--population-scale", type=float, default=5.0, help="Weibull scale for non-subgroup subjects.")
    parser.add_argument("--subgroup-scale", type=float, default=1.0, help="Weibull scale for subgroup subjects.")
    parser.add_argument("--population-shape", type=float, default=1.5, help="Weibull shape for non-subgroup subjects.")
    parser.add_argument("--subgroup-shape", type=float, default=1.5, help="Weibull shape for subgroup subjects.")
    parser.add_argument("--vary", choices=VARY_CHOICES, default=VARY_NONE, help="Dataset aspect varied one at a time.")
    parser.add_argument("--values", nargs="*", default=None, help="Values for the selected variation.")
    parser.add_argument("--repeats", type=int, default=1, help="Independent replicas per configuration.")
    return parser


def config_from_args(args: argparse.Namespace) -> SyntheticSurvivalConfig:
    return SyntheticSurvivalConfig(
        n=args.n,
        p=args.p,
        k=args.k,
        subgroup_ratio=args.subgroup_ratio,
        censoring_ratio=args.censoring_ratio,
        population_scale=args.population_scale,
        subgroup_scale=args.subgroup_scale,
        population_shape=args.population_shape,
        subgroup_shape=args.subgroup_shape,
    )


def generate_and_save(
    base_config: SyntheticSurvivalConfig,
    *,
    output_dir: Path,
    dataset_name: str,
    vary: str,
    values: Sequence[str] | None,
    repeats: int,
    seed: int | None,
) -> list[tuple[Path, Path]]:
    configs = list(_expand_configs(base_config, vary=vary, values=values, repeats=repeats, seed=seed))
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_paths: list[tuple[Path, Path]] = []
    for config in configs:
        rng = np.random.default_rng(config.seed)
        df, metadata = make_survival_data(config, rng)
        output_stem = _output_stem(dataset_name, config, total_outputs=len(configs))
        metadata["dataset_name"] = output_stem
        parquet_path, metadata_path = _save_artifacts(df, metadata, output_dir, output_stem)
        generated_paths.append((parquet_path, metadata_path))

    return generated_paths


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_config = config_from_args(args)

    try:
        generated_paths = generate_and_save(
            base_config,
            output_dir=args.output_dir,
            dataset_name=args.dataset_name,
            vary=args.vary,
            values=args.values,
            repeats=args.repeats,
            seed=args.seed,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    for parquet_path, metadata_path in generated_paths:
        print(f"Saved {parquet_path} and {metadata_path}")
    return 0


def _validate_config(config: SyntheticSurvivalConfig) -> None:
    if config.n <= 0:
        raise ValueError("--n must be greater than zero.")
    if config.p <= 0:
        raise ValueError("--p must be greater than zero.")
    if config.k <= 0:
        raise ValueError("--k must be greater than zero.")
    if config.k > config.p:
        raise ValueError("--k cannot be greater than --p.")
    if not 0.0 < config.subgroup_ratio < 1.0:
        raise ValueError("--subgroup-ratio must be greater than 0 and lower than 1.")
    if not 0.0 <= config.censoring_ratio <= 1.0:
        raise ValueError("--censoring-ratio must be between 0 and 1.")
    if config.population_scale <= 0.0:
        raise ValueError("--population-scale must be greater than zero.")
    if config.subgroup_scale <= 0.0:
        raise ValueError("--subgroup-scale must be greater than zero.")
    if config.population_shape <= 0.0:
        raise ValueError("--population-shape must be greater than zero.")
    if config.subgroup_shape <= 0.0:
        raise ValueError("--subgroup-shape must be greater than zero.")


def _sample_uniform_outside_interval(
    rng: np.random.Generator,
    *,
    lower: float,
    upper: float,
    size: int,
) -> np.ndarray:
    if size == 0:
        return np.empty(0, dtype=float)

    left_width = lower
    right_width = 1.0 - upper
    complement_width = left_width + right_width
    if complement_width <= 0.0:
        raise ValueError("Cannot sample non-subgroup features outside an interval covering [0, 1].")

    choose_left = rng.random(size) < (left_width / complement_width)
    values = np.empty(size, dtype=float)

    left_count = int(choose_left.sum())
    right_count = size - left_count
    if left_count:
        values[choose_left] = rng.uniform(0.0, lower, size=left_count)
    if right_count:
        values[~choose_left] = rng.uniform(upper, 1.0, size=right_count)

    return values


def _sample_true_times(
    config: SyntheticSurvivalConfig,
    subgroup_mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    subgroup_times = rng.weibull(config.subgroup_shape, size=config.n) * config.subgroup_scale
    population_times = rng.weibull(config.population_shape, size=config.n) * config.population_scale
    true_time = np.where(subgroup_mask, subgroup_times, population_times)

    return np.maximum(true_time, np.finfo(float).tiny)


def _build_metadata(
    *,
    config: SyntheticSurvivalConfig,
    true_feature_indices: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    epsilon: float,
    df: pd.DataFrame,
) -> dict[str, Any]:
    intervals = [
        {
            "feature": f"feature_{int(feature_idx)}",
            "feature_index": int(feature_idx),
            "lower": float(lower),
            "upper": float(upper),
        }
        for feature_idx, lower, upper in zip(true_feature_indices, lower_bounds, upper_bounds)
    ]
    rule = " and ".join(f"{item['feature']} in [{item['lower']:.6f}, {item['upper']:.6f}]" for item in intervals)

    return {
        "generator": "experiments/artificial_datsets/main.py",
        "article_reference": {
            "title": "Learning and Naming Subgroups with Exceptional Survival Characteristics",
            "url": "https://arxiv.org/abs/2602.22179",
            "method": "Section 5.1 and Algorithm 5 synthetic survival data generator",
        },
        "ground_truth_warning": (
            "The 'subgroup' column is the planted ground-truth label. "
            "Using it as a covariate when running MEASE leaks the answer."
        ),
        "parameters": asdict(config),
        "seed": config.seed,
        "variation": {
            "factor": config.vary,
            "value": config.vary_value,
            "repeat": config.repeat,
        },
        "columns": {
            "time": "time",
            "event": "event",
            "ground_truth_subgroup": "subgroup",
            "features": [f"feature_{i}" for i in range(config.p)],
        },
        "true_rule": {
            "epsilon": float(epsilon),
            "features": [int(idx) for idx in true_feature_indices],
            "intervals": intervals,
            "rule": rule,
        },
        "actual": {
            "subgroup_ratio": float(df["subgroup"].mean()),
            "censoring_ratio": float((df["event"] == 0).mean()),
            "events_ratio": float((df["event"] == 1).mean()),
            "n": int(len(df)),
            "p": int(config.p),
        },
    }


def _expand_configs(
    base_config: SyntheticSurvivalConfig,
    *,
    vary: str,
    values: Sequence[str] | None,
    repeats: int,
    seed: int | None,
) -> list[SyntheticSurvivalConfig]:
    if repeats <= 0:
        raise ValueError("--repeats must be greater than zero.")

    base_seed = seed if seed is not None else 42
    variation_values = _variation_values(vary, values)

    configs: list[SyntheticSurvivalConfig] = []
    output_index = 0
    for value in variation_values:
        varied_config = _apply_variation(base_config, vary, value)
        for repeat_idx in range(repeats):
            configs.append(
                replace(
                    varied_config,
                    seed=base_seed + output_index,
                    vary=vary,
                    vary_value=value,
                    repeat=repeat_idx,
                )
            )
            output_index += 1

    return configs


def _variation_values(vary: str, values: Sequence[str] | None):
    if vary == VARY_NONE:
        if values:
            raise ValueError("--values can only be used when --vary is not 'none'.")
        return [None]

    raw_values: Sequence[str | int | float] = values if values else DEFAULT_VARIATION_VALUES[vary]
    if vary in {VARY_FEATURES, VARY_SUBJECTS}:
        return [_parse_positive_int(value, vary) for value in raw_values]

    parsed_values = [float(value) for value in raw_values]
    if vary == VARY_HAZARD_RATIO and any(value <= 0.0 for value in parsed_values):
        raise ValueError("hazard_ratio values must be greater than zero.")

    return parsed_values


def _parse_positive_int(value: str | int | float, vary: str) -> int:
    numeric_value = float(value)
    int_value = int(numeric_value)
    if int_value <= 0 or numeric_value != int_value:
        raise ValueError(f"{vary} values must be positive integers.")

    return int_value


def _apply_variation(
    config: SyntheticSurvivalConfig,
    vary: str,
    value: int | float | None,
) -> SyntheticSurvivalConfig:
    if vary == VARY_NONE or value is None:
        return config
    if vary == VARY_FEATURES:
        return replace(config, p=int(value))
    if vary == VARY_CENSORING:
        return replace(config, censoring_ratio=float(value))
    if vary == VARY_SUBGROUP_SIZE:
        return replace(config, subgroup_ratio=float(value))
    if vary == VARY_SUBJECTS:
        return replace(config, n=int(value))
    if vary == VARY_HAZARD_RATIO:
        hazard_ratio = float(value)
        subgroup_scale = config.population_scale * hazard_ratio ** (1.0 / config.population_shape)
        return replace(config, subgroup_scale=subgroup_scale, hazard_ratio=hazard_ratio)

    raise ValueError(f"Unsupported variation: {vary}")


def _output_stem(dataset_name: str, config: SyntheticSurvivalConfig, *, total_outputs: int) -> str:
    if total_outputs == 1:
        return dataset_name
    if config.vary == VARY_NONE:
        return f"{dataset_name}_rep{config.repeat:02d}"
    return f"{dataset_name}_{config.vary}-{_format_value(config.vary_value)}_rep{config.repeat:02d}"


def _format_value(value: int | float | None) -> str:
    if value is None:
        return "base"
    if isinstance(value, int):
        return str(value)
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _save_artifacts(
    df: pd.DataFrame,
    metadata: dict[str, Any],
    output_dir: Path,
    output_stem: str,
) -> tuple[Path, Path]:
    parquet_path = output_dir / f"{output_stem}.parquet"
    metadata_path = output_dir / f"{output_stem}_metadata.json"

    metadata["files"] = {
        "parquet": parquet_path.name,
        "metadata": metadata_path.name,
    }

    df.to_parquet(parquet_path, engine="pyarrow", index=False)
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return parquet_path, metadata_path


if __name__ == "__main__":
    raise SystemExit(main())
