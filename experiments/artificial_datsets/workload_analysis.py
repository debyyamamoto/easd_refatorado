from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DATASET_PATTERN = "synthetic_survival_subgroup_size-0p*_rep00"
METRICS_RE = re.compile(r"^(?P<dataset>.+)_(?P<execution>\d+)_(?P<baseline>[^_]+)_RulesMetricsResult\.csv$")

PRIMARY_METRICS = [
    "max_f1_score",
    "mean_rule_score",
    "exceptionality",
    "total_time",
    "length",
    "sgCov",
    "setCov",
    "model redundancy",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze repeated MEASE workload-characterization runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--metadata-dir", type=Path, default=Path("experiments/artificial_datsets/generated"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/artificial_datsets/analysis"))
    parser.add_argument("--article-img-dir", type=Path, default=Path("meqt_article/imgs/workload"))
    parser.add_argument("--baseline", default="complement")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level for tests and CIs.")
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=7)
    return parser


def read_metadata(metadata_dir: Path, dataset_name: str) -> dict:
    metadata_file = metadata_dir / f"{dataset_name}_metadata.json"
    with metadata_file.open() as file:
        return json.load(file)


def parse_execution_file(path: Path, baseline: str) -> tuple[str, int] | None:
    match = METRICS_RE.match(path.name)
    if match is None or match.group("baseline") != baseline:
        return None
    return match.group("dataset"), int(match.group("execution"))


def collect_rows(results_dir: Path, metadata_dir: Path, baseline: str) -> pd.DataFrame:
    rows = []

    for dataset_dir in sorted(results_dir.glob(DATASET_PATTERN)):
        run_dir = dataset_dir / baseline
        if not run_dir.exists():
            continue

        dataset_name = dataset_dir.name
        metadata = read_metadata(metadata_dir, dataset_name)
        target_ratio = float(metadata["parameters"]["subgroup_ratio"])
        actual_ratio = float(metadata["actual"]["subgroup_ratio"])

        for metrics_file in sorted(run_dir.glob(f"*_{baseline}_RulesMetricsResult.csv")):
            parsed = parse_execution_file(metrics_file, baseline)
            if parsed is None:
                continue

            parsed_dataset, execution = parsed
            if parsed_dataset != dataset_name:
                continue

            metrics = pd.read_csv(metrics_file).iloc[0].to_dict()
            info_file = run_dir / f"{dataset_name}_{execution}_{baseline}_Info.csv"
            info = pd.read_csv(info_file).iloc[0].to_dict() if info_file.exists() else {}

            row = {
                "dataset": dataset_name,
                "target_subgroup_ratio": target_ratio,
                "actual_subgroup_ratio": actual_ratio,
                "execution": execution,
                "baseline": baseline,
            }
            row.update(metrics)
            row.update({"total_time": info.get("total_time", math.nan), "mean_size": info.get("mean_size", math.nan)})
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No individual workload-characterization runs were found.")

    numeric_columns = [column for column in df.columns if column not in {"dataset", "baseline"}]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    return df.sort_values(["target_subgroup_ratio", "execution"]).reset_index(drop=True)


def bootstrap_mean_interval(
    values: pd.Series,
    alpha: float,
    rng: np.random.Generator,
    n_resamples: int,
) -> tuple[float, float, float, float]:
    clean = values.dropna().to_numpy(dtype=float)
    if len(clean) < 2:
        return math.nan, math.nan, math.nan, math.nan

    sample_indices = rng.integers(0, len(clean), size=(n_resamples, len(clean)))
    bootstrap_means = clean[sample_indices].mean(axis=1)
    mean = float(clean.mean())
    ci_low, ci_high = np.quantile(bootstrap_means, [alpha / 2, 1 - alpha / 2])
    return float(ci_low), float(ci_high), mean - float(ci_low), float(ci_high) - mean


def build_summary(df: pd.DataFrame, alpha: float, bootstrap_resamples: int, bootstrap_seed: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(bootstrap_seed)
    metrics = [metric for metric in PRIMARY_METRICS if metric in df.columns]
    for (dataset, target_ratio, actual_ratio), group in df.groupby(
        ["dataset", "target_subgroup_ratio", "actual_subgroup_ratio"], sort=True
    ):
        for metric in metrics:
            values = group[metric].dropna()
            ci_low, ci_high, ci_low_error, ci_high_error = bootstrap_mean_interval(
                values,
                alpha,
                rng,
                bootstrap_resamples,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "target_subgroup_ratio": target_ratio,
                    "actual_subgroup_ratio": actual_ratio,
                    "metric": metric,
                    "n": int(values.size),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if values.size > 1 else math.nan,
                    "sem": float(stats.sem(values)) if values.size > 1 else math.nan,
                    "ci_method": "bootstrap_percentile_mean",
                    "bootstrap_resamples": bootstrap_resamples,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "ci95_low_error": ci_low_error,
                    "ci95_high_error": ci_high_error,
                    "ci95_margin": max(ci_low_error, ci_high_error),
                    "median": float(values.median()),
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [math.nan] * len(p_values)
    finite = [(idx, p) for idx, p in enumerate(p_values) if math.isfinite(p)]
    finite_sorted = sorted(finite, key=lambda item: item[1])
    running_max = 0.0
    m = len(finite_sorted)
    for rank, (idx, p_value) in enumerate(finite_sorted, start=1):
        adjusted_value = min(1.0, (m - rank + 1) * p_value)
        running_max = max(running_max, adjusted_value)
        adjusted[idx] = running_max
    return adjusted


def build_omnibus_tests(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rows = []
    metrics = [metric for metric in ["max_f1_score", "mean_rule_score"] if metric in df.columns]
    for metric in metrics:
        groups = [group[metric].dropna().to_numpy(dtype=float) for _, group in df.groupby("target_subgroup_ratio")]
        kruskal = stats.kruskal(*groups)
        rows.append(
            {
                "metric": metric,
                "test": "kruskal_wallis",
                "statistic": float(kruskal.statistic),
                "p_value": float(kruskal.pvalue),
                "reject_h0": bool(kruskal.pvalue < alpha),
            }
        )
    return pd.DataFrame(rows)


def paired_metric_values(df: pd.DataFrame, metric: str, ratio_a: float, ratio_b: float) -> pd.DataFrame:
    left = (
        df.loc[df["target_subgroup_ratio"] == ratio_a, ["execution", metric]]
        .dropna()
        .rename(columns={metric: "value_a"})
    )
    right = (
        df.loc[df["target_subgroup_ratio"] == ratio_b, ["execution", metric]]
        .dropna()
        .rename(columns={metric: "value_b"})
    )
    return left.merge(right, on="execution", how="inner").sort_values("execution")


def wilcoxon_signed_rank(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    differences = left - right
    if len(differences) == 0:
        return math.nan, math.nan
    if np.allclose(differences, 0.0):
        return 0.0, 1.0
    test = stats.wilcoxon(left, right, alternative="two-sided", zero_method="wilcox", method="auto")
    return float(test.statistic), float(test.pvalue)


def build_pairwise_tests(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rows = []
    metrics = [metric for metric in ["max_f1_score", "mean_rule_score"] if metric in df.columns]
    ratios = sorted(df["target_subgroup_ratio"].unique())
    for metric in metrics:
        metric_rows = []
        for i, ratio_a in enumerate(ratios):
            for ratio_b in ratios[i + 1 :]:
                paired = paired_metric_values(df, metric, ratio_a, ratio_b)
                a = paired["value_a"].to_numpy(dtype=float)
                b = paired["value_b"].to_numpy(dtype=float)
                statistic, p_value = wilcoxon_signed_rank(a, b)
                metric_rows.append(
                    {
                        "metric": metric,
                        "test": "wilcoxon_signed_rank_paired",
                        "ratio_a": ratio_a,
                        "ratio_b": ratio_b,
                        "n_pairs": int(len(paired)),
                        "mean_a": float(a.mean()),
                        "mean_b": float(b.mean()),
                        "mean_diff_a_minus_b": float(a.mean() - b.mean()),
                        "median_diff_a_minus_b": float(np.median(a - b)),
                        "wilcoxon_w": statistic,
                        "p_value": p_value,
                    }
                )
        adjusted_wilcoxon = holm_adjust([row["p_value"] for row in metric_rows])
        for row, adjusted_p in zip(metric_rows, adjusted_wilcoxon, strict=True):
            row["p_value_holm"] = adjusted_p
            row["reject_h0_holm"] = bool(adjusted_p < alpha)
            rows.append(row)
    return pd.DataFrame(rows)


def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sorted_values = np.sort(values)
    y = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    return sorted_values, y


def plot_f1_cdf(df: pd.DataFrame, output_dirs: list[Path]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for ratio, group in df.groupby("target_subgroup_ratio", sort=True):
        x, y = empirical_cdf(group["max_f1_score"].dropna().to_numpy(dtype=float))
        actual = group["actual_subgroup_ratio"].iloc[0]
        ax.step(x, y, where="post", linewidth=2, label=f"alvo={ratio:.1f}; real={actual:.3f}")
    ax.set_xlabel("Maior F1 por execucao")
    ax.set_ylabel("CDF empirica")
    ax.set_title("Caracterizacao de carga: distribuicao da recuperacao por F1")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    save_plot(fig, output_dirs, "workload_f1_cdf")


def plot_f1_histogram(df: pd.DataFrame, output_dirs: list[Path]) -> None:
    ratios = sorted(df["target_subgroup_ratio"].unique())
    bins = np.linspace(0.0, 1.0, 11)
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.8), sharex=True, sharey=True)

    for ax, ratio in zip(axes.ravel(), ratios, strict=True):
        group = df.loc[df["target_subgroup_ratio"] == ratio]
        actual = group["actual_subgroup_ratio"].iloc[0]
        values = group["max_f1_score"].dropna().to_numpy(dtype=float)
        ax.hist(values, bins=bins, color="#4C78A8", edgecolor="white", alpha=0.82)
        ax.axvline(values.mean(), color="#D62728", linestyle="--", linewidth=1.8, label="media")
        ax.set_title(f"alvo={ratio:.1f}; real={actual:.3f}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)

    for ax in axes[-1, :]:
        ax.set_xlabel("Maior F1 por execucao")
    for ax in axes[:, 0]:
        ax.set_ylabel("Frequencia")

    fig.suptitle("Histogramas do F1 maximo por carga sintetica", y=0.98)
    fig.tight_layout()
    save_plot(fig, output_dirs, "workload_f1_histogram")


def plot_score_cdf(df: pd.DataFrame, output_dirs: list[Path]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for ratio, group in df.groupby("target_subgroup_ratio", sort=True):
        values = df.loc[df["target_subgroup_ratio"] == ratio, "mean_rule_score"].dropna().to_numpy(dtype=float)
        x, y = empirical_cdf(values)
        actual = group["actual_subgroup_ratio"].iloc[0]
        ax.step(x, y, where="post", linewidth=2, label=f"alvo={ratio:.1f}; real={actual:.3f}")
    ax.set_xlabel("Score medio das regras Top-K")
    ax.set_ylabel("CDF empirica")
    ax.set_title("Caracterizacao de carga: distribuicao do score medio do Top-K")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    save_plot(fig, output_dirs, "workload_mean_rule_score_cdf")


def plot_f1_by_dataset(summary: pd.DataFrame, df: pd.DataFrame, output_dirs: list[Path]) -> None:
    metric_summary = summary[summary["metric"] == "max_f1_score"].sort_values("target_subgroup_ratio")
    labels = [f"{ratio:.1f}" for ratio in metric_summary["target_subgroup_ratio"]]
    grouped = [
        df.loc[df["target_subgroup_ratio"] == ratio, "max_f1_score"].dropna().to_numpy(dtype=float)
        for ratio in metric_summary["target_subgroup_ratio"]
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.boxplot(grouped, tick_labels=labels, showmeans=True)
    ax.set_xlabel("Proporcao alvo do subgrupo plantado")
    ax.set_ylabel("Maior F1 por execucao")
    ax.set_title("Maior F1 por carga sintetica")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_plot(fig, output_dirs, "workload_f1_by_dataset")


def save_plot(fig: plt.Figure, output_dirs: list[Path], stem: str) -> None:
    for output_dir in output_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"{stem}.png", dpi=250, bbox_inches="tight")
        fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = collect_rows(args.results_dir, args.metadata_dir, args.baseline)
    summary = build_summary(runs, args.alpha, args.bootstrap_resamples, args.bootstrap_seed)
    omnibus = build_omnibus_tests(runs, args.alpha)
    pairwise = build_pairwise_tests(runs, args.alpha)

    runs.to_csv(args.output_dir / "workload_runs.csv", index=False)
    summary.to_csv(args.output_dir / "workload_summary.csv", index=False)
    omnibus.to_csv(args.output_dir / "workload_omnibus_tests.csv", index=False)
    pairwise.to_csv(args.output_dir / "workload_pairwise_tests.csv", index=False)

    output_dirs = [args.output_dir]
    if args.article_img_dir is not None:
        output_dirs.append(args.article_img_dir)
    plot_f1_cdf(runs, output_dirs)
    plot_f1_histogram(runs, output_dirs)
    plot_score_cdf(runs, output_dirs)
    plot_f1_by_dataset(summary, runs, output_dirs)

    print(f"Collected {len(runs)} runs from {runs['dataset'].nunique()} datasets.")
    print(f"Saved analysis to {args.output_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
