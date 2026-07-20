from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

EXPERIMENTS_DIR = Path(__file__).resolve().parent
DEFAULT_MEASE_RESULTS = EXPERIMENTS_DIR / "resultados_complement.csv"
DEFAULT_BASELINE_RESULTS = EXPERIMENTS_DIR / "results_esmam/metrics_baseline-complement.csv"
DEFAULT_OUTPUT = EXPERIMENTS_DIR / "wilcoxon.csv"

METRICS = {
    "exceptionality": True,
    "length": False,
    "sgCov": True,
    "setCov": True,
    "description redundancy": False,
    "coverage redundancy": False,
    "CR": False,
    "model redundancy": False,
}
METRIC_ALIAS = {
    "coverage redundancy": "cover redundancy",
}
DATASETS = ["carcinoma", "breast-cancer", "cancer", "lung", "mgus2", "veteran"]
RIVALS = ["EsmamDS-cpm", "Esmam-cpm", "BS-EMM-cpm", "BS-SD-cpm", "LR-Rules"]


def run_wilcoxon(
    mease_results: Path = DEFAULT_MEASE_RESULTS,
    baseline_results: Path = DEFAULT_BASELINE_RESULTS,
    output_file: Path = DEFAULT_OUTPUT,
) -> pd.DataFrame:
    df_mease = pd.read_csv(mease_results)
    df_base = pd.read_csv(baseline_results, header=[0, 1], index_col=[0, 1])
    report = []

    for dataset in DATASETS:
        df_mease_ds = df_mease[df_mease["Dataset"] == dataset]

        for metric, higher_is_better in METRICS.items():
            baseline_metric = METRIC_ALIAS.get(metric, metric)
            mease_values = df_mease_ds[metric].values

            for rival in RIVALS:
                try:
                    rival_values = df_base.loc[dataset, (baseline_metric, rival)].values
                except KeyError:
                    print(f"Missing metric '{baseline_metric}' / rival '{rival}' for '{dataset}'. Skipping.")
                    continue

                limit = min(len(mease_values), len(rival_values))
                v1 = np.array(mease_values[:limit])
                v2 = np.array(rival_values[:limit])

                mean1, std1, median1 = np.mean(v1), np.std(v1), np.median(v1)
                mean2, std2, median2 = np.mean(v2), np.std(v2), np.median(v2)

                try:
                    _, p_value = wilcoxon(v1, v2, zero_method="wilcox", alternative="two-sided")
                    p_value = float(p_value)
                    significant = p_value < 0.05
                except Exception:
                    p_value = 1.0
                    significant = False

                if significant:
                    result = (
                        "better"
                        if (higher_is_better and mean1 > mean2) or (not higher_is_better and mean1 < mean2)
                        else "worse"
                    )
                else:
                    result = "tie"

                report.append(
                    {
                        "Dataset": dataset,
                        "Metric": metric,
                        "Rival": rival,
                        "Mean MEASE": round(mean1, 4),
                        "Std MEASE": round(std1, 4),
                        "Median MEASE": round(median1, 4),
                        "Mean Rival": round(mean2, 4),
                        "Std Rival": round(std2, 4),
                        "Median Rival": round(median2, 4),
                        "p-value": round(p_value, 4),
                        "Result": result,
                    }
                )

    df_final = pd.DataFrame(report)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_file, index=False)
    print(df_final.to_string(index=False))
    print(f"\nSaved to {output_file}")

    print("\n=== Summary: MEASE vs each rival across all datasets and metrics ===")
    print(df_final.groupby(["Rival", "Result"]).size().unstack(fill_value=0))
    return df_final


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Wilcoxon signed-rank tests for the MEASE paper metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mease_results", type=Path, default=DEFAULT_MEASE_RESULTS)
    parser.add_argument("--baseline_results", type=Path, default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--output_file", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_wilcoxon(args.mease_results, args.baseline_results, args.output_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
