import argparse
from pathlib import Path
import pandas as pd
from easd.runner import RunConfig, run_dataset

FACTORIAL_RUNTIME_CSV = "experiments/factorial_design/designs/factorial_runtime.csv"
FACTORIAL_SCORE_CSV = "experiments/factorial_design/designs/factorial_score.csv"
FACTORIAL_TOPK_QUALITY_CSV = "experiments/factorial_design/designs/factorial_topk_quality.csv"
FACTORIAL_TOPK_QUALITY_CONTROL_CSV = "experiments/factorial_design/designs/factorial_topk_quality_controls.csv"
FACTORIAL_TOPK_QUALITY_MEAN_CSV = "experiments/factorial_design/designs/factorial_topk_mean_score.csv"
FACTORIAL_PROJECTS_LIST = [
    FACTORIAL_RUNTIME_CSV,
    # FACTORIAL_SCORE_CSV,
    # FACTORIAL_TOPK_QUALITY_CSV,
    # FACTORIAL_TOPK_QUALITY_CONTROL_CSV,
    # FACTORIAL_TOPK_QUALITY_MEAN_CSV,
]
EXPERIMENT_NAME_COLUMN = "experiment"
RESPONSE_VAR_COLUMN = "response"
FACTOR_COLUMN = "factor"
POPULATION_SIZE_COLUMN = "population"
DATASET_COLUMN = "dataset"
GENERATIONS_COLUMN = "generations"
RATE_POLICY_COLUMN = "rate_policy"
KSIZE_COLUMN = "ksize"
OUTPUT_DIR = "./experiments/factorial_design/factorial_2k"
OPTIONAL_CONFIG_COLUMNS = {
    "alpha",
    "comparacao",
    "debug_performance",
    "executions",
    "ksize",
    "plot_rank",
    "restart_gen",
    "restart_pop",
    "restart_pct",
    "seed",
    "threshold",
}

arguments_dict = {
    "config": None,
    "filepath": None,
    "time_col": None,
    "event_col": None,
    "output_dir": OUTPUT_DIR,
    "dataset_name": "",
    "seed": None,
    "executions": 15,
    "generations": None,
    "population": None,
    "restart_gen": 5,
    "restart_pop": 5,
    "restart_pct": 10,
    "comparacao": "complement",
    "alpha": 0.10,
    "ksize": 10,
    "plot_rank": 10,
    "threshold": 0.90,
    "debug_performance": "off",
    "rate_policy": None,
}


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def config_from_args(p_args: dict) -> RunConfig:
    if p_args["label_col"] == "":
        p_args["label_col"] = None
    return RunConfig(
        filepath=Path(p_args["filepath"]),
        time_col=p_args["time_col"],
        label_col=p_args["label_col"],
        event_col=p_args["event_col"],
        output_dir=Path(p_args["output_dir"]),
        dataset_name=p_args["dataset_name"],
        seed=p_args["seed"],
        executions=p_args["executions"],
        generations=p_args["generations"],
        population=p_args["population"],
        restart_gen=p_args["restart_gen"],
        restart_pop=p_args["restart_pop"],
        restart_pct=p_args["restart_pct"],
        comparacao=p_args["comparacao"],
        alpha=p_args["alpha"],
        ksize=p_args["ksize"],
        plot_rank=p_args["plot_rank"],
        threshold=p_args["threshold"],
        debug_performance=as_bool(p_args["debug_performance"]),
        rate_policy=p_args["rate_policy"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MEASE 2^k factorial designs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--designs", nargs="+", type=Path, default=[Path(path) for path in FACTORIAL_PROJECTS_LIST])
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--executions", type=int, default=arguments_dict["executions"])
    return parser


def row_value(experiment: pd.Series, column: str, default):
    if column not in experiment.index or pd.isna(experiment[column]):
        return default
    return experiment[column]


def run_factorial_designs(designs: list[Path], output_dir: str, executions: int) -> None:
    for project in designs:
        exp_df = pd.read_csv(project)
        response_var = exp_df.loc[0, RESPONSE_VAR_COLUMN]

        for _, experiment in exp_df.iterrows():
            run_arguments = dict(arguments_dict)
            dataset_name, file_path, time_column, event_column, label_column = str(experiment[DATASET_COLUMN]).split(
                "|"
            )
            run_arguments["config"] = experiment[FACTOR_COLUMN]
            run_arguments["filepath"] = file_path
            run_arguments["time_col"] = time_column
            run_arguments["event_col"] = event_column
            run_arguments["label_col"] = label_column
            run_arguments["dataset_name"] = dataset_name
            run_arguments["output_dir"] = f"{output_dir}/{response_var}/{experiment[FACTOR_COLUMN]}"
            run_arguments["executions"] = executions

            run_arguments[POPULATION_SIZE_COLUMN] = experiment[POPULATION_SIZE_COLUMN]
            run_arguments[GENERATIONS_COLUMN] = experiment[GENERATIONS_COLUMN]
            run_arguments[RATE_POLICY_COLUMN] = experiment[RATE_POLICY_COLUMN]

            for column in OPTIONAL_CONFIG_COLUMNS:
                run_arguments[column] = row_value(experiment, column, run_arguments[column])

            configs = config_from_args(run_arguments)
            run_dataset(configs)


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_factorial_designs(args.designs, args.output_dir, args.executions)
